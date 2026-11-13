"""Trade system store — parameterized, read-only, entitlement-scoped.

Implements the ``TradesStore`` protocol with pagination, freshness
timestamps (fresh_at + max(trade_date) as source freshness), explicit
authz, latency metrics, and structured errors. Module functions delegate
to a default instance for back-compat.

Author: venkateshwara doijode
"""

import time

from app.security.authn import Principal
from app.tools.ops.client import (
    assert_entitled_rows,
    assert_ops_allowed,
    client_clause,
    max_source_ts,
    run_query,
    utc_now,
)
from app.tools.ops.schemas import (
    OpsError,
    OpsErrorCode,
    OpsResult,
    StoreHealth,
    TradeRow,
    TradesQuery,
    TradeStatus,
    TradeType,
)

_COLS = "trade_id, client_id, type, status, amount, currency, trade_date"


class PostgresTradesStore:
    """Postgres implementation of the trades read interface."""

    store = "trades"

    def get(self, query: TradesQuery, principal: Principal, conn=None) -> OpsResult[TradeRow]:
        """Fetch a trade by id or run a filtered, paginated trade search.

        ``trade_id`` set → point lookup. Otherwise optional client/status/
        type/date-range filters are ANDed with the entitlement clause; rows
        come back newest-first with ``has_more`` pagination.
        """
        started = time.monotonic()
        try:
            assert_ops_allowed(principal, self.store)
            clause, params = client_clause(principal)
            filters = [clause]
            if query.trade_id:
                filters.append("trade_id = :tid")
                params["tid"] = query.trade_id
            if query.client_id:
                filters.append("client_id = :cid")
                params["cid"] = query.client_id
            if query.status:
                filters.append("status = :status")
                params["status"] = query.status
            if query.trade_type:
                filters.append("type = :ttype")
                params["ttype"] = query.trade_type
            if query.start_date:
                filters.append("trade_date >= :start_d")
                params["start_d"] = query.start_date
            if query.end_date:
                filters.append("trade_date <= :end_d")
                params["end_d"] = query.end_date
            params["cap"] = query.limit + 1
            params["off"] = query.offset
            sql = f"""
                SELECT {_COLS} FROM trades
                WHERE {" AND ".join(filters)}
                ORDER BY trade_date DESC
                LIMIT :cap OFFSET :off"""
            rows = run_query(sql, params, conn=conn)
            assert_entitled_rows(rows, principal)
        except Exception as exc:
            return self._err(exc, started)
        has_more = len(rows) > query.limit
        rows = rows[: query.limit]
        return OpsResult[TradeRow](
            rows=[TradeRow(**r) for r in rows],
            has_more=has_more,
            fresh_at=utc_now(),
            source_updated_at=max_source_ts(rows, "trade_date"),
            store=self.store,
            latency_ms=(time.monotonic() - started) * 1000,
        )

    def healthcheck(self) -> StoreHealth:
        """SELECT 1 round-trip — surfaced by /ready and the ops-health endpoint."""
        started = time.monotonic()
        try:
            run_query("SELECT 1 AS ok", {})
            return StoreHealth(
                store=self.store,
                ok=True,
                latency_ms=(time.monotonic() - started) * 1000,
            )
        except Exception as exc:
            return StoreHealth(store=self.store, ok=False, error=type(exc).__name__)

    def _err(self, exc: Exception, started: float) -> OpsResult[TradeRow]:
        """Map an exception to a structured OpsResult error."""
        code: OpsErrorCode = (
            "unauthorized"
            if getattr(exc, "code", "") in ("unauthorized", "forbidden")
            else "store_unavailable"
        )
        return OpsResult[TradeRow](
            store=self.store,
            latency_ms=(time.monotonic() - started) * 1000,
            error=OpsError(code=code, message=type(exc).__name__, detail=str(exc)[:200]),
        )


_default = PostgresTradesStore()


def search_trades(
    principal: Principal,
    client_id: str | None = None,
    status: TradeStatus | None = None,
    trade_type: TradeType | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """Back-compat shim → ``PostgresTradesStore().get`` search."""
    result = _default.get(
        TradesQuery(
            client_id=client_id,
            status=status,
            trade_type=trade_type,
            start_date=start_date,
            end_date=end_date,
        ),
        principal,
    )
    return [r.model_dump() for r in result.rows]


def get_trade(trade_id: str, principal: Principal) -> list[dict]:
    """Back-compat shim → ``PostgresTradesStore().get`` point lookup."""
    result = _default.get(TradesQuery(trade_id=trade_id), principal)
    return [r.model_dump() for r in result.rows]
