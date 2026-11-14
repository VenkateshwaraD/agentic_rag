"""Ticket system store — parameterized, read-only, entitlement-scoped.

Ticket subjects may contain raw PII (deliberately, for evals) — replies pass
through the output guardrail's PII rescan before reaching the user.

Implements the ``TicketsStore`` protocol with pagination, freshness
timestamps (fresh_at + max(created_at) as source freshness), explicit
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
    TicketCategory,
    TicketRow,
    TicketsQuery,
    TicketStatus,
)

_COLS = "ticket_id, client_id, category, status, subject, resolution, created_at"


class PostgresTicketsStore:
    """Postgres implementation of the tickets read interface."""

    store = "tickets"

    def get(self, query: TicketsQuery, principal: Principal, conn=None) -> OpsResult[TicketRow]:
        """Fetch a ticket by id or run a filtered, paginated ticket search.

        ``ticket_id`` set → point lookup (an unauthorized id returns empty,
        indistinguishable from nonexistent). Otherwise optional client/
        status/category/keyword filters narrow within the entitled scope;
        rows come back newest-first with ``has_more`` pagination.
        """
        started = time.monotonic()
        try:
            assert_ops_allowed(principal, self.store)
            clause, params = client_clause(principal)
            filters = [clause]
            if query.ticket_id:
                filters.append("ticket_id = :tid")
                params["tid"] = query.ticket_id
            if query.client_id:
                filters.append("client_id = :cid")
                params["cid"] = query.client_id
            if query.status:
                filters.append("status = :status")
                params["status"] = query.status
            if query.category:
                filters.append("category = :cat")
                params["cat"] = query.category
            if query.keyword:
                filters.append("subject ILIKE :kw")
                params["kw"] = f"%{query.keyword}%"
            params["cap"] = query.limit + 1
            params["off"] = query.offset
            sql = f"""
                SELECT {_COLS} FROM tickets
                WHERE {" AND ".join(filters)}
                ORDER BY created_at DESC
                LIMIT :cap OFFSET :off"""
            rows = run_query(sql, params, conn=conn)
            assert_entitled_rows(rows, principal)
        except Exception as exc:
            return self._err(exc, started)
        has_more = len(rows) > query.limit
        rows = rows[: query.limit]
        return OpsResult[TicketRow](
            rows=[TicketRow(**r) for r in rows],
            has_more=has_more,
            fresh_at=utc_now(),
            source_updated_at=max_source_ts(rows, "created_at"),
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

    def _err(self, exc: Exception, started: float) -> OpsResult[TicketRow]:
        """Map an exception to a structured OpsResult error."""
        code: OpsErrorCode = (
            "unauthorized"
            if getattr(exc, "code", "") in ("unauthorized", "forbidden")
            else "store_unavailable"
        )
        return OpsResult[TicketRow](
            store=self.store,
            latency_ms=(time.monotonic() - started) * 1000,
            error=OpsError(code=code, message=type(exc).__name__),
        )


_default = PostgresTicketsStore()


def get_ticket(ticket_id: str, principal: Principal) -> list[dict]:
    """Back-compat shim → ``PostgresTicketsStore().get`` point lookup."""
    result = _default.get(TicketsQuery(ticket_id=ticket_id), principal)
    return [r.model_dump() for r in result.rows]


def search_tickets(
    principal: Principal,
    client_id: str | None = None,
    status: TicketStatus | None = None,
    category: TicketCategory | None = None,
    keyword: str | None = None,
) -> list[dict]:
    """Back-compat shim → ``PostgresTicketsStore().get`` search."""
    result = _default.get(
        TicketsQuery(client_id=client_id, status=status, category=category, keyword=keyword),
        principal,
    )
    return [r.model_dump() for r in result.rows]
