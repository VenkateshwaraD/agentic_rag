"""Account system store — parameterized, read-only, entitlement-scoped.

Implements the ``AccountsStore`` protocol: typed query in → ``OpsResult``
out, with pagination, freshness stamps, latency metrics, explicit authz,
and structured errors. Back-compat module functions delegate to a default
instance for existing callers.

Author: venkateshwara doijode
"""

import time

from app.security.authn import Principal
from app.tools.ops.client import (
    assert_entitled_rows,
    assert_ops_allowed,
    client_clause,
    run_query,
    utc_now,
)
from app.tools.ops.schemas import (
    AccountRow,
    AccountsQuery,
    OpsError,
    OpsErrorCode,
    OpsResult,
    StoreHealth,
)

_COLS = "client_id, name, region, tier, owner_team, agreement_doc"


class PostgresAccountsStore:
    """Postgres implementation of the accounts read interface."""

    store = "accounts"

    def get(self, query: AccountsQuery, principal: Principal, conn=None) -> OpsResult[AccountRow]:
        """Fetch one account or page through entitled accounts.

        ``query.client_id`` set → point lookup (still ACL-scoped). Otherwise
        an ordered offset page over the principal's entitled clients. Fetches
        ``limit+1`` rows to compute ``has_more``.
        """
        started = time.monotonic()
        try:
            assert_ops_allowed(principal, self.store)
            clause, params = client_clause(principal)
            filters = [clause]
            if query.client_id:
                filters.append("client_id = :cid")
                params["cid"] = query.client_id
            params["cap"] = query.limit + 1
            params["off"] = query.offset
            sql = f"""
                SELECT {_COLS} FROM accounts
                WHERE {" AND ".join(filters)}
                ORDER BY client_id
                LIMIT :cap OFFSET :off"""
            rows = run_query(sql, params, conn=conn)
            assert_entitled_rows(rows, principal)
        except Exception as exc:
            return self._err(exc, started)
        has_more = len(rows) > query.limit
        rows = rows[: query.limit]
        return OpsResult[AccountRow](
            rows=[AccountRow(**r) for r in rows],
            has_more=has_more,
            fresh_at=utc_now(),
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

    def _err(self, exc: Exception, started: float) -> OpsResult[AccountRow]:
        """Map an exception to a structured OpsResult error."""
        code: OpsErrorCode = (
            "unauthorized"
            if getattr(exc, "code", "") in ("unauthorized", "forbidden")
            else "store_unavailable"
        )
        return OpsResult[AccountRow](
            store=self.store,
            latency_ms=(time.monotonic() - started) * 1000,
            error=OpsError(code=code, message=type(exc).__name__, detail=str(exc)[:200]),
        )


_default = PostgresAccountsStore()


def get_account(client_id: str, principal: Principal) -> list[dict]:
    """Back-compat shim → ``PostgresAccountsStore().get`` point lookup."""
    result = _default.get(AccountsQuery(client_id=client_id), principal)
    return [r.model_dump() for r in result.rows]


def list_accounts(principal: Principal) -> list[dict]:
    """Back-compat shim → ``PostgresAccountsStore().get`` list call."""
    result = _default.get(AccountsQuery(), principal)
    return [r.model_dump() for r in result.rows]
