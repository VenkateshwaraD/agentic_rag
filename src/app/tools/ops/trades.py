"""Trade system tool — parameterized, read-only, entitlement-scoped.

Author: venkateshwara doijode
"""

from app.security.authn import Principal
from app.tools.ops.client import client_clause, row_cap, run_query


def search_trades(
    principal: Principal,
    client_id: str | None = None,
    status: str | None = None,
    trade_type: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> list[dict]:
    """Search trades with optional client/status/type/date-range filters.

    The entitlement clause is always ANDed in; results are newest-first and
    row-capped.
    """
    clause, params = client_clause(principal)
    filters = [clause]
    if client_id:
        filters.append("client_id = :cid")
        params["cid"] = client_id
    if status:
        filters.append("status = :status")
        params["status"] = status
    if trade_type:
        filters.append("type = :ttype")
        params["ttype"] = trade_type
    if start_date:
        filters.append("trade_date >= :start_d")
        params["start_d"] = start_date
    if end_date:
        filters.append("trade_date <= :end_d")
        params["end_d"] = end_date
    params["cap"] = row_cap()
    sql = f"""
        SELECT trade_id, client_id, type, status, amount, currency, trade_date
        FROM trades
        WHERE {" AND ".join(filters)}
        ORDER BY trade_date DESC
        LIMIT :cap"""
    return run_query(sql, params)


def get_trade(trade_id: str, principal: Principal) -> list[dict]:
    """Fetch one trade by id, but only when its client is in the principal's entitlements."""
    clause, params = client_clause(principal)
    params.update({"tid": trade_id, "cap": 1})
    sql = f"""
        SELECT trade_id, client_id, type, status, amount, currency, trade_date
        FROM trades
        WHERE trade_id = :tid AND {clause}
        LIMIT :cap"""
    return run_query(sql, params)
