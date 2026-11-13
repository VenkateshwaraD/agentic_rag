"""Ticket system tool — parameterized, read-only, entitlement-scoped.

Ticket subjects may contain raw PII (deliberately, for evals) — replies pass
through the output guardrail's PII rescan before reaching the user.

Author: venkateshwara doijode
"""

from app.security.authn import Principal
from app.tools.ops.client import client_clause, row_cap, run_query


def get_ticket(ticket_id: str, principal: Principal) -> list[dict]:
    """Entitlements enforced in WHERE — an unauthorized ticket_id returns
    empty, indistinguishable from nonexistent."""
    clause, params = client_clause(principal)
    params.update({"tid": ticket_id, "cap": 1})
    sql = f"""
        SELECT ticket_id, client_id, category, status, subject, resolution, created_at
        FROM tickets
        WHERE ticket_id = :tid AND {clause}
        LIMIT :cap"""
    return run_query(sql, params)


def search_tickets(
    principal: Principal,
    client_id: str | None = None,
    status: str | None = None,
    category: str | None = None,
    keyword: str | None = None,
) -> list[dict]:
    """Search tickets with optional client/status/category/keyword filters.

    The entitlement clause is always ANDed in; extra filters narrow within
    the entitled scope. Results are newest-first and row-capped.
    """
    clause, params = client_clause(principal)
    filters = [clause]
    if client_id:
        filters.append("client_id = :cid")
        params["cid"] = client_id
    if status:
        filters.append("status = :status")
        params["status"] = status
    if category:
        filters.append("category = :cat")
        params["cat"] = category
    if keyword:
        filters.append("subject ILIKE :kw")
        params["kw"] = f"%{keyword}%"
    params["cap"] = row_cap()
    sql = f"""
        SELECT ticket_id, client_id, category, status, subject, resolution, created_at
        FROM tickets
        WHERE {" AND ".join(filters)}
        ORDER BY created_at DESC
        LIMIT :cap"""
    return run_query(sql, params)
