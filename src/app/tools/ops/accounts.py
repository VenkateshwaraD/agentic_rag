"""Account system tool — parameterized, read-only, entitlement-scoped.

Author: venkateshwara doijode
"""

from app.security.authn import Principal
from app.tools.ops.client import client_clause, row_cap, run_query

_SQL = """
SELECT client_id, name, region, tier, owner_team, agreement_doc
FROM accounts
WHERE client_id = :cid AND {acl}
LIMIT :cap
"""


def get_account(client_id: str, principal: Principal) -> list[dict]:
    """Fetch one account row, but only when the client is in the principal's entitlements."""
    clause, params = client_clause(principal)
    params.update({"cid": client_id, "cap": row_cap()})
    return run_query(_SQL.format(acl=clause), params)


def list_accounts(principal: Principal) -> list[dict]:
    """List all accounts within the principal's entitled client ids (row-capped)."""
    clause, params = client_clause(principal)
    params["cap"] = row_cap()
    sql = f"""
        SELECT client_id, name, region, tier, owner_team, agreement_doc
        FROM accounts
        WHERE {clause}
        ORDER BY client_id
        LIMIT :cap
    """
    return run_query(sql, params)
