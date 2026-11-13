"""Authorization / ACL — entitlement filter builders.

ACL is enforced as a PRE-FILTER inside pgvector and OpenSearch (and inside
SQL params for ops tools) so unauthorized documents never leave the store.
Rule: a chunk is visible iff
    chunk.entitlements.roles ∋ user.role
AND (chunk.entitlements.clients == "*" OR chunk.client_id ∈ user.client_ids)

Author: venkateshwara doijode
"""

from dataclasses import dataclass
from typing import Any

from app.security.authn import Principal


def is_wildcard_entitlement(entitlements: dict) -> bool:
    """Canonical wildcard check shared by every store and the oracle.

    A document is wildcard-visible when ``entitlements['clients']`` is the
    scalar ``"*"`` *or* a list containing ``"*"`` — both the pg ``?`` text
    operator and the OpenSearch ``entitlement_wildcard`` field are derived
    from this one rule so the stores can never drift apart.
    """
    clients = entitlements.get("clients")
    return clients == "*" or (isinstance(clients, list) and "*" in clients)


@dataclass(frozen=True)
class ACLFilter:
    """A principal's entitlements rendered into per-store filter clauses.

    Produces the equivalent SQL ``WHERE`` fragment (pgvector), OpenSearch
    bool filter, and an in-memory predicate so the same ACL rule is applied
    identically in every store.
    """

    client_ids: tuple[str, ...]
    role: str

    def pg_where(self, focus_client: str | None = None) -> tuple[str, dict[str, Any]]:
        """SQL fragment for the chunks table. Wildcard docs have
        entitlements->'clients' == '"*"'; client docs must match an entitled id.
        focus_client narrows to that client + wildcard docs."""
        params = {"acl_role": self.role}
        if focus_client:
            params["acl_focus"] = focus_client
            client_clause = "client_id = :acl_focus"
        else:
            params.update({f"acl_c{i}": c for i, c in enumerate(self.client_ids)})
            placeholders = ", ".join(f":acl_c{i}" for i in range(len(self.client_ids)))
            client_clause = f"client_id IN ({placeholders})" if placeholders else "FALSE"
        sql = (
            f"(entitlements->'clients' ? '*' OR {client_clause}) "
            "AND entitlements->'roles' ? :acl_role"
        )
        return sql, params

    def os_filter(self, focus_client: str | None = None) -> dict:
        """OpenSearch bool filter applied as a query-time filter clause.
        focus_client narrows to that client + wildcard docs."""
        client_terms = (
            [{"term": {"client_id": focus_client}}]
            if focus_client
            else [{"terms": {"client_id": list(self.client_ids)}}]
        )
        return {
            "bool": {
                "filter": [
                    {"term": {"entitlement_roles": self.role}},
                    {
                        "bool": {
                            "should": [
                                {"term": {"entitlement_wildcard": True}},
                                *client_terms,
                            ],
                            "minimum_should_match": 1,
                        }
                    },
                ]
            }
        }

    def allows(self, entitlements: dict, client_id: str) -> bool:
        """In-memory check used by tests and ops-tool result validation.

        This is the oracle for the pgvector/OpenSearch parity tests — it
        must encode exactly the same rule both stores render.
        """
        if self.role not in entitlements.get("roles", []):
            return False
        return is_wildcard_entitlement(entitlements) or client_id in self.client_ids


def acl_for(principal: Principal) -> ACLFilter:
    """Build the ``ACLFilter`` for a principal's client ids and role."""
    return ACLFilter(client_ids=principal.client_ids, role=principal.role)
