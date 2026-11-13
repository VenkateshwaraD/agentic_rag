"""Authorization / ACL — entitlement filter builders.

ACL is enforced as a PRE-FILTER inside pgvector and OpenSearch (and inside
SQL params for ops tools) so unauthorized documents never leave the store.
Rule: a chunk is visible iff
    chunk.entitlements.roles ∋ user.role
AND (chunk.entitlements.clients == "*" OR chunk.client_id ∈ user.client_ids)
"""

from dataclasses import dataclass
from typing import Any

from app.security.authn import Principal


@dataclass(frozen=True)
class ACLFilter:
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
            f"(entitlements->'clients' ? '\"*\"' OR {client_clause}) "
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
        """In-memory check used by tests and ops-tool result validation."""
        if self.role not in entitlements.get("roles", []):
            return False
        clients = entitlements.get("clients", [])
        return clients == "*" or client_id in self.client_ids


def acl_for(principal: Principal) -> ACLFilter:
    return ACLFilter(client_ids=principal.client_ids, role=principal.role)
