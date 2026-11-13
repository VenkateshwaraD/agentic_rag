"""Unit tests for the authorization/ACL filter builders.

Author: venkateshwara doijode
"""

from app.security.authn import Principal
from app.security.authz import acl_for


def _principal(clients=("acme", "bluepeak"), role="agent"):
    """Build a test principal with the given client ids and role."""
    return Principal(user_id="u1", role=role, client_ids=clients)


def test_allows_client_scoped():
    """Verify client-scoped docs are allowed only for entitled clients."""
    acl = acl_for(_principal())
    assert acl.allows({"clients": ["acme"], "roles": ["agent"]}, "acme")
    assert not acl.allows({"clients": ["everfield"], "roles": ["agent"]}, "everfield")


def test_allows_wildcard():
    """Verify wildcard docs are visible to any authenticated principal."""
    acl = acl_for(_principal())
    assert acl.allows({"clients": "*", "roles": ["agent"]}, "")


def test_role_denied():
    """Verify a doc restricted to admins is denied to an agent."""
    acl = acl_for(_principal(role="agent"))
    assert not acl.allows({"clients": ["acme"], "roles": ["admin"]}, "acme")


def test_pg_where_binds_all_clients():
    """Verify the SQL fragment binds every entitled client id plus the role."""
    sql, params = acl_for(_principal()).pg_where()
    assert "entitlements->'clients'" in sql
    # wildcard check must query the literal string '*', not a quoted '"*"'
    assert "? '*'" in sql
    assert params["acl_role"] == "agent"
    assert params["acl_c0"] == "acme" and params["acl_c1"] == "bluepeak"


def test_os_filter_structure():
    """Verify the OpenSearch filter includes role term and client/wildcard shoulds."""
    f = acl_for(_principal()).os_filter()
    filters = f["bool"]["filter"]
    assert {"term": {"entitlement_roles": "agent"}} in filters
    terms = filters[1]["bool"]["should"]
    assert {"term": {"entitlement_wildcard": True}} in terms
    assert {"terms": {"client_id": ["acme", "bluepeak"]}} in terms
