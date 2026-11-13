from app.security.authn import Principal
from app.security.authz import acl_for


def _principal(clients=("acme", "bluepeak"), role="agent"):
    return Principal(user_id="u1", role=role, client_ids=clients)


def test_allows_client_scoped():
    acl = acl_for(_principal())
    assert acl.allows({"clients": ["acme"], "roles": ["agent"]}, "acme")
    assert not acl.allows({"clients": ["everfield"], "roles": ["agent"]}, "everfield")


def test_allows_wildcard():
    acl = acl_for(_principal())
    assert acl.allows({"clients": "*", "roles": ["agent"]}, "")


def test_role_denied():
    acl = acl_for(_principal(role="agent"))
    assert not acl.allows({"clients": ["acme"], "roles": ["admin"]}, "acme")


def test_pg_where_binds_all_clients():
    sql, params = acl_for(_principal()).pg_where()
    assert "entitlements->'clients'" in sql
    assert params["acl_role"] == "agent"
    assert params["acl_c0"] == "acme" and params["acl_c1"] == "bluepeak"


def test_os_filter_structure():
    f = acl_for(_principal()).os_filter()
    filters = f["bool"]["filter"]
    assert {"term": {"entitlement_roles": "agent"}} in filters
    terms = filters[1]["bool"]["should"]
    assert {"term": {"entitlement_wildcard": True}} in terms
    assert {"terms": {"client_id": ["acme", "bluepeak"]}} in terms
