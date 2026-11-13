"""SQL tool arg binding — verifies parameterized clauses, no string concat
of user input into SQL."""

from app.security.authn import Principal
from app.tools.ops.client import client_clause


def test_client_clause_placeholders():
    p = Principal("u1", "agent", ("acme", "bluepeak"))
    clause, params = client_clause(p)
    assert "client_id IN (:ent_c0, :ent_c1)" in clause
    assert params == {"ent_c0": "acme", "ent_c1": "bluepeak"}
    # no raw client values inside the SQL fragment
    assert "acme" not in clause


def test_client_clause_empty():
    p = Principal("u1", "agent", ())
    clause, params = client_clause(p)
    assert clause == "FALSE"
    assert params == {}
