"""Ops-tool schema + read-only SQL gate unit tests.

Verifies the static write-keyword assertion rejects anything but SELECT,
the explicit authz gate denies role-less/scope-less principals, and the
readonly-role seed grants are SELECT-only at the SQL level.

Author: venkateshwara doijode
"""

import re
from pathlib import Path

import pytest

from app.core.errors import AppError
from app.security.authn import Principal
from app.tools.ops.client import assert_ops_allowed, assert_readonly_sql


def test_readonly_sql_allows_select():
    """SELECT and WITH..SELECT statements pass the gate."""
    assert_readonly_sql("SELECT 1")
    assert_readonly_sql("  select * from accounts")
    assert_readonly_sql("WITH t AS (SELECT 1) SELECT * FROM t")


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO accounts VALUES (1)",
        "UPDATE accounts SET tier='x'",
        "DELETE FROM trades",
        "DROP TABLE tickets",
        "TRUNCATE accounts",
        "GRANT SELECT ON accounts TO public",
        "SELECT 1; DELETE FROM accounts",
        "COPY accounts FROM '/tmp/x.csv'",
    ],
)
def test_readonly_sql_rejects_writes(sql: str):
    """Write/mutation statements — including those hidden after a SELECT — fail."""
    with pytest.raises(AppError):
        assert_readonly_sql(sql)


def test_ops_allowed_requires_role_and_scope():
    """Principals with a bad role or empty client scope are denied."""
    with pytest.raises(AppError):
        assert_ops_allowed(Principal("u1", "contractor", ("acme",)), "accounts")
    with pytest.raises(AppError):
        assert_ops_allowed(Principal("u1", "agent", ()), "accounts")
    # agent with scope passes
    assert_ops_allowed(Principal("u1", "agent", ("acme",)), "accounts")


def test_readonly_role_seed_is_select_only():
    """Statically verify db/seed/readonly_role.sql grants SELECT and nothing more."""
    sql = (Path(__file__).parents[2] / "db" / "seed" / "readonly_role.sql").read_text()
    grants = re.findall(r"GRANT\s+(\w+)", sql, re.IGNORECASE)
    # every GRANT statement must be SELECT-only (CONNECT/USAGE are schema-level, fine)
    table_grants = [g for g in grants if g.upper() not in ("CONNECT", "USAGE")]
    assert table_grants and all(g.upper() == "SELECT" for g in table_grants)
    assert not re.search(r"GRANT\s+(INSERT|UPDATE|DELETE|TRUNCATE|ALL)", sql, re.IGNORECASE)
