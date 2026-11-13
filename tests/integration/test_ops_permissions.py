"""SQL-level read-only enforcement — independent of application authz.

Runs against the live ops DB as the rag_readonly role: any write must be
rejected by Postgres itself, proving the role has no write grants even if
application-level checks were bypassed. Skips when the DB is unavailable.

Author: venkateshwara doijode
"""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError


def _ops_conn():
    """Return an open ops-engine connection, or skip when the DB is down."""
    from app.db.session import ops_engine

    try:
        conn = ops_engine().connect()
        conn.execute(text("SELECT 1"))
        return conn
    except Exception as exc:  # noqa: BLE001 — any connect failure → skip
        pytest.skip(f"ops DB unavailable: {exc}")


@pytest.mark.integration
def test_readonly_role_cannot_insert():
    """INSERT as rag_readonly is rejected by Postgres privileges/session mode."""
    with _ops_conn() as conn, pytest.raises(DBAPIError):
        conn.execute(
            text(
                "INSERT INTO accounts (client_id, name, region, tier, owner_team, agreement_doc) "
                "VALUES ('x','x','x','x','x','x')"
            )
        )


@pytest.mark.integration
def test_readonly_role_cannot_update_or_delete():
    """UPDATE and DELETE as rag_readonly are rejected."""
    with _ops_conn() as conn, pytest.raises(DBAPIError):
        conn.execute(text("UPDATE accounts SET tier = 'x' WHERE 1=1"))
    with _ops_conn() as conn, pytest.raises(DBAPIError):
        conn.execute(text("DELETE FROM accounts WHERE 1=1"))


@pytest.mark.integration
def test_readonly_role_cannot_ddl():
    """DDL (CREATE TABLE) as rag_readonly is rejected."""
    with _ops_conn() as conn, pytest.raises(DBAPIError):
        conn.execute(text("CREATE TABLE probe_write (id int)"))
