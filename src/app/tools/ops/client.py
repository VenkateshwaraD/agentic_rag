"""Shared read-only ops query runner.

Defense-in-depth for read-only access:
  - dedicated rag_readonly DB role (SELECT-only grants)
  - session-level READ ONLY transaction mode + statement_timeout (session.py)
  - fixed parameterized query templates only — no free-text SQL path exists
  - hard row cap applied to every query

Author: venkateshwara doijode
"""

from typing import Any

from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import ops_engine
from app.security.authn import Principal


def run_query(sql: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    """Execute a fixed parameterized query on the ops read replica."""
    with ops_engine().connect() as conn:
        rows = conn.execute(text(sql), params).mappings()
        return [dict(r) for r in rows]


def client_clause(principal: Principal, column: str = "client_id") -> tuple[str, dict]:
    """(sql_fragment, params) restricting rows to the principal's entitled
    client_ids — same pre-filter rule as the doc stores."""
    params = {f"ent_c{i}": c for i, c in enumerate(principal.client_ids)}
    placeholders = ", ".join(f":ent_c{i}" for i in range(len(params)))
    clause = f"{column} IN ({placeholders})" if placeholders else "FALSE"
    return clause, params


def row_cap() -> int:
    """Return the configured max rows an ops query may return."""
    return get_settings().ops_row_cap
