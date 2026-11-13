"""Shared read-only ops query runner.

Defense-in-depth for read-only access:
  - dedicated rag_readonly DB role (SELECT-only grants)
  - session-level READ ONLY transaction mode + statement_timeout (session.py)
  - static write-keyword assertion on every statement template
  - fixed parameterized query templates only — no free-text SQL path exists
  - hard row cap applied to every query
  - pooled engine (session.py); a shared snapshot context keeps multi-store
    reads consistent

Author: venkateshwara doijode
"""

import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import text

from app.core.config import get_settings
from app.core.constants import ALL_ROLES
from app.core.errors import AppError
from app.core.logging import get_logger
from app.db.session import ops_engine
from app.security.authn import Principal

log = get_logger(__name__)

_WRITE_TOKENS = re.compile(
    r"\b(insert|update|delete|drop|alter|truncate|grant|revoke|copy|call|execute)\b",
    re.IGNORECASE,
)
_SELECT_ONLY = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def assert_readonly_sql(sql: str) -> None:
    """Reject any statement that isn't a bare SELECT — last gate before exec.

    The static templates are trusted, but this check runs unconditionally so
    a future dynamic-SQL bug can't turn the read path writable.
    """
    if not _SELECT_ONLY.match(sql) or _WRITE_TOKENS.search(sql):
        raise AppError(
            "ops queries must be read-only SELECT statements",
            code="forbidden_statement",
        )


def assert_ops_allowed(principal: Principal, domain: str) -> None:
    """Explicit per-call authorization gate for every ops lookup.

    Beyond the row-level entitlement clause: the principal must carry a
    recognised role and a non-empty client scope — a principal with no
    entitled clients sees nothing rather than querying unfiltered.
    """
    if principal.role not in ALL_ROLES:
        raise AppError(f"role {principal.role!r} cannot read {domain}", code="unauthorized")
    if not principal.client_ids:
        raise AppError("principal has no entitled client scope", code="unauthorized")


def assert_entitled_rows(rows: list[dict[str, Any]], principal: Principal) -> None:
    """Post-filter safety: drop any row whose client_id escapes entitlements.

    The SQL clause should already enforce this; the check exists so a query
    bug surfaces as an error instead of a leak.
    """
    leaks = [r.get("client_id") for r in rows if r.get("client_id") not in principal.client_ids]
    if leaks:
        raise AppError(
            "ops row returned outside principal entitlements",
            code="acl_violation",
        )


def run_query(sql: str, params: dict[str, Any], conn=None) -> list[dict[str, Any]]:
    """Execute a fixed parameterized query on the ops read replica.

    Pass ``conn`` from ``ops_snapshot()`` to pin the statement inside a
    shared REPEATABLE-READ snapshot; otherwise a fresh pooled connection is
    used. Emits ``ops.query`` metrics (latency, row count).
    """
    assert_readonly_sql(sql)
    started = time.monotonic()
    if conn is not None:
        rows = [dict(r) for r in conn.execute(text(sql), params).mappings()]
    else:
        with ops_engine().connect() as c:
            rows = [dict(r) for r in c.execute(text(sql), params).mappings()]
    log.info(
        "ops.query",
        latency_ms=round((time.monotonic() - started) * 1000, 1),
        rows=len(rows),
    )
    return rows


def max_source_ts(rows: list[dict[str, Any]], column: str) -> str | None:
    """Return the newest value of a timestamp-ish column (source freshness)."""
    values = [str(r[column]) for r in rows if r.get(column)]
    return max(values) if values else None


@contextmanager
def ops_snapshot() -> Iterator[tuple[Any, datetime]]:
    """Pin a shared REPEATABLE-READ snapshot across multiple store queries.

    Yields ``(conn, snapshot_ts)``: pass ``conn`` into each store's ``get()``
    and stamp ``snapshot_ts`` on every result so callers know the rows share
    one consistent view of the operational data.
    """
    conn = ops_engine().connect()
    ts = datetime.now(UTC)
    try:
        conn.execute(text("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ"))
        yield conn, ts
    finally:
        conn.close()


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


def utc_now() -> datetime:
    """Timezone-aware UTC now — result freshness stamp."""
    return datetime.now(UTC)
