"""Synchronous SQLAlchemy engines — rag DB (rw) and ops DB (read-only role).

Author: venkateshwara doijode
"""

from functools import lru_cache

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine

from app.core.config import get_settings


@lru_cache
def rag_engine() -> Engine:
    """Return the cached read-write engine for the rag database."""
    return create_engine(get_settings().database_url, pool_pre_ping=True)


@lru_cache
def ops_engine() -> Engine:
    """Read-only engine for ops tools. Connects as rag_readonly; a session-level
    statement_timeout is applied as defense-in-depth."""
    settings = get_settings()
    engine = create_engine(settings.ops_database_url, pool_pre_ping=True)

    @event.listens_for(engine, "connect")
    def _set_readonly_session(dbapi_conn, _):
        """Force each new ops-DB connection into read-only mode with a statement timeout."""
        cur = dbapi_conn.cursor()
        cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION READ ONLY")
        cur.execute(f"SET statement_timeout = {settings.ops_statement_timeout_ms}")
        cur.close()

    return engine
