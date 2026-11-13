"""Compliance audit log — one audit_events row per query/retrieval/
tool_call/guardrail/agent_step/answer/refusal.

Payloads are PII-redacted before insert. Writes are synchronous (audit must
not silently drop); callers treat failures as logged warnings so an audit
outage degrades observability without taking the copilot down — flagged in
runbooks for production hardening (outbox pattern).

Author: venkateshwara doijode
"""

from typing import Any

from sqlalchemy import text

from app.core.config import Settings
from app.core.dependencies import get_pii_redactor
from app.core.logging import get_logger
from app.db.session import rag_engine

log = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS audit_events (
    id          SERIAL PRIMARY KEY,
    request_id  TEXT NOT NULL,
    trace_id    TEXT DEFAULT '',
    user_id     TEXT NOT NULL,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_type  TEXT NOT NULL,
    payload     JSONB NOT NULL DEFAULT '{}',
    latency_ms  DOUBLE PRECISION DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_audit_request ON audit_events(request_id);
CREATE INDEX IF NOT EXISTS idx_audit_user    ON audit_events(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_ts      ON audit_events(ts);
"""

_INSERT = """
INSERT INTO audit_events (request_id, trace_id, user_id, event_type, payload, latency_ms)
VALUES (:request_id, :trace_id, :user_id, :event_type, CAST(:payload AS jsonb), :latency_ms)
"""


def _redact_payload(value: Any) -> Any:
    """Recursively PII-redact string values in an audit payload."""
    redactor = get_pii_redactor()
    if isinstance(value, str):
        return redactor.redact(value)
    if isinstance(value, dict):
        return {k: _redact_payload(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact_payload(v) for v in value]
    return value


class AuditWriter:
    """Writes PII-redacted audit events to the ``audit_events`` table."""

    def __init__(self, settings: Settings):
        """Store the settings used to locate the rag database."""
        self._settings = settings

    def ensure_table(self) -> None:
        """Create the audit table and its indexes if they don't already exist."""
        with rag_engine().begin() as conn:
            for stmt in _DDL.split(";"):
                if stmt.strip():
                    conn.execute(text(stmt))

    def log_event(
        self,
        request_id: str,
        trace_id: str,
        user_id: str,
        event_type: str,
        payload: dict,
        latency_ms: float = 0.0,
    ) -> None:
        """Insert one PII-redacted audit row; logs a warning instead of raising on failure."""
        import json as _json

        try:
            with rag_engine().begin() as conn:
                conn.execute(
                    text(_INSERT),
                    {
                        "request_id": request_id,
                        "trace_id": trace_id,
                        "user_id": user_id,
                        "event_type": event_type,
                        "payload": _json.dumps(_redact_payload(payload)),
                        "latency_ms": latency_ms,
                    },
                )
        except Exception as exc:
            log.warning("audit.write_failed", event_type=event_type, error=str(exc))

    def events_for_request(self, request_id: str) -> list[dict]:
        """Return every audit event for ``request_id``, ordered by insertion."""
        with rag_engine().begin() as conn:
            rows = conn.execute(
                text(
                    "SELECT request_id, trace_id, user_id, ts, event_type, payload, latency_ms "
                    "FROM audit_events WHERE request_id = :rid ORDER BY id"
                ),
                {"rid": request_id},
            ).mappings()
        return [dict(r) for r in rows]
