"""Shared helpers for graph nodes: audit logging + prompt loading.

Author: venkateshwara doijode
"""

import time
from pathlib import Path

from app.core.dependencies import get_audit
from app.core.logging import get_logger

log = get_logger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str) -> str:
    """Read and return the prompt template file ``name`` from the prompts dir."""
    return (PROMPTS_DIR / name).read_text()


def audit(state, event_type: str, payload: dict, latency_ms: float = 0.0) -> None:
    """Write an audit event for the current request using ids and user from state."""
    get_audit().log_event(
        request_id=state.get("request_id", ""),
        trace_id=state.get("trace_id", ""),
        user_id=state["principal"].user_id,
        event_type=event_type,
        payload=payload,
        latency_ms=latency_ms,
    )


def add_error(state, message: str) -> list[str]:
    """Return ``state['errors']`` with a PII-sanitized ``message`` appended.

    Errors are surfaced in API diagnostics, so they must never carry raw
    exception payloads containing user or client data — the redactor is
    applied defensively even though exception strings *should* be safe.
    """
    from app.core.dependencies import get_pii_redactor

    errors = list(state.get("errors", []))
    errors.append(get_pii_redactor().redact(str(message))[:300])
    return errors


class Timer:
    """Context manager that measures elapsed wall-clock time in ``self.ms``."""

    def __enter__(self):
        """Record the start time and return the timer instance."""
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        """Compute elapsed milliseconds into ``self.ms``; never suppresses errors."""
        self.ms = (time.perf_counter() - self.start) * 1000
        return False
