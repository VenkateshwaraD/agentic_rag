"""Shared helpers for graph nodes: audit logging + prompt loading."""

import time
from pathlib import Path

from app.core.dependencies import get_audit
from app.core.logging import get_logger

log = get_logger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parents[1] / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text()


def audit(state, event_type: str, payload: dict, latency_ms: float = 0.0) -> None:
    get_audit().log_event(
        request_id=state.get("request_id", ""),
        trace_id=state.get("trace_id", ""),
        user_id=state["principal"].user_id,
        event_type=event_type,
        payload=payload,
        latency_ms=latency_ms,
    )


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self.start) * 1000
        return False
