"""Input guardrail node — injection screen + length cap."""

from app.agent.nodes.common import Timer, audit
from app.core.constants import EV_GUARDRAIL
from app.security.guardrails import check_input


def guardrail_in(state):
    q = state["question"]
    with Timer() as t:
        result = check_input(q)
    audit(
        state,
        EV_GUARDRAIL,
        {"stage": "input", "allowed": result.allowed, "reason": result.reason},
        t.ms,
    )
    if not result.allowed:
        return {
            "refusal": True,
            "refusal_reason": result.reason,
            "guardrail_log": [{"stage": "input", "allowed": False, "reason": result.reason}],
        }
    return {"guardrail_log": [{"stage": "input", "allowed": True}]}
