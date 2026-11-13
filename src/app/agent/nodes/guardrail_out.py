"""Output guardrail node — PII rescan of the draft + final refusal handling."""

from app.agent.nodes.common import Timer, audit
from app.core.constants import EV_ANSWER, EV_GUARDRAIL, EV_REFUSAL
from app.security.guardrails import check_output

_REFUSAL = (
    "I don't have enough entitled information to answer that question. "
    "Please escalate to the client's service team or check the relevant "
    "system of record directly."
)


def guardrail_out(state):
    reply = state.get("reply", _REFUSAL)
    refusal = state.get("refusal", False)

    with Timer() as t:
        if not refusal:
            res = check_output(reply)
            if not res.allowed:
                refusal, reply = True, _REFUSAL
            decision = {"stage": "output", "allowed": res.allowed, "reason": res.reason}
        else:
            decision = {
                "stage": "output",
                "allowed": False,
                "reason": state.get("refusal_reason", "refusal"),
            }
    audit(state, EV_GUARDRAIL, decision, t.ms)
    audit(
        state,
        EV_REFUSAL if refusal else EV_ANSWER,
        {"reply": reply, "reason": state.get("refusal_reason", "")},
    )
    return {
        "reply": reply,
        "refusal": refusal,
        "guardrail_log": state.get("guardrail_log", []) + [decision],
    }
