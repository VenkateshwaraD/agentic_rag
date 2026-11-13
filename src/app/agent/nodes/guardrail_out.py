"""Output guardrail node — PII rescan of the draft + final refusal handling.

Author: venkateshwara doijode
"""

from app.agent.nodes.common import Timer, audit
from app.core.constants import (
    ABSTENTION_MESSAGE,
    EV_ABSTAIN,
    EV_ANSWER,
    EV_GUARDRAIL,
    EV_REFUSAL,
    REFUSAL_MESSAGE,
)
from app.security.guardrails import check_output

_REFUSAL = REFUSAL_MESSAGE


def guardrail_out(state):
    """Apply the output guardrail and finalize the reply.

    Rescans a non-refusal, non-abstained draft for PII/policy issues and
    swaps in the refusal message when blocked; already-refused states skip
    the rescan. Abstentions pass through with the abstention message.
    Queues high-impact or ambiguous replies for human review, audits the
    guardrail decision plus the terminal event, and returns the final
    reply, refusal/abstain flags, review flag, and guardrail log.
    """
    refusal = state.get("refusal", False)
    abstained = state.get("abstained", False)
    reply = state.get("reply") or (ABSTENTION_MESSAGE if abstained else _REFUSAL)

    with Timer() as t:
        if not refusal and not abstained:
            res = check_output(reply)
            if not res.allowed:
                refusal, reply = True, _REFUSAL
            decision = {"stage": "output", "allowed": res.allowed, "reason": res.reason}
        else:
            decision = {
                "stage": "output",
                "allowed": False,
                "reason": state.get("refusal_reason") or state.get("abstention_reason", "refusal"),
            }
    audit(state, EV_GUARDRAIL, decision, t.ms)

    terminal = EV_ABSTAIN if abstained else (EV_REFUSAL if refusal else EV_ANSWER)
    audit(
        state,
        terminal,
        {
            "reply": reply,
            "reason": state.get("refusal_reason") or state.get("abstention_reason", ""),
            "answerability": str(state.get("answerability", "")),
        },
    )

    # async human review — flagged replies land in review_queue for later
    # approval; the response is not blocked
    from app.observability import review as review_queue

    pending_review = False
    if not refusal and review_queue.needs_human_review(state):
        pending_review = bool(review_queue.enqueue(state, reply))

    return {
        "reply": reply,
        "refusal": refusal,
        "abstained": abstained,
        "pending_review": pending_review,
        "guardrail_log": state.get("guardrail_log", []) + [decision],
    }
