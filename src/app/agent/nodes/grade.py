"""Grade node — decides whether retrieved evidence is sufficient.

Heuristic first (non-empty, decent rerank score, or tool rows); if empty and
not yet rewritten, rewrites the query once (adds client context) and the
graph loops back through retrieve. Still empty → downstream refusal.

Author: venkateshwara doijode
"""

from app.agent.nodes.common import Timer, audit
from app.core.constants import EV_AGENT_STEP

_MIN_RERANK_SCORE = -2.0  # BGE logits; negative ~0 is a reasonable relevance bar


def grade(state):
    """Decide whether the retrieved evidence supports an answer.

    Marks evidence sufficient when any doc beats the rerank-score bar or any
    tool returned rows. Otherwise rewrites the question once with client
    context (``retry``) or sets a refusal (``refuse``). Emits an audit event
    and returns the state updates including ``grade_action``.
    """
    retrieved = state.get("retrieved", [])
    tool_results = state.get("tool_results", [])
    has_rows = any(tr["rows"] for tr in tool_results)
    good_docs = [r for r in retrieved if r.get("rerank_score", -99) > _MIN_RERANK_SCORE]

    with Timer() as t:
        sufficient = bool(good_docs or has_rows)
        updates: dict = {}
        if sufficient:
            action = "answer"
        elif not state.get("rewritten"):
            action = "retry"
            cid = state.get("extra", {}).get("client_id")
            updates = {
                "question": state["question"] + (f" client {cid}" if cid else ""),
                "rewritten": True,
            }
        else:
            action = "refuse"
            updates = {
                "refusal": True,
                "refusal_reason": "insufficient evidence in entitled sources",
            }
        updates["grade_action"] = action
    audit(
        state,
        EV_AGENT_STEP,
        {
            "node": "grade",
            "sufficient": sufficient,
            "good_docs": len(good_docs),
            "tool_rows": sum(len(tr["rows"]) for tr in tool_results),
            "action": action,
        },
        t.ms,
    )
    return updates
