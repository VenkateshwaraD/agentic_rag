"""LangGraph assembly — explicit agent state machine.

Flow: guardrail_in → route → retrieve → grade ──┐
        ↑ rewrite (bounded retries) ─────────────┘
        → (answerable) → synthesize → validate ─┐
        ↑ resynthesize once (validator feedback)─┘
        → guardrail_out → END

Every conditional edge also checks the execution budget and the
cooperative cancel flag: hard breaches and cancellations jump straight to
guardrail_out so loops can never run indefinitely.

Author: venkateshwara doijode
"""

from typing import cast

from langgraph.graph import END, StateGraph

from app.agent import budget
from app.agent.nodes import (
    grade,
    guardrail_in,
    guardrail_out,
    retrieve,
    rewrite,
    route,
    synthesize,
    validate,
)
from app.agent.state import AgentState


def _hard_breach_or_refusal(state: AgentState) -> bool:
    """Check budget/cancel without mutating state — used by edge functions."""
    if state.get("refusal") or state.get("cancelled"):
        return True
    b = budget.breach(state)
    return bool(b and b["hard"])


def _after_guardrail_in(state: AgentState) -> str:
    """Route to the refusal path when the input guardrail blocked the question."""
    return "refuse" if _hard_breach_or_refusal(state) else "route"


def _after_route(state: AgentState) -> str:
    """Refuse when routing set a refusal (e.g. unauthorized client) or budget broke."""
    return "refuse" if _hard_breach_or_refusal(state) else "retrieve"


def _after_grade(state: AgentState) -> str:
    """Grade decision: rewrite (bounded), answer, or refuse — budget-checked."""
    if _hard_breach_or_refusal(state):
        return "refuse"
    return state.get("grade_action", "refuse")


def _after_synthesize(state: AgentState) -> str:
    """Refusal drafts skip validation; budget/cancel also short-circuits out."""
    return "out" if _hard_breach_or_refusal(state) else "validate"


def _after_validate(state: AgentState) -> str:
    """Validator verdict: one bounded re-synthesis, else out."""
    if _hard_breach_or_refusal(state):
        return "out"
    return str(state.get("validation_action", "out"))


def build_graph():
    """Wire the agent nodes and conditional edges into a compiled LangGraph."""
    g = StateGraph(AgentState)
    g.add_node("guardrail_in", guardrail_in.guardrail_in)
    g.add_node("route", route.route)
    g.add_node("retrieve", retrieve.retrieve)
    g.add_node("grade", grade.grade)
    g.add_node("rewrite", rewrite.rewrite)
    g.add_node("synthesize", synthesize.synthesize)
    g.add_node("validate", validate.validate)
    g.add_node("guardrail_out", guardrail_out.guardrail_out)

    g.set_entry_point("guardrail_in")
    g.add_conditional_edges(
        "guardrail_in", _after_guardrail_in, {"route": "route", "refuse": "guardrail_out"}
    )
    g.add_conditional_edges(
        "route", _after_route, {"retrieve": "retrieve", "refuse": "guardrail_out"}
    )
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        _after_grade,
        {"rewrite": "rewrite", "answer": "synthesize", "refuse": "guardrail_out"},
    )
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges(
        "synthesize", _after_synthesize, {"validate": "validate", "out": "guardrail_out"}
    )
    g.add_conditional_edges(
        "validate", _after_validate, {"resynthesize": "synthesize", "out": "guardrail_out"}
    )
    g.add_edge("guardrail_out", END)
    return g.compile()


def _seed_state(
    question: str,
    principal,
    request_id: str,
    trace_id: str,
    max_elapsed_ms: int | None = None,
) -> AgentState:
    """Build the initial agent state including budget counters.

    ``max_elapsed_ms`` is an optional per-request timeout override — it can
    only tighten the configured elapsed-time hard limit, never widen it.
    """
    state: AgentState = {
        "question": question,
        "principal": principal,
        "request_id": request_id,
        "trace_id": trace_id or request_id,
    }
    state.update(cast("AgentState", budget.begin()))
    if max_elapsed_ms:
        state["max_elapsed_ms_override"] = max_elapsed_ms
    return state


def _graph_config(request_id: str, user_id: str):
    """Return (handler, config) with Langfuse tracing wired when enabled."""
    from app.observability.tracing import build_callback_handler, graph_config

    handler, _ = build_callback_handler()
    return handler, graph_config(handler, request_id, user_id)


def _result_dict(result: AgentState, request_id: str, trace_id: str) -> dict:
    """Normalize terminal state into the API/CLI response dict."""
    breach = result.get("budget_breach")
    return {
        "reply": result.get("reply", ""),
        "citations": result.get("citations", []),
        "retrieved": result.get("retrieved", []),
        "tool_calls": [
            {
                "tool": tr["tool"],
                "client_id": tr.get("client_id"),
                "row_count": len(tr["rows"]),
                "has_more": tr.get("has_more", False),
                "fresh_at": tr.get("fresh_at"),
                "source_updated_at": tr.get("source_updated_at"),
                "snapshot_ts": tr.get("snapshot_ts"),
                "error": tr.get("error"),
            }
            for tr in result.get("tool_results", [])
        ],
        "refusal": result.get("refusal", False),
        "abstained": result.get("abstained", False),
        "abstention_reason": result.get("abstention_reason", ""),
        "answerability": getattr(
            result.get("answerability"), "value", result.get("answerability", "")
        ),
        "unsupported_facts": result.get("unsupported_facts", []),
        "pending_review": result.get("pending_review", False),
        "validation": result.get("validation", {}),
        "budget": {
            "iterations": result.get("iterations", 0),
            "tool_calls": result.get("tool_call_count", 0),
            "llm_calls": result.get("llm_calls", 0),
            "tokens_used": result.get("tokens_used", 0),
            "cost_usd": round(result.get("cost_usd", 0.0), 6),
            "breach": breach,
        },
        "errors": result.get("errors", []),
        "diagnostics": result.get("diagnostics", {}),
        "trace_id": trace_id,
        "request_id": request_id,
    }


def answer(
    question: str,
    principal,
    request_id: str,
    trace_id: str = "",
    max_elapsed_ms: int | None = None,
) -> dict:
    """Synchronous entry point used by API, CLI, and eval runner.

    Seeds the agent state (with budget counters), audits the query event,
    invokes the compiled graph with tracing configured, and normalizes the
    result — reply, citations, retrieved docs, tool calls, refusal/abstain
    metadata, answerability, budget usage, errors, and diagnostics.
    """
    from app.agent.nodes.common import audit
    from app.core.constants import EV_QUERY
    from app.core.dependencies import get_graph

    state = _seed_state(question, principal, request_id, trace_id, max_elapsed_ms)
    audit(state, EV_QUERY, {"question": question})
    _, config = _graph_config(request_id, principal.user_id)
    result = get_graph().invoke(state, config=config)
    return _result_dict(result, request_id, state["trace_id"])


def stream_answer(
    question: str,
    principal,
    request_id: str,
    trace_id: str = "",
    max_elapsed_ms: int | None = None,
):
    """Yield ``{type, node, payload}`` events for the SSE endpoint.

    Emits one event per node completion (``node_update``) and a ``final``
    event carrying the normalized result — node-level streaming via
    ``graph.stream(stream_mode="updates")``; the terminal state is
    reconstructed by applying each node's delta to the seeded state in
    order. Token-level streaming is not implemented (sync LLM path).
    """
    from app.agent.nodes.common import audit
    from app.core.constants import EV_QUERY
    from app.core.dependencies import get_graph

    state = _seed_state(question, principal, request_id, trace_id, max_elapsed_ms)
    audit(state, EV_QUERY, {"question": question})
    _, config = _graph_config(request_id, principal.user_id)
    for update in get_graph().stream(state, config=config, stream_mode="updates"):
        for node, delta in update.items():
            yield {"type": "node_update", "node": node, "payload": _safe_delta(delta)}
            if isinstance(delta, dict):
                state.update(cast("AgentState", delta))
    yield {
        "type": "final",
        "node": None,
        "payload": _result_dict(state, request_id, state["trace_id"]),
    }


def _safe_delta(delta) -> dict:
    """Return a JSON-safe summary of a node update (no principals/objects)."""
    import json

    out = {}
    for k, v in (delta or {}).items():
        if k == "principal":
            continue
        try:
            json.dumps(v, default=str)
            out[k] = v
        except (TypeError, ValueError):
            out[k] = str(v)[:500]
    return out
