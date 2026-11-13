"""LangGraph assembly.

Flow: guardrail_in → route → retrieve → grade ─┐
        ↑ rewrite once ────────────────────────┘
        → (sufficient) → synthesize → guardrail_out → END
        → (insufficient after rewrite) → guardrail_out → END
"""

from langgraph.graph import END, StateGraph

from app.agent.nodes import grade, guardrail_in, guardrail_out, retrieve, route, synthesize
from app.agent.state import AgentState


def _after_guardrail_in(state: AgentState) -> str:
    return "refuse" if state.get("refusal") else "route"


def _after_grade(state: AgentState) -> str:
    return state.get("grade_action", "refuse")


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("guardrail_in", guardrail_in.guardrail_in)
    g.add_node("route", route.route)
    g.add_node("retrieve", retrieve.retrieve)
    g.add_node("grade", grade.grade)
    g.add_node("synthesize", synthesize.synthesize)
    g.add_node("guardrail_out", guardrail_out.guardrail_out)

    g.set_entry_point("guardrail_in")
    g.add_conditional_edges(
        "guardrail_in",
        _after_guardrail_in,
        {"route": "route", "refuse": "guardrail_out"},
    )
    g.add_conditional_edges(
        "route",
        lambda s: "refuse" if s.get("refusal") else "retrieve",
        {"retrieve": "retrieve", "refuse": "guardrail_out"},
    )
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        _after_grade,
        {
            "retry": "retrieve",
            "answer": "synthesize",
            "refuse": "guardrail_out",
        },
    )
    g.add_edge("synthesize", "guardrail_out")
    g.add_edge("guardrail_out", END)
    return g.compile()


def answer(question: str, principal, request_id: str, trace_id: str = "") -> dict:
    """Synchronous entry point used by API, CLI, and eval runner."""
    from app.agent.nodes.common import audit
    from app.core.constants import EV_QUERY
    from app.observability.tracing import build_callback_handler, graph_config

    state: AgentState = {
        "question": question,
        "principal": principal,
        "request_id": request_id,
        "trace_id": trace_id or request_id,
    }
    audit(state, EV_QUERY, {"question": question})
    handler, _ = build_callback_handler()
    config = graph_config(handler, request_id, principal.user_id)
    result = build_graph().invoke(state, config=config)
    return {
        "reply": result.get("reply", ""),
        "citations": result.get("citations", []),
        "retrieved": result.get("retrieved", []),
        "tool_calls": [
            {"tool": tr["tool"], "client_id": tr.get("client_id"), "row_count": len(tr["rows"])}
            for tr in result.get("tool_results", [])
        ],
        "refusal": result.get("refusal", False),
        "diagnostics": result.get("diagnostics", {}),
        "trace_id": state["trace_id"],
        "request_id": request_id,
    }
