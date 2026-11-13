"""Agent state passed through the LangGraph."""

from typing import Any, TypedDict

from app.security.authn import Principal


class AgentState(TypedDict, total=False):
    # inputs
    question: str
    principal: Principal
    request_id: str
    trace_id: str

    # routing
    sources: list[str]  # doc collections to search
    ops_calls: list[dict]  # [{tool, args}]
    rewritten: bool
    grade_action: str  # answer | retry | refuse

    # retrieval results
    retrieved: list[dict]  # top-m chunks {doc_id, section, text, ...}
    diagnostics: dict  # per-stage retrieval diagnostics
    tool_results: list[dict]  # [{tool, rows}]

    # output
    draft: str
    reply: str
    citations: list[dict]
    refusal: bool
    refusal_reason: str
    guardrail_log: list[dict]
    extra: dict[str, Any]
