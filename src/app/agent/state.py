"""Agent state passed through the LangGraph.

Author: venkateshwara doijode
"""

from typing import Any, TypedDict

from app.security.authn import Principal


class AgentState(TypedDict, total=False):
    """Typed dict of the state threaded through every node of the agent graph.

    Groups: inputs (question, principal, ids), routing decisions (sources,
    ops calls, rewrite/grade flags), retrieval results (chunks, diagnostics,
    tool rows), and outputs (draft/reply, citations, refusal metadata).
    """

    # inputs
    question: str
    principal: Principal
    request_id: str
    trace_id: str

    # routing
    sources: list[str]  # doc collections to search
    ops_calls: list[dict]  # [{tool, args}]
    rewritten: bool
    grade_action: str  # answer | rewrite | refuse

    # answerability / retry classification (grade → rewrite → validate)
    answerability: str  # Answerability value
    failure_type: str  # FailureType value — drives rewrite strategy
    retries: int  # rewrite attempts used
    resynthesized: bool  # one bounded re-synthesis after validation failure
    conflicts: list[dict]  # detected contradiction pairs across docs
    unsupported_facts: list[str]  # named facts lacking evidence (partial answers)

    # execution budget (agent/budget.py)
    iterations: int  # retrieve passes through the graph
    tool_call_count: int  # total ops tool invocations
    llm_calls: int
    tokens_used: int
    cost_usd: float
    started_at: float
    max_elapsed_ms_override: int  # per-request timeout cap (tightens the default)
    cancelled: bool  # cooperative cancel flag (client disconnect)
    budget_breach: dict  # {limit, used, cap, hard}
    errors: list[str]  # sanitized failure descriptions (surface in diagnostics)

    # independent answer validation (validate node)
    validation: dict  # {claims[], unsupported_claims, invalid_citations, verdict}
    validation_feedback: str  # injected into the re-synthesis prompt

    # retrieval results
    retrieved: list[dict]  # top-m chunks {doc_id, section, text, ...}
    diagnostics: dict  # per-stage retrieval diagnostics
    tool_results: list[dict]  # [{tool, rows, error?}]

    # output
    draft: str
    reply: str
    citations: list[dict]  # {doc_id, section, chunk_id?, doc_version?, effective_date?}
    refusal: bool
    refusal_reason: str
    abstained: bool  # explicit abstention (insufficient evidence, not a block)
    abstention_reason: str
    pending_review: bool  # queued for human review post-response
    guardrail_log: list[dict]
    extra: dict[str, Any]
