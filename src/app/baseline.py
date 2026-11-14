"""Plain-RAG baseline — retrieve-then-answer with none of the agent loop.

A defensible comparison point for the agentic pipeline: the same corpus,
stores, embedding model, and draft machinery — but no routing, no
collection classification, no ops tools, no grading/rewrites, no
validation, no guardrails, and no answerability verdicts. Any quality
delta between ``baseline.plain_answer`` and ``graph.answer`` is therefore
attributable to the agent machinery rather than to better retrieval or a
stronger model.

Two axes are parameterised so the comparison can isolate *what* matters:

- ``strategy`` — ``"dense"`` (what naive RAG demos ship: embed → top-k)
  vs ``"hybrid_rerank"`` (identical retrieval to the agentic pipeline;
  remaining deltas are provably the agent loop, not retrieval quality)
- ``use_acl`` — ``False`` is realistic naive RAG: no entitlement filter,
  so unauthorized content leaks into context (a real, demonstrable win
  for the agentic system). ``True`` adds the ACL pre-filter, isolating
  the delta attributable to the agent loop alone.

The output dict mirrors ``graph._result_dict`` so every eval metric
applies unchanged — refusals surface as ``refusal_correct=0``,
hallucinated answers as ``answerability_correct=0``, echoed injection as
``no_banned_content=0``, and leaked clients as ``acl_clean=0``.

Author: venkateshwara doijode
"""

from app.agent.nodes.synthesize import _extract_citations, _offline_draft
from app.core.config import get_settings
from app.core.constants import (
    COLLECTION_AGREEMENTS,
    COLLECTION_SERVICE_TEAMS,
    COLLECTION_SLAS,
    Answerability,
)
from app.core.dependencies import get_hybrid
from app.security.authz import acl_for

ALL_COLLECTIONS = [COLLECTION_AGREEMENTS, COLLECTION_SLAS, COLLECTION_SERVICE_TEAMS]


def _search_all(query: str, acl, strategy: str, top_k: int) -> list[dict]:
    """Query every collection like one merged index; keep the top-k by score.

    A naive RAG system has a single index — the baseline reproduces that
    by merging per-collection result lists and keeping the global top-k,
    ordered by whatever score the strategy produced (cosine similarity
    for ``dense``, reranker logits for ``hybrid_rerank``).
    """
    hybrid = get_hybrid()
    merged: list[dict] = []
    for coll in ALL_COLLECTIONS:
        try:
            out = hybrid.search(query, coll, acl, strategy=strategy)
        except Exception:
            continue  # a store being down degrades the baseline silently
        for r in out["results"]:
            r["collection"] = coll
        merged.extend(out["results"])
    merged.sort(key=lambda r: r.get("score") or 0.0, reverse=True)
    return merged[:top_k]


def _baseline_draft(question: str, retrieved: list[dict]) -> str:
    """One-shot draft: identical generation machinery to the agentic
    pipeline (same LLM when configured, same extractive echo offline) so
    the comparison measures architecture, not model quality."""
    if get_settings().anthropic_api_key:
        from app.agent.llm import get_llm
        from app.agent.nodes.synthesize import _format_evidence

        evidence = _format_evidence({"retrieved": retrieved, "tool_results": []})
        prompt = (
            "Answer the question using only the evidence below. "
            "Cite passages as [DOC §section].\n\n=== EVIDENCE ===\n"
            f"{evidence}\n\n=== QUESTION ===\n{question}\n\nReply:"
        )
        try:
            return get_llm().invoke(prompt).content
        except Exception:
            pass  # fall through to the deterministic draft
    return _offline_draft({"retrieved": retrieved, "tool_results": []})


def plain_answer(
    question: str,
    principal,
    request_id: str,
    *,
    strategy: str = "dense",
    use_acl: bool = False,
    top_k: int = 5,
) -> dict:
    """Answer ``question`` with naive retrieve-then-generate.

    No routing (all collections searched), no tools, no guardrails, no
    grading, no validation — so the response always claims
    ``fully_answerable`` and never refuses or abstains. Output shape
    matches ``graph.answer`` so eval metrics compute identically.
    """
    acl = acl_for(principal) if use_acl else None
    retrieved = _search_all(question, acl, strategy, top_k)
    reply = _baseline_draft(question, retrieved)
    citations = _extract_citations(reply, retrieved)
    return {
        "reply": reply,
        "citations": citations,
        "retrieved": retrieved,
        "tool_calls": [],
        "refusal": False,
        "abstained": False,
        "abstention_reason": "",
        "answerability": Answerability.FULL.value,
        "unsupported_facts": [],
        "pending_review": False,
        "validation": {},
        "budget": {
            "iterations": 1,
            "tool_calls": 0,
            "llm_calls": 0,
            "tokens_used": 0,
            "cost_usd": 0.0,
            "breach": None,
        },
        "errors": [],
        "diagnostics": {"baseline": {"strategy": strategy, "use_acl": use_acl}},
        "trace_id": request_id,
        "request_id": request_id,
    }
