"""Synthesize node — Claude drafts the cited reply (or emits a refusal)."""

import json
import re

from app.agent.llm import get_llm
from app.agent.nodes.common import Timer, audit, load_prompt
from app.core.constants import CITATION_PATTERN, EV_AGENT_STEP

_REFUSAL = (
    "I don't have enough entitled information to answer that question. "
    "Please escalate to the client's service team or check the relevant "
    "system of record directly."
)


def _format_evidence(state) -> str:
    from app.core.dependencies import get_pii_redactor

    redactor = get_pii_redactor()
    parts = []
    for r in state.get("retrieved", []):
        parts.append(f"[{r['doc_id']} §{r['section']}] ({r['source']})\n{r['text']}")
    for tr in state.get("tool_results", []):
        if tr["rows"]:
            # raw system rows are PII-redacted before entering the context
            safe = redactor.redact(json.dumps(tr["rows"][:10], default=str))
            parts.append(f"[{tr['tool']} system]\n{safe}")
    return "\n\n---\n\n".join(parts) or "NO EVIDENCE RETRIEVED"


def _extract_citations(draft: str) -> list[dict]:
    return [
        {"doc_id": m.group(1), "section": m.group(2).strip()}
        for m in re.finditer(CITATION_PATTERN, draft)
    ]


def _offline_draft(state) -> str:
    """Deterministic extractive draft used when no LLM key is configured —
    keeps the pipeline runnable offline. Clearly marked; never used in prod."""
    parts = ["[offline-draft — no LLM configured] Based on the retrieved evidence:"]
    for r in state.get("retrieved", [])[:3]:
        snippet = r["text"].strip().split("\n")[0][:300]
        parts.append(f"- {snippet} [{r['doc_id']} §{r['section']}]")
    for tr in state.get("tool_results", []):
        for row in tr["rows"][:3]:
            parts.append(f"- {row} [{tr['tool']} system]")
    return "\n".join(parts)


def synthesize(state):
    if state.get("refusal"):
        return {"reply": _REFUSAL, "citations": []}

    evidence = _format_evidence(state)
    prompt = (
        load_prompt("system.md")
        + "\n\n=== EVIDENCE ===\n"
        + evidence
        + f"\n\n=== QUESTION ===\n{state['question']}\n\nDraft reply:"
    )
    with Timer() as t:
        from app.core.config import get_settings

        draft = (
            get_llm().invoke(prompt).content
            if get_settings().anthropic_api_key
            else _offline_draft(state)
        )
        citations = _extract_citations(draft)
        has_sys_refs = any(
            f"[{tr['tool']} system]" in draft for tr in state.get("tool_results", [])
        )
        # citation-required policy: evidence used → at least one citation/ref
        refusal = bool(state.get("retrieved") or has_rows(state)) and not (
            citations or has_sys_refs
        )
    audit(
        state,
        EV_AGENT_STEP,
        {"node": "synthesize", "citations": citations, "has_sys_refs": has_sys_refs},
        t.ms,
    )
    if refusal:
        return {
            "refusal": True,
            "refusal_reason": "draft lacked required citations",
            "reply": _REFUSAL,
            "citations": [],
        }
    return {"draft": draft, "reply": draft, "citations": citations}


def has_rows(state) -> bool:
    return any(tr["rows"] for tr in state.get("tool_results", []))
