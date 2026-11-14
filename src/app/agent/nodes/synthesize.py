"""Synthesize node — Claude drafts the cited reply (or emits a refusal).

Handles partial/conflicting answerability caveats, validator feedback on
the bounded re-synthesis pass, LLM outages (falls back to the offline
extractive draft and records ``llm_unavailable``), and versioned citation
objects enriched from retrieved chunk metadata.

Author: venkateshwara doijode
"""

import json
import re

from app.agent import budget
from app.agent.llm import get_llm
from app.agent.nodes.common import Timer, add_error, audit, load_prompt
from app.core.constants import CITATION_PATTERN, EV_AGENT_STEP, REFUSAL_MESSAGE, Answerability

_REFUSAL = REFUSAL_MESSAGE


def _format_evidence(state) -> str:
    """Render retrieved chunks and PII-redacted tool rows as a prompt evidence block."""
    from app.core.dependencies import get_pii_redactor

    redactor = get_pii_redactor()
    parts = []
    for r in state.get("retrieved", []):
        meta = f"v{r['doc_version']}" if r.get("doc_version") else ""
        eff = f" eff {r['effective_date']}" if r.get("effective_date") else ""
        parts.append(f"[{r['doc_id']} §{r['section']}] ({r['source']}) {meta}{eff}\n{r['text']}")
    for tr in state.get("tool_results", []):
        if tr.get("rows"):
            # raw system rows are PII-redacted before entering the context
            safe = redactor.redact(json.dumps(tr["rows"][:10], default=str))
            parts.append(f"[{tr['tool']} system]\n{safe}")
    return "\n\n---\n\n".join(parts) or "NO EVIDENCE RETRIEVED"


def _extract_citations(draft: str, retrieved: list[dict]) -> list[dict]:
    """Parse ``[DOC-§section]`` markers and enrich with retrieved metadata.

    Each citation is matched back to a retrieved chunk by (doc_id, section)
    so it carries ``chunk_id``, ``doc_version`` and ``effective_date`` when
    those fields are present in the index metadata.
    """
    citations = []
    for m in re.finditer(CITATION_PATTERN, draft):
        doc_id, section = m.group(1), m.group(2).strip()
        cit = {"doc_id": doc_id, "section": section}
        for r in retrieved:
            if r.get("doc_id") == doc_id and str(r.get("section")) == section:
                for k in ("chunk_id", "doc_version", "effective_date"):
                    if r.get(k):
                        cit[k] = r[k]
                break
        citations.append(cit)
    return citations


def _offline_draft(state) -> str:
    """Deterministic extractive draft used when no LLM key is configured —
    keeps the pipeline runnable offline. Clearly marked; never used in prod."""
    parts = ["[offline-draft — no LLM configured] Based on the retrieved evidence:"]
    # surface detected contradictions with the divergent values + a dual
    # citation — a conflict-aware answer must name what disagrees
    for c in state.get("conflicts", [])[:2]:
        if c.get("kind") != "numeric_divergence":
            continue
        docs = {r.get("doc_id"): r for r in state.get("retrieved", [])}
        a, b = docs.get(c["doc_a"], {}), docs.get(c["doc_b"], {})
        field = re.sub(r"[|]+", " ", c.get("field", "")).strip() or "the same field"
        parts.append(
            f"- Sources disagree on {field}: {', '.join(c['values_a'][:3])} "
            f"versus {', '.join(c['values_b'][:3])} "
            f"[{c['doc_a']} §{a.get('section', c['section'])}] "
            f"[{c['doc_b']} §{b.get('section', c['section'])}]"
        )
    for r in state.get("retrieved", [])[:3]:
        # flatten to one line so the trailing citation marker covers every
        # sentence in the snippet — the validator scopes citations per line
        snippet = " ".join(r["text"].split())[:300]
        parts.append(f"- {snippet} [{r['doc_id']} §{r['section']}]")
    for tr in state.get("tool_results", []):
        for row in tr.get("rows", [])[:3]:
            parts.append(f"- {row} [{tr['tool']} system]")
    return "\n".join(parts)


def _caveat_prefix(state) -> str:
    """Build the caveat sentence prepended to PARTIAL / CONFLICTING answers."""
    answerability = state.get("answerability")
    if answerability == Answerability.CONFLICTING or state.get("conflicts"):
        return (
            "Note: the retrieved sources conflict on parts of this answer; "
            "I've flagged the conflicting points where they appear. "
        )
    if answerability == Answerability.PARTIAL:
        missing = "; ".join(state.get("unsupported_facts", [])) or "some requested facts"
        return f"Note: I could only partially answer this — {missing}.\n\n"
    return ""


def has_rows(state) -> bool:
    """Return True when any ops tool call produced at least one row."""
    return any(tr.get("rows") for tr in state.get("tool_results", []))


def synthesize(state):
    """Draft the final cited answer from the gathered evidence.

    Short-circuits to the refusal message when upstream already refused.
    Otherwise builds the evidence prompt (with partial/conflict caveats and
    validator feedback when re-synthesizing), calls the LLM — falling back
    to the offline draft on timeout/error — extracts versioned citations,
    enforces the citation-required policy, audits the step, and returns the
    reply plus citations — or a refusal when the draft lacks citations.
    """
    if state.get("refusal") or state.get("abstained"):
        return {"reply": _REFUSAL if state.get("refusal") else state.get("reply"), "citations": []}

    evidence = _format_evidence(state)
    updates: dict = {}
    prompt = load_prompt("system.md") + "\n\n=== EVIDENCE ===\n" + evidence
    if state.get("answerability") in (Answerability.PARTIAL, Answerability.CONFLICTING):
        missing = "; ".join(state.get("unsupported_facts", []))
        prompt += (
            "\n\nThe evidence is incomplete or conflicting. Answer only what the "
            "evidence supports, and explicitly say which parts could not be "
            f"answered or are contradictory. Unanswered aspects: {missing or 'unspecified'}."
        )
    if state.get("validation_feedback"):
        # bounded re-synthesis pass — fix what the validator flagged
        prompt += (
            "\n\nA previous draft failed validation. Fix these problems — remove "
            "unsupported claims and only cite passages that exist in the evidence: "
            f"{state['validation_feedback']}"
        )
        updates["resynthesized"] = True
    prompt += f"\n\n=== QUESTION ===\n{state['question']}\n\nDraft reply:"

    from app.core.config import get_settings

    with Timer() as t:
        draft = ""
        if get_settings().anthropic_api_key:
            try:
                response = get_llm().invoke(prompt)
                updates.update(budget.record_llm_usage(state, response))
                draft = response.content
            except Exception as exc:
                # LLM timeout / unavailable → degraded offline draft
                updates["errors"] = add_error(state, f"synthesize.llm_failed: {exc}")
        if not draft:
            draft = _offline_draft(state)
            if get_settings().anthropic_api_key:
                updates["errors"] = add_error(
                    {**state, "errors": updates.get("errors", [])},
                    "synthesize.llm_unavailable: using offline draft",
                )

        draft = _caveat_prefix(state) + draft
        citations = _extract_citations(draft, state.get("retrieved", []))
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
        {
            "node": "synthesize",
            "citations": citations,
            "has_sys_refs": has_sys_refs,
            "resynthesized": updates.get("resynthesized", False),
        },
        t.ms,
    )
    if refusal:
        return {
            "refusal": True,
            "refusal_reason": "draft lacked required citations",
            "reply": _REFUSAL,
            "citations": [],
            **updates,
        }
    return {"draft": draft, "reply": draft, "citations": citations, **updates}
