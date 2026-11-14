"""Synthesize node — Claude drafts the cited reply (or emits a refusal).

Handles partial/conflicting answerability caveats, validator feedback on
the bounded re-synthesis pass, LLM outages (falls back to the offline
extractive draft and records ``llm_unavailable``), and versioned citation
objects enriched from retrieved chunk metadata.

Author: venkateshwara doijode
"""

import datetime as dt
import json
import re

from app.agent import budget
from app.agent.llm import get_llm
from app.agent.nodes.common import Timer, add_error, audit, load_prompt
from app.core.constants import CITATION_PATTERN, EV_AGENT_STEP, REFUSAL_MESSAGE, Answerability

_REFUSAL = REFUSAL_MESSAGE


def _sec_key(section) -> str:
    """Normalize a section label for citation markers.

    Indexed sections already carry a leading '§' ("§2 Fee Schedule"), so
    composing ``§{section}`` produced '§§'. Everything that emits or
    matches a marker goes through this so a single '§' appears exactly once.
    """
    return str(section or "").lstrip("§").strip()


def _pii():
    """Lazy accessor for the shared PII redactor."""
    from app.core.dependencies import get_pii_redactor

    return get_pii_redactor()


def _format_evidence(state) -> str:
    """Render retrieved chunks and PII-redacted tool rows as a prompt evidence block."""
    redactor = _pii()
    parts = []
    for r in state.get("retrieved", []):
        meta = f"v{r['doc_version']}" if r.get("doc_version") else ""
        eff = f" eff {r['effective_date']}" if r.get("effective_date") else ""
        parts.append(
            f"[{r['doc_id']} §{_sec_key(r['section'])}] ({r['source']}) {meta}{eff}\n{r['text']}"
        )
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
    seen: set[tuple[str, str]] = set()
    for m in re.finditer(CITATION_PATTERN, draft):
        doc_id, section = m.group(1), _sec_key(m.group(2))
        if (doc_id, section) in seen:
            continue  # one Source entry per (doc, section) — not per chunk
        seen.add((doc_id, section))
        cit = {"doc_id": doc_id, "section": section}
        for r in retrieved:
            if r.get("doc_id") == doc_id and _sec_key(r.get("section")) == section:
                for k in ("chunk_id", "doc_version", "effective_date"):
                    if r.get(k):
                        cit[k] = r[k]
                break
        citations.append(cit)
    return citations


_AGREE_WORDS = re.compile(r"\b(agree|consistent|the same|match|aligned)\b")


def _conflict_lead(state, conflict: dict, docs: dict) -> str:
    """Lead sentence that *answers* a conflict question instead of burying it —
    'No — SLA-002 reports 17:00, SLA-010 reports 15:00' for 'do X and Y agree?'."""
    a, b = docs.get(conflict["doc_a"], {}), docs.get(conflict["doc_b"], {})
    field = re.sub(r"[|]+", " ", conflict.get("field", "")).strip() or "the same field"
    va, vb = ", ".join(conflict["values_a"][:3]), ", ".join(conflict["values_b"][:3])
    cite_a = f"[{conflict['doc_a']} §{_sec_key(a.get('section', conflict['section']))}]"
    cite_b = f"[{conflict['doc_b']} §{_sec_key(b.get('section', conflict['section']))}]"
    no = "No — " if _AGREE_WORDS.search(state.get("question", "").lower()) else ""
    return (
        f"{no}sources disagree on {field}: {conflict['doc_a']} reports {va} {cite_a},"
        f" {conflict['doc_b']} reports {vb} {cite_b}."
    )


def _table_rows(text: str) -> list[str]:
    """Render a markdown table in the chunk as 'label: value' lines.

    Returns [] for non-table text. The offline draft emits each row as its
    own cited bullet, so the per-line citation rule still holds without
    flattening the table into one unreadable line.
    """
    rows = [ln.strip() for ln in text.splitlines() if ln.strip().startswith("|")]
    if len(rows) < 2:
        return []
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    out = []
    for line in rows[1:]:
        cells = [c.strip() for c in line.strip("|").split("|")]
        if not any(c.strip("-: ") for c in cells):
            continue  # separator row |---|---|
        if len(cells) == 2:
            out.append(f"{cells[0]}: {cells[1]}")
        else:
            out.append(
                f"{cells[0]} — "
                + "; ".join(
                    f"{header[i]} {c}" if i < len(header) and header[i] else c
                    for i, c in enumerate(cells[1:], 1)
                )
            )
    return out


_TOOL_ROW_FIELDS = {
    "tickets": ("ticket_id", "status", "category", "subject", "created_at"),
    "trades": ("trade_id", "status", "amount", "currency", "value_date", "settled_at"),
    "accounts": ("name", "tier", "region", "owner_team", "agreement_doc"),
}
_TOOL_ROW_CAP = 5


def _fmt_val(v) -> str:
    """One display value — datetimes become dates, everything else str()."""
    if isinstance(v, dt.datetime):
        return v.date().isoformat()
    if isinstance(v, dt.date):
        return v.isoformat()
    return str(v)


def _format_tool_row(tool: str, row) -> str:
    """Render one ops row as a readable summary — never a Python dict repr.

    Field whitelists keep unasked-for fields (and the internal client_id)
    out of the draft; dates render as ISO, not ``datetime.date(2025,...)``.
    """
    if not isinstance(row, dict):
        return str(row)[:200]
    fields = _TOOL_ROW_FIELDS.get(tool) or tuple(k for k in row if k != "client_id")[:5]
    parts = [f"{k}={_fmt_val(row[k])}" for k in fields if row.get(k) not in (None, "")]
    return ", ".join(parts) or "row"


def _offline_draft(state) -> str:
    """Deterministic extractive draft used when no LLM key is configured —
    keeps the pipeline runnable offline. Clearly marked; never used in prod."""
    parts = ["[offline-draft — no LLM configured] Based on the retrieved evidence:"]
    # surface detected contradictions first — a conflict-aware answer leads
    # with the disagreement (and a dual citation), not the boilerplate
    docs = {r.get("doc_id"): r for r in state.get("retrieved", [])}
    num_conflicts = [c for c in state.get("conflicts", []) if c.get("kind") == "numeric_divergence"]
    if num_conflicts:
        parts.append(f"- {_conflict_lead(state, num_conflicts[0], docs)}")
        for c in num_conflicts[1:2]:
            a, b = docs.get(c["doc_a"], {}), docs.get(c["doc_b"], {})
            field = re.sub(r"[|]+", " ", c.get("field", "")).strip() or "the same field"
            parts.append(
                f"- also disagree on {field}: {', '.join(c['values_a'][:3])} "
                f"versus {', '.join(c['values_b'][:3])} "
                f"[{c['doc_a']} §{_sec_key(a.get('section', c['section']))}] "
                f"[{c['doc_b']} §{_sec_key(b.get('section', c['section']))}]"
            )
    for r in state.get("retrieved", [])[:3]:
        cite = f"[{r['doc_id']} §{_sec_key(r['section'])}]"
        if rows := _table_rows(r["text"]):
            for row in rows[:3]:
                parts.append(f"- {row} {cite}")
        else:
            # flatten to one line so the trailing citation marker covers every
            # sentence in the snippet — the validator scopes citations per line
            snippet = " ".join(r["text"].split())[:300]
            parts.append(f"- {snippet} {cite}")
    for tr in state.get("tool_results", []):
        rows = tr.get("rows", [])
        for row in rows[:_TOOL_ROW_CAP]:
            # ops rows are raw DB output — PII-redact before they enter the
            # draft, same as the LLM evidence path does (else check_output
            # would have to refuse rather than show masked values)
            safe_row = _pii().redact(_format_tool_row(tr["tool"], row))
            parts.append(f"- {safe_row} [{tr['tool']} system]")
        if len(rows) > _TOOL_ROW_CAP:
            # 'Note:' lines are skipped by the claim extractor — not a claim
            parts.append(f"Note: {len(rows) - _TOOL_ROW_CAP} further rows omitted for brevity.")
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
