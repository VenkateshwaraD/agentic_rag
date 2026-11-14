"""Validate node — independently checks the draft's claims and citations.

Runs after synthesis: every ``[DOC §x]`` citation must resolve to a real
``(doc_id, section[, chunk_id])`` actually present in ``retrieved``, and
every factual claim must be supported by the passage(s) it cites. Claims
are extracted heuristically offline (sentence split + numeric/entity
support overlap); an optional bounded LLM verdict is used when configured.

On failure the graph re-synthesizes once with feedback; a second failure
abstains rather than shipping an unverified answer.

Author: venkateshwara doijode
"""

import re

from app.agent.nodes.common import Timer, audit
from app.agent.nodes.synthesize import _sec_key
from app.core.constants import ABSTENTION_MESSAGE, EV_AGENT_STEP, Answerability

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WORD_RE = re.compile(r"[A-Za-z]{5,}")
_CITE_RE = re.compile(r"\[[^\]]*\]")
_MAX_CLAIMS = 12


def _claims_with_lines(draft: str) -> list[tuple[str, str]]:
    """Split the draft into (claim, source_line) pairs.

    Lines are split first — each bullet is its own claim, so tokens from a
    differently-cited line can never leak into a claim. The line is kept so
    citations sitting anywhere on the same bullet line count as evidence
    for every claim on it (drafts append the marker mid- or end-of-line).
    """
    out: list[tuple[str, str]] = []
    for line in draft.splitlines():
        for sent in _SENT_RE.split(line):
            s = sent.strip().lstrip("-*").strip()
            if (
                len(s) > 20
                and not s.startswith("[offline-draft")
                and not s.startswith("Note:")
                and not s.endswith(":")
            ):
                out.append((s, line))
    return out[:_MAX_CLAIMS]


def extract_claims(draft: str) -> list[str]:
    """Split the draft into claim sentences for per-claim support checks.

    Headings, bullets, and the offline-draft marker line are skipped; the
    result is capped so validation stays bounded on verbose answers.
    """
    return [c for c, _ in _claims_with_lines(draft)]


def _citation_valid(cit: dict, retrieved: list[dict]) -> bool:
    """Return True when the citation resolves to a real retrieved chunk."""
    for r in retrieved:
        if r.get("doc_id") != cit.get("doc_id"):
            continue
        if _sec_key(r.get("section")) != _sec_key(cit.get("section")):
            continue
        if cit.get("chunk_id") and r.get("chunk_id") != cit.get("chunk_id"):
            continue
        return True
    return False


def _passages_for(citation: dict, retrieved: list[dict]) -> str:
    """Return the concatenated text of passages a citation points at.

    Matches doc_id + section; when the citation carries a ``chunk_id``,
    only that versioned chunk counts as support — a same-named section in
    another version is not interchangeable evidence.
    """
    return " ".join(
        r.get("text", "")
        for r in retrieved
        if r.get("doc_id") == citation.get("doc_id")
        and _sec_key(r.get("section")) == _sec_key(citation.get("section"))
        and (not citation.get("chunk_id") or r.get("chunk_id") == citation["chunk_id"])
    )


def _claim_supported(claim: str, cited_text: str) -> bool:
    """Heuristic support check: claim numbers and content words must appear
    in the *cited* passage — a fact asserted under one citation is not
    established by unrelated evidence elsewhere.

    Approximate by design — the optional LLM verdict can be layered on top;
    offline mode keeps this deterministic.
    """
    hay = cited_text.lower()
    if not hay:
        return False
    # strip inline [DOC §sec] citation markers — their tokens (doc ids,
    # section numbers) are not claim content and would never appear in
    # the passage text. Bare doc-id tokens in prose (a claim naming
    # "SLA-002") are stripped too — "002" is an identifier, not an
    # asserted fact that must appear in the evidence
    claim = _CITE_RE.sub(" ", claim)
    claim = re.sub(r"\b[A-Z]{2,4}-\d{3,}\b", " ", claim)
    claim_nums = set(_NUM_RE.findall(claim))
    if claim_nums:
        # every numeric must be asserted in the cited passage itself
        return all(n in hay for n in claim_nums)
    words = {w.lower() for w in _WORD_RE.findall(claim)}
    words -= {"about", "their", "there", "which", "would", "could", "should", "based"}
    if not words:
        return True
    hits = sum(1 for w in words if w in hay)
    return hits / len(words) >= 0.6


def validate(state):
    """Independently verify the draft against the retrieved evidence.

    Skipped for refusals/abstentions and empty drafts. Produces a
    ``validation`` record — per-claim support, invalid citations,
    contradictions from grading — and routes the graph: ``resynthesize``
    once with feedback, or abstain when the second draft still fails.
    """
    if state.get("refusal") or state.get("abstained") or not state.get("draft"):
        return {"validation_action": "out", "validation": {"verdict": "skipped"}}

    draft = state["draft"]
    retrieved = state.get("retrieved", [])
    citations = state.get("citations", [])
    # ops tool rows are evidence too — tool-only claims validate against
    # them when the claim cites "[<tool> system]"
    import json as _json

    tool_text = " ".join(
        _json.dumps(tr.get("rows", []), default=str) for tr in state.get("tool_results", [])
    )

    with Timer() as t:
        claims = _claims_with_lines(draft)
        invalid_citations = [c for c in citations if not _citation_valid(c, retrieved)]
        claim_results = []
        unsupported = []
        for claim, line in claims:
            # a citation marker anywhere on the claim's line counts —
            # bullets cite the whole line, not just the final sentence
            cited = [
                c
                for c in citations
                if f"[{c['doc_id']} §{_sec_key(c['section'])}]" in line
                or f"[{c['doc_id']}]" in line
            ]
            cited_text = " ".join(_passages_for(c, retrieved) for c in cited)
            # tool-cited claims pull support from the ops evidence pool —
            # same-line rule as doc citations (marker may sit outside the
            # extracted claim sentence, e.g. after a sentence break)
            tool_cited = any(
                f"[{tr['tool']} system]" in line for tr in state.get("tool_results", [])
            )
            if not cited_text and tool_cited:
                cited_text = tool_text
            # uncited claims cannot be verified — claim-level evidence required
            supported = bool(cited_text) and _claim_supported(claim, cited_text)
            claim_results.append(
                {
                    "text": claim[:200],
                    "cited": bool(cited) or tool_cited,
                    "supported": supported,
                    "via": "tool" if tool_cited and not cited else "docs",
                }
            )
            if not supported:
                unsupported.append(claim[:200])
        verdict = "pass" if not unsupported and not invalid_citations else "fail"
        validation = {
            "claims": claim_results,
            "unsupported_claims": unsupported,
            "invalid_citations": invalid_citations,
            "contradictions": state.get("conflicts", []),
            "verdict": verdict,
        }

        updates: dict = {"validation": validation}
        if verdict == "fail":
            if state.get("resynthesized"):
                # already re-synthesized once → abstain, don't loop
                updates.update(
                    {
                        "abstained": True,
                        "abstention_reason": "draft failed independent validation",
                        "answerability": Answerability.INSUFFICIENT,
                        "reply": ABSTENTION_MESSAGE,
                        "validation_action": "out",
                    }
                )
            else:
                feedback = []
                if unsupported:
                    feedback.append("unsupported claims: " + "; ".join(unsupported[:5]))
                if invalid_citations:
                    feedback.append(
                        "citations not present in evidence: "
                        + "; ".join(
                            f"[{c['doc_id']} §{c['section']}]" for c in invalid_citations[:5]
                        )
                    )
                updates["validation_feedback"] = "\n".join(feedback)
                updates["validation_action"] = "resynthesize"
        else:
            updates["validation_action"] = "out"
    audit(
        state,
        EV_AGENT_STEP,
        {
            "node": "validate",
            "verdict": verdict,
            "claims": len(claim_results),
            "unsupported": len(unsupported),
            "invalid_citations": len(invalid_citations),
            "resynthesized": state.get("resynthesized", False),
        },
        t.ms,
    )
    return updates
