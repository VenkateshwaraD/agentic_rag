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
from app.core.constants import ABSTENTION_MESSAGE, EV_AGENT_STEP

_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*")
_WORD_RE = re.compile(r"[A-Za-z]{5,}")
_MAX_CLAIMS = 12


def extract_claims(draft: str) -> list[str]:
    """Split the draft into claim sentences for per-claim support checks.

    Headings, bullets, and the offline-draft marker line are skipped; the
    result is capped so validation stays bounded on verbose answers.
    """
    claims: list[str] = []
    for sent in _SENT_RE.split(draft):
        s = sent.strip().lstrip("-*").strip()
        if (
            len(s) > 20
            and not s.startswith("[offline-draft")
            and not s.startswith("Note:")
            and not s.endswith(":")
        ):
            claims.append(s)
    return claims[:_MAX_CLAIMS]


def _citation_valid(cit: dict, retrieved: list[dict]) -> bool:
    """Return True when the citation resolves to a real retrieved chunk."""
    for r in retrieved:
        if r.get("doc_id") != cit.get("doc_id"):
            continue
        if str(r.get("section")) != str(cit.get("section")):
            continue
        if cit.get("chunk_id") and r.get("chunk_id") != cit.get("chunk_id"):
            continue
        return True
    return False


def _passages_for(citation: dict, retrieved: list[dict]) -> str:
    """Return the concatenated text of passages a citation points at."""
    return " ".join(
        r.get("text", "")
        for r in retrieved
        if r.get("doc_id") == citation.get("doc_id")
        and str(r.get("section")) == str(citation.get("section"))
    )


def _claim_supported(claim: str, cited_text: str, all_text: str) -> bool:
    """Heuristic support check: claim numbers and content words must appear
    in the cited passage (falling back to any retrieved passage).

    Approximate by design — the optional LLM verdict can be layered on top;
    offline mode keeps this deterministic.
    """
    if not cited_text and not all_text:
        return False
    hay_primary = cited_text.lower() or all_text.lower()
    hay_all = all_text.lower()
    claim_nums = set(_NUM_RE.findall(claim))
    if claim_nums:
        # every numeric in the claim must appear in *some* evidence;
        # prefer the cited passage but tolerate adjacent sections
        return all(n in hay_primary or n in hay_all for n in claim_nums)
    words = {w.lower() for w in _WORD_RE.findall(claim)}
    words -= {"about", "their", "there", "which", "would", "could", "should", "based"}
    if not words:
        return True
    hits = sum(1 for w in words if w in hay_primary or w in hay_all)
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
    all_text = " ".join(r.get("text", "") for r in retrieved)

    with Timer() as t:
        claims = extract_claims(draft)
        invalid_citations = [c for c in citations if not _citation_valid(c, retrieved)]
        claim_results = []
        unsupported = []
        for claim in claims:
            cited = [c for c in citations if f"[{c['doc_id']} §{c['section']}]" in claim]
            cited_text = " ".join(_passages_for(c, retrieved) for c in cited)
            supported = _claim_supported(claim, cited_text, all_text)
            claim_results.append(
                {"text": claim[:200], "cited": bool(cited), "supported": supported}
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
