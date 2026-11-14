"""Unit tests for the plain-RAG baseline and the report comparison tool.

Author: venkateshwara doijode
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

import compare as C  # noqa: E402

from app import baseline  # noqa: E402
from app.security.authn import Principal  # noqa: E402


class _FakeHybrid:
    """Canned retriever — returns one scored chunk per collection."""

    def __init__(self):
        self.calls = []

    def search(self, query, collection, acl, strategy=None):
        self.calls.append((collection, acl, strategy))
        return {
            "results": [
                {
                    "doc_id": "DOC-001",
                    "section": "s",
                    "text": f"{collection} says wire fee is $12.73 per item",
                    "score": 0.9,
                }
            ],
            "diagnostics": {},
        }


def _principal():
    return Principal(user_id="u1", role="agent", client_ids=("acme",))


def test_plain_answer_output_shape(monkeypatch):
    """Baseline output mirrors graph.answer so eval metrics apply unchanged."""
    monkeypatch.setattr(baseline, "get_hybrid", lambda: _FakeHybrid())
    out = baseline.plain_answer("What is the wire fee?", _principal(), "r1")
    for key in (
        "reply",
        "citations",
        "retrieved",
        "tool_calls",
        "refusal",
        "abstained",
        "answerability",
        "validation",
        "budget",
        "errors",
        "diagnostics",
        "request_id",
    ):
        assert key in out
    assert out["refusal"] is False and out["abstained"] is False
    # no classification machinery → no verdict; the metric scores n/a,
    # never a constant-label value guaranteed by construction
    assert out["answerability"] is None
    assert out["tool_calls"] == []
    assert out["retrieved"]  # merged-index behavior: results came back
    # offline draft echoes snippets + [DOC §section] markers → parsed citations
    assert out["citations"] and out["citations"][0]["doc_id"] == "DOC-001"


def test_plain_answer_acl_and_strategy_passthrough(monkeypatch):
    """use_acl=False must retrieve unfiltered; strategy reaches the retriever."""
    fake = _FakeHybrid()
    monkeypatch.setattr(baseline, "get_hybrid", lambda: fake)
    baseline.plain_answer("q", _principal(), "r1", strategy="dense", use_acl=False)
    assert all(call[1] is None for call in fake.calls)
    assert all(call[2] == "dense" for call in fake.calls)
    fake.calls.clear()
    baseline.plain_answer("q", _principal(), "r1", strategy="hybrid_rerank", use_acl=True)
    assert all(call[1] is not None for call in fake.calls)
    assert all(call[2] == "hybrid_rerank" for call in fake.calls)


def test_plain_answer_top_k_merges_by_score(monkeypatch):
    """Results across collections merge into one global top-k by score."""
    fake = _FakeHybrid()
    monkeypatch.setattr(baseline, "get_hybrid", lambda: fake)
    out = baseline.plain_answer("q", _principal(), "r1", top_k=2)
    assert len(out["retrieved"]) == 2
    assert out["retrieved"][0]["score"] >= out["retrieved"][1]["score"]


def test_rank_key_prefers_rerank_score(monkeypatch):
    """Hybrid-rerank merges must rank by reranker verdict, not raw leg score.

    BM25 ``score`` (~5-30) dwarfs cosine (0-1) — sorting by it across
    collections would bury the reranker's actual ranking.
    """

    class _Scored:
        def search(self, query, collection, acl, strategy=None):
            return {
                "results": [
                    {  # BM25-flavoured hit: huge leg score, weak rerank verdict
                        "doc_id": "DOC-001",
                        "section": "s",
                        "text": "lo",
                        "score": 24.0,
                        "rerank_score": -3.0,
                    },
                    {  # reranker's actual winner despite a tiny cosine score
                        "doc_id": "DOC-002",
                        "section": "s",
                        "text": "hi",
                        "score": 0.4,
                        "rerank_score": 2.0,
                    },
                ],
                "diagnostics": {},
            }

    monkeypatch.setattr(baseline, "get_hybrid", lambda: _Scored())
    out = baseline.plain_answer("q", _principal(), "r1", strategy="hybrid_rerank", top_k=1)
    assert out["retrieved"][0]["doc_id"] == "DOC-002"


# --- compare.py --------------------------------------------------------------


def _report(system, summary, by_cat=None):
    return {"system": system, "summary": summary, "by_category": by_cat or {}, "acl_leaks": 0}


def test_compare_deltas_vs_reference(tmp_path):
    """Delta = system − reference; n/a excluded from subtraction."""
    a = _report("agentic", {"m": 1.0, "p": 0.5})
    b = _report("plain", {"m": 0.4, "p": 0.9})
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    import json

    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    cmp = C.compare_reports([pa, pb])
    assert cmp["systems"] == ["agentic", "plain"]
    assert cmp["delta_vs_ref"]["m"]["plain"] == pytest.approx(-0.6)
    assert cmp["delta_vs_ref"]["p"]["plain"] == pytest.approx(0.4)
    assert cmp["delta_vs_ref"]["m"]["agentic"] == 0.0


def test_compare_na_not_zero(tmp_path):
    """A missing (n/a) metric is carried as None — never treated as 0."""
    import json

    a = _report("agentic", {"m": None})
    b = _report("plain", {"m": 0.3})
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    cmp = C.compare_reports([pa, pb])
    assert cmp["summary"]["m"]["agentic"] is None
    assert cmp["delta_vs_ref"]["m"]["plain"] is None  # no delta against n/a


def test_render_markdown_tables(tmp_path):
    import json

    a = _report("agentic", {"m": 1.0}, {"doc_factual": {"m": 1.0}})
    b = _report("plain", {"m": 0.2}, {"doc_factual": {"m": 0.2}})
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    md = C.render_markdown(C.compare_reports([pa, pb]))
    assert "| m | 1.0 | 0.2 |" in md
    assert "Δ vs agentic" in md
