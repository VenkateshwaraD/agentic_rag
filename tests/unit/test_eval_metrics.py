"""Unit tests for deterministic eval metrics — N/A semantics, key_points,
gates loader, and deterministic threshold checks.

Author: venkateshwara doijode
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

import metrics as M  # noqa: E402
import ragas_config as RC  # noqa: E402

# --- citation_metrics N/A semantics ---------------------------------------


def test_citation_metrics_no_required_is_na():
    """Rows with no required docs: metric undefined, not 1.0 or 0.0."""
    out = {"citations": [{"doc_id": "AGR-002"}]}
    m = M.citation_metrics(out, {"required_doc_ids": []})
    assert m == {"citation_precision": None, "citation_recall": None}


def test_citation_metrics_uncited_is_precision_na():
    """Required docs but nothing cited (abstain/refusal): precision
    undefined, recall a real miss — no double-punishment."""
    m = M.citation_metrics({"citations": []}, {"required_doc_ids": ["AGR-002"]})
    assert m["citation_precision"] is None
    assert m["citation_recall"] == 0.0


def test_citation_metrics_normal_scoring():
    """Standard path unchanged: precision/recall over cited ∩ required."""
    out = {"citations": [{"doc_id": "AGR-002"}, {"doc_id": "XXX-9"}]}
    m = M.citation_metrics(out, {"required_doc_ids": ["AGR-002"]})
    assert m["citation_precision"] == 0.5
    assert m["citation_recall"] == 1.0


# --- key_points_metric -----------------------------------------------------


def test_key_points_all_hit():
    """Every key point present (case-insensitive) → 1.0."""
    out = {"reply": "The fee is $12.73 per item; status In_Progress."}
    assert M.key_points_metric(out, {"key_points": ["12.73", "in_progress"]}) == 1.0


def test_key_points_partial():
    """Only some key points found → fractional score."""
    out = {"reply": "The fee is $12.73 per item."}
    assert M.key_points_metric(out, {"key_points": ["12.73", "monthly"]}) == 0.5


def test_key_points_none_defined():
    """Row without key_points → None (n/a), not a free 1.0."""
    assert M.key_points_metric({"reply": "x"}, {"key_points": []}) is None
    assert M.key_points_metric({"reply": "x"}, {}) is None


def test_key_points_refusal_is_na():
    """A refused/abstained answer doesn't get scored on key points —
    refusal correctness is its own metric."""
    assert M.key_points_metric({"reply": "no", "refusal": True}, {"key_points": ["12.73"]}) is None
    assert (
        M.key_points_metric({"reply": "no", "abstained": True}, {"key_points": ["12.73"]}) is None
    )


# --- answerability_metric --------------------------------------------------


def test_answerability_no_verdict_is_na():
    """A system without classification machinery emits no verdict → n/a,
    not a fabricated score (same convention as claim-support metrics)."""
    row = {"expected_answerability": "fully_answerable"}
    assert M.answerability_metric({"answerability": None}, row) is None
    assert M.answerability_metric({}, row) is None


def test_answerability_verdict_still_scored():
    """Verdicts score normally when a system does classify."""
    row = {"expected_answerability": "conflicting_evidence"}
    assert M.answerability_metric({"answerability": "conflicting_evidence"}, row) == 1.0
    assert M.answerability_metric({"answerability": "fully_answerable"}, row) == 0.0
    assert (
        M.answerability_metric(
            {"answerability": "unauthorized"},
            {"expected_answerability": "unauthorized", "must_refuse": True},
        )
        == 1.0
    )
    # rows without an expectation don't penalize a system that does classify
    assert M.answerability_metric({"answerability": "fully_answerable"}, {}) == 1.0


# --- aggregate None-safety ------------------------------------------------


def test_aggregate_excludes_none_from_means():
    """n/a rows neither inflate nor deflate the mean; coverage recorded."""
    per_row = [
        {
            "id": "a",
            "category": "c",
            "question": "q",
            "output": {},
            "metrics": {"m": 1.0, "p": 0.5},
        },
        {
            "id": "b",
            "category": "c",
            "question": "q",
            "output": {},
            "metrics": {"m": None, "p": None},
        },
    ]
    agg = M.aggregate(per_row)
    assert agg["summary"]["m"] == 1.0  # mean over the 1 defined row only
    assert agg["summary"]["p"] == 0.5
    assert agg["summary"]["m_n"] == 1  # coverage count
    assert agg["summary"]["p_n"] == 1


def test_aggregate_all_none_is_null_not_crash():
    """A metric n/a on every row → null summary, not ZeroDivisionError."""
    agg = M.aggregate(
        [{"id": "a", "category": "c", "question": "q", "output": {}, "metrics": {"m": None}}]
    )
    # key never enters numeric_keys → absent entirely
    assert "m" not in agg["summary"]


# --- gates loader + deterministic threshold checks ------------------------


def test_load_gates_missing_file(tmp_path):
    """Absent gates file → empty dict (no deterministic gates)."""
    assert RC.load_gates(tmp_path / "nope.yaml") == {}


def test_load_gates_parses_floors(tmp_path):
    """YAML floors parse to floats keyed by metric name."""
    p = tmp_path / "gates.yaml"
    p.write_text("refusal_correct: 1.0\nkey_points_hit: 0.8\n")
    assert RC.load_gates(p) == {"refusal_correct": 1.0, "key_points_hit": 0.8}


def test_load_gates_rejects_non_numeric(tmp_path):
    """Non-numeric gate values fail loudly."""
    p = tmp_path / "gates.yaml"
    p.write_text("refusal_correct: high\n")
    with pytest.raises(ValueError, match="numeric"):
        RC.load_gates(p)


def test_threshold_checks_deterministic_prefix():
    """prefix="" compares plain summary keys; absent → passed None."""
    checks = RC.threshold_checks(
        {"refusal_correct": 1.0, "key_points_hit": 0.5},
        {"refusal_correct": 1.0, "key_points_hit": 0.8, "tool_selection": 0.9},
        prefix="",
    )
    assert checks["refusal_correct"]["passed"] is True
    assert checks["key_points_hit"]["passed"] is False
    assert checks["tool_selection"]["passed"] is None
