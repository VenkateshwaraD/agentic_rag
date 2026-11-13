"""Unit tests for the Ragas YAML config loader and threshold gate.

Author: venkateshwara doijode
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "eval"))

import ragas_config as RC  # noqa: E402


def _write(tmp_path: Path, body: str) -> Path:
    """Materialize a ragas.yaml in tmp_path and return its path."""
    p = tmp_path / "ragas.yaml"
    p.write_text(body)
    return p


def test_missing_file_returns_defaults(tmp_path):
    """Absent config file yields the stock defaults."""
    cfg = RC.load_ragas_config(tmp_path / "nope.yaml")
    assert cfg.enabled is True
    assert cfg.metrics == RC.KNOWN_METRICS
    assert cfg.api_key_env == "ANTHROPIC_API_KEY"


def test_custom_config_parsed(tmp_path):
    """Full YAML parses into the dataclass fields."""
    p = _write(
        tmp_path,
        """
enabled: false
judge:
  model: claude-haiku-x
  api_key_env: JUDGE_KEY
metrics: [faithfulness, answer_relevancy]
run:
  timeout_s: 60
  max_retries: 1
  max_workers: 4
thresholds:
  faithfulness: 0.8
""",
    )
    cfg = RC.load_ragas_config(p)
    assert cfg.enabled is False
    assert cfg.model == "claude-haiku-x"
    assert cfg.api_key_env == "JUDGE_KEY"
    assert cfg.metrics == ("faithfulness", "answer_relevancy")
    assert cfg.timeout_s == 60 and cfg.max_retries == 1 and cfg.max_workers == 4
    assert cfg.thresholds == {"faithfulness": 0.8}


def test_unknown_metric_rejected(tmp_path):
    """Unknown metric names fail loudly at load time."""
    p = _write(tmp_path, "metrics: [faithfulness, hallucination_index]\n")
    with pytest.raises(ValueError, match="hallucination_index"):
        RC.load_ragas_config(p)


def test_api_key_resolution(tmp_path, monkeypatch):
    """api_key() prefers the named env var, then the fallback."""
    cfg = RC.load_ragas_config(_write(tmp_path, "judge: {api_key_env: JUDGE_KEY}\n"))
    monkeypatch.setenv("JUDGE_KEY", "from-env")
    assert cfg.api_key() == "from-env"
    monkeypatch.delenv("JUDGE_KEY")
    assert cfg.api_key(fallback="from-settings") == "from-settings"


def test_threshold_checks():
    """Score vs floor comparison; missing score → passed None."""
    checks = RC.threshold_checks(
        {"ragas_faithfulness": 0.9, "ragas_context_precision": 0.5},
        {"faithfulness": 0.8, "context_precision": 0.7, "context_recall": 0.5},
    )
    assert checks["faithfulness"]["passed"] is True
    assert checks["context_precision"]["passed"] is False
    assert checks["context_recall"]["passed"] is None
