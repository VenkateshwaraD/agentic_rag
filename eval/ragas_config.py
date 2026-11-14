"""Ragas judge configuration loader — reads ``eval/ragas.yaml``.

Keeps the Ragas knobs out of code: which metrics to run, which model
judges them, runtime limits (timeout/retries/workers), and optional
quality-gate thresholds for release gating. A missing file yields the
defaults — the eval behaves exactly as before.

Secrets never live in this file: ``judge.api_key_env`` names the
environment variable the key is read from.

Author: venkateshwara doijode
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Every metric the loader knows how to resolve inside ``ragas_metrics``.
KNOWN_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "ragas.yaml"


@dataclass(frozen=True)
class RagasConfig:
    """Resolved Ragas settings — the single object eval code consumes."""

    enabled: bool = True
    model: str = ""  # blank → Settings.anthropic_model
    api_key_env: str = "ANTHROPIC_API_KEY"
    metrics: tuple[str, ...] = KNOWN_METRICS
    timeout_s: int = 180
    max_retries: int = 3
    max_wait_s: int = 60
    max_workers: int = 8
    seed: int = 42
    thresholds: dict[str, float] = field(default_factory=dict)

    def api_key(self, fallback: str = "") -> str:
        """Resolve the judge key: named env var first, then app settings."""
        return os.environ.get(self.api_key_env, "") or fallback


def load_ragas_config(path: Path | None = None) -> RagasConfig:
    """Load ``ragas.yaml`` into a ``RagasConfig``; defaults when absent.

    Raises ``ValueError`` on unknown metric names or malformed YAML so
    config mistakes surface loudly at run start, not as silent skips.
    """
    path = path or DEFAULT_CONFIG_PATH
    if not path.exists():
        return RagasConfig()

    import yaml  # PyYAML — declared dependency, also pulled in transitively

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level mapping expected, got {type(raw).__name__}")

    judge = raw.get("judge") or {}
    run = raw.get("run") or {}
    metrics = tuple(raw.get("metrics") or KNOWN_METRICS)
    thresholds = {k: float(v) for k, v in (raw.get("thresholds") or {}).items()}

    unknown = [m for m in (*metrics, *thresholds) if m not in KNOWN_METRICS]
    if unknown:
        raise ValueError(
            f"{path}: unknown ragas metric(s) {unknown} — choose from {list(KNOWN_METRICS)}"
        )

    return RagasConfig(
        enabled=bool(raw.get("enabled", True)),
        model=str(judge.get("model") or ""),
        api_key_env=str(judge.get("api_key_env") or "ANTHROPIC_API_KEY"),
        metrics=metrics,
        timeout_s=int(run.get("timeout_s", 180)),
        max_retries=int(run.get("max_retries", 3)),
        max_wait_s=int(run.get("max_wait_s", 60)),
        max_workers=int(run.get("max_workers", 8)),
        seed=int(run.get("seed", 42)),
        thresholds=thresholds,
    )


DEFAULT_GATES_PATH = Path(__file__).resolve().parent / "gates.yaml"


def load_gates(path: Path | None = None) -> dict[str, float]:
    """Load deterministic quality-gate floors from ``eval/gates.yaml``.

    A missing file yields ``{}`` — no deterministic gates. Values are
    floors on the report's summary means; malformed YAML or non-numeric
    values raise ``ValueError`` so config mistakes surface loudly.
    """
    path = path or DEFAULT_GATES_PATH
    if not path.exists():
        return {}

    import yaml  # PyYAML — declared dependency

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level mapping expected, got {type(raw).__name__}")
    try:
        return {str(k): float(v) for k, v in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: gate values must be numeric — {exc}") from exc


def threshold_checks(summary: dict, thresholds: dict[str, float], prefix: str = "ragas_") -> dict:
    """Compare ``<prefix><metric>`` summary scores against configured floors.

    Returns ``{metric: {value, minimum, passed}}``; a metric is
    ``passed: None`` when the score is absent (judge didn't run, or the
    metric was n/a on every row) so the caller can decide whether
    missing data is a failure. Pass ``prefix=""`` for deterministic
    metrics (gates.yaml); Ragas thresholds use the default ``ragas_``.
    """
    return {
        name: {
            "value": summary.get(f"{prefix}{name}"),
            "minimum": floor,
            "passed": (
                None
                if summary.get(f"{prefix}{name}") is None
                else bool(summary[f"{prefix}{name}"] >= floor)
            ),
        }
        for name, floor in thresholds.items()
    }
