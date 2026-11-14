"""Report writer — dated JSON + Markdown under eval/reports/.

Author: venkateshwara doijode
"""

import json
from datetime import UTC, datetime
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


def write_report(result: dict) -> Path:
    """Write the eval result as timestamped JSON + Markdown files.

    The Markdown file contains summary, per-category, and failure tables.
    Returns the JSON report path.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"eval_{ts}.json"
    md_path = REPORTS_DIR / f"eval_{ts}.md"

    json_path.write_text(json.dumps(result, indent=2, default=str))

    lines = [f"# Eval report — {ts}", ""]
    lines.append(f"Questions: {result.get('n_questions')}")
    lines.append(f"Runs: {result.get('runs', 1)}")
    lines.append(f"ACL leaks (zero tolerance): {result.get('acl_leaks', 0)}")
    vs = result.get("version_snapshot") or {}
    if vs:
        lines.append("\n## Version snapshot\n")
        emb = vs.get("embedding", {})
        lines.append(f"- model: {vs.get('model')}")
        if vs.get("judge_model"):
            lines.append(f"- judge model: {vs.get('judge_model')}")
        lines.append(
            f"- embedding: {emb.get('provider')}/{emb.get('model')} (dim {emb.get('dim')})"
        )
        lines.append(
            f"- dataset: {vs.get('dataset', {}).get('path')} "
            f"({vs.get('dataset', {}).get('rows')} rows, sha {vs.get('dataset', {}).get('sha256')})"
        )
        lines.append(f"- git: {vs.get('git_sha')}")
        for name, h in (vs.get("prompts") or {}).items():
            lines.append(f"- prompt {name}: {h}")

    def _fmt(v):
        return "n/a" if v is None else v

    summary = result.get("summary", {})
    metric_keys = [k for k in summary if not k.endswith("_n")]
    lines.append("\n## Summary\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in metric_keys:
        lines.append(f"| {k} | {_fmt(summary.get(k))} |")
    lines.append("\n## By category\n")
    lines.append("| category | n | " + " | ".join(metric_keys) + " |")
    sep = "|---|---|" + "---|" * len(metric_keys)
    lines.append(sep)
    for cat, stats in result.get("by_category", {}).items():
        vals = " | ".join(str(_fmt(stats.get(k, ""))) for k in metric_keys)
        lines.append(f"| {cat} | {stats.get('n')} | {vals} |")
    for key, title in (
        ("gate_checks", "Deterministic quality gates"),
        ("threshold_checks", "Ragas quality gates"),
    ):
        checks = result.get(key)
        if checks:
            lines.append(f"\n## {title}\n")
            lines.append("| metric | score | minimum | result |")
            lines.append("|---|---|---|---|")
            for name, c in checks.items():
                verdict = {True: "pass", False: "**FAIL**", None: "not run"}[c["passed"]]
                value = c["value"] if c["value"] is not None else "—"
                lines.append(f"| {name} | {value} | {c['minimum']} | {verdict} |")
    variance = result.get("variance")
    if variance:
        lines.append("\n## Variance across runs\n")
        lines.append("| metric | mean | stdev | 95% CI | n |")
        lines.append("|---|---|---|---|---|")
        for k, v in variance.items():
            lines.append(f"| {k} | {v['mean']} | {v['stdev']} | ±{v['ci95']} | {v['n']} |")
    lines.append("\n## Failures\n")
    for f in result.get("failures", [])[:25]:
        lines.append(f"- **{f['id']}** ({f['category']}): {f['question']}")
        lines.append(f"  - reply: {f['reply']!r}")
        lines.append(f"  - metrics: {f['metrics']}")
    md_path.write_text("\n".join(lines))
    return json_path
