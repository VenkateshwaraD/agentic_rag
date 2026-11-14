"""Baseline comparison — side-by-side metric deltas across answer systems.

Merges two or more eval reports (``eval/runner.py --system <name>``) into
one table: summary means per system, deltas vs. the first report (the
reference — run ``agentic`` first), and per-category breakdowns for the
metrics where the agent loop is expected to earn its keep (refusals,
abstentions, banned content, key facts, ACL). ``n/a`` values are shown
as such and excluded from delta math — undefined is not zero.

Usage:
    python eval/compare.py eval/reports/eval_A.json eval/reports/eval_plain_B.json ...
    python eval/runner.py --suite dev --system all   # runs + compares in one go

Author: venkateshwara doijode
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# Metrics where the agentic loop should demonstrably differ from plain
# RAG — per-category tables are generated for these only (the full
# metric matrix appears in the summary table anyway).
GAP_METRICS = [
    "answerability_correct",
    "refusal_correct",
    "key_points_hit",
    "acl_clean",
    "no_banned_content",
    "unsupported_claim_rate",
]


def _system_name(report: dict, path: Path) -> str:
    """System label for a report — stored ``system`` field, else filename."""
    return report.get("system") or path.stem


def compare_reports(paths: list[Path]) -> dict:
    """Load reports and build the comparison structure.

    Returns ``{systems, summary, delta_vs_ref, by_category}`` where
    ``delta_vs_ref`` holds each metric's (system − reference) difference.
    ``None`` (n/a) values are carried through and excluded from deltas.
    """
    reports = []
    for p in paths:
        r = json.loads(p.read_text())
        reports.append((r, _system_name(r, p)))
    systems = [name for _, name in reports]
    ref = systems[0]

    metric_keys = {
        k
        for r, _ in reports
        for k, v in (r.get("summary") or {}).items()
        if isinstance(v, (int, float)) and not k.endswith("_n")
    }
    summary = {m: {s: r["summary"].get(m) for r, s in reports} for m in sorted(metric_keys)}
    delta = {}
    for m in sorted(metric_keys):
        base = summary[m][ref]
        delta[m] = {
            s: (
                round(summary[m][s] - base, 4)
                if isinstance(summary[m][s], (int, float)) and isinstance(base, (int, float))
                else None
            )
            for s in systems
        }

    categories = sorted({cat for r, _ in reports for cat in (r.get("by_category") or {})})
    by_category = {
        m: {
            cat: {s: (r.get("by_category") or {}).get(cat, {}).get(m) for r, s in reports}
            for cat in categories
        }
        for m in GAP_METRICS
    }
    return {
        "systems": systems,
        "reference": ref,
        "summary": summary,
        "delta_vs_ref": delta,
        "by_category": by_category,
        "acl_leaks": {s: r.get("acl_leaks") for r, s in reports},
    }


def _fmt(v) -> str:
    """Render a metric cell — 'n/a' for undefined, signed for deltas."""
    return "n/a" if v is None else str(v)


def render_markdown(cmp: dict) -> str:
    """Render the comparison as a Markdown report."""
    ts = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    systems = cmp["systems"]
    ref = cmp["reference"]
    lines = [f"# Baseline comparison — {ts}", ""]
    lines.append(f"Reference (deltas measured against): **{ref}**")
    lines.append(f"ACL leaks: {cmp['acl_leaks']}")
    lines += ["", "## Summary metrics", ""]
    lines.append("| metric | " + " | ".join(systems) + " |")
    lines.append("|" + "---|" * (len(systems) + 1))
    for m, vals in cmp["summary"].items():
        lines.append(f"| {m} | " + " | ".join(_fmt(vals[s]) for s in systems) + " |")

    lines += ["", f"## Δ vs {ref}", ""]
    lines.append("| metric | " + " | ".join(systems) + " |")
    lines.append("|" + "---|" * (len(systems) + 1))
    for m, vals in cmp["delta_vs_ref"].items():
        cells = []
        for s in systems:
            v = vals[s]
            cells.append("—" if v is None else ("+0.0000" if v == 0 else f"{v:+.4f}"))
        lines.append(f"| {m} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-category — metrics where the agent loop should differ", ""]
    for m, cats in cmp["by_category"].items():
        lines += [f"### {m}", "", "| category | " + " | ".join(systems) + " |"]
        lines.append("|" + "---|" * (len(systems) + 1))
        for cat, vals in cats.items():
            lines.append(f"| {cat} | " + " | ".join(_fmt(vals[s]) for s in systems) + " |")
        lines.append("")
    return "\n".join(lines)


def main_compare(paths: list[Path]) -> Path:
    """Compare reports and write compare_<ts>.json + .md; returns the .md path.

    Named ``compare_*`` (not ``eval_*``) so the Streamlit report picker —
    which globs ``eval_*.json`` — never mistakes a comparison for a
    single-system report.
    """
    cmp = compare_reports(paths)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    json_path = REPORTS_DIR / f"compare_{ts}.json"
    md_path = REPORTS_DIR / f"compare_{ts}.md"
    json_path.write_text(json.dumps(cmp, indent=2, default=str))
    md_path.write_text(render_markdown(cmp))
    return md_path


if __name__ == "__main__":
    args = [Path(a) for a in sys.argv[1:]]
    if len(args) < 2:
        sys.exit("usage: python eval/compare.py <report.json> <report.json> [...]")
    print(main_compare(args))
