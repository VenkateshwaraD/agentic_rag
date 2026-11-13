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
    lines.append("\n## Summary\n")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k, v in result.get("summary", {}).items():
        lines.append(f"| {k} | {v} |")
    lines.append("\n## By category\n")
    lines.append("| category | n | " + " | ".join(result.get("summary", {}).keys()) + " |")
    sep = "|---|---|" + "---|" * len(result.get("summary", {}))
    lines.append(sep)
    for cat, stats in result.get("by_category", {}).items():
        vals = " | ".join(str(stats.get(k, "")) for k in result.get("summary", {}))
        lines.append(f"| {cat} | {stats.get('n')} | {vals} |")
    lines.append("\n## Failures\n")
    for f in result.get("failures", [])[:25]:
        lines.append(f"- **{f['id']}** ({f['category']}): {f['question']}")
        lines.append(f"  - reply: {f['reply']!r}")
        lines.append(f"  - metrics: {f['metrics']}")
    md_path.write_text("\n".join(lines))
    return json_path
