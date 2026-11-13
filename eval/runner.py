"""Eval runner — executes the copilot over a scenario suite and scores it.

Suites (``--suite``):
- ``golden``  — the generated ~85-question set (eval/golden_set.jsonl)
- ``pr``      — small deterministic PR gate (eval/scenarios/pr.jsonl)
- ``dev``     — manually reviewed development scenarios
- ``holdout`` — held-out scenarios; never used for tuning

``--repeat N`` reruns the suite and reports metric variance; ``--fail-on-leak``
exits non-zero if any row fails the zero-tolerance ACL-leakage check
(release gate).

Author: venkateshwara doijode
"""

import json
import sys
import uuid
from pathlib import Path

import typer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import metrics as M  # noqa: E402
from report import write_report  # noqa: E402

SCENARIOS = ROOT / "eval" / "scenarios"
SUITES = {
    "golden": ROOT / "eval" / "golden_set.jsonl",
    "pr": SCENARIOS / "pr.jsonl",
    "dev": SCENARIOS / "dev.jsonl",
    "holdout": SCENARIOS / "holdout.jsonl",
}


def _load_rows(path: Path) -> list[dict]:
    """Parse a JSONL scenario file into a list of question rows."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _score_rows(rows: list[dict], users: dict, entitlements: dict, run_ragas: bool) -> dict:
    """Run every row through the graph once and compute per-row + aggregate metrics."""
    from app.agent.graph import answer
    from app.core.logging import get_logger
    from app.security.authn import authenticate

    log = get_logger("eval")
    per_row, ragas_samples = [], []
    for row in rows:
        principal = authenticate(users[row["user_id"]]["api_key"], row["user_id"])
        output = answer(
            row["question"], principal, request_id=f"eval-{row['id']}-{uuid.uuid4().hex[:6]}"
        )
        m = {
            **M.citation_metrics(output, row),
            **M.retrieval_metrics(output, row),
            **M.claim_support_metrics(output),
            "refusal_correct": M.refusal_metric(output, row),
            "answerability_correct": M.answerability_metric(output, row),
            "tool_selection": M.tool_selection_metric(output, row),
            "acl_clean": M.acl_leakage_metric(output, row, entitlements),
            "no_banned_content": M.must_not_contain_metric(output, row),
        }
        per_row.append({**row, "output": output, "metrics": m})
        contexts = [r["text"] for r in output.get("retrieved", [])]
        ragas_samples.append(
            {
                "user_input": row["question"],
                "response": output.get("reply", ""),
                "retrieved_contexts": contexts or ["<none>"],
                "reference": row.get("expected_answer", ""),
            }
        )
        log.info("eval.row_done", id=row["id"], refusal=output.get("refusal"))

    result = M.aggregate(per_row)
    if run_ragas:
        result["summary"].update(
            {f"ragas_{k}": v for k, v in M.ragas_metrics(ragas_samples).items()}
        )
    result["n_questions"] = len(per_row)
    result["acl_leaks"] = sum(1 for r in per_row if r["metrics"]["acl_clean"] < 1.0)
    return result


def run_eval(
    suite: str = "golden",
    golden_path: Path | None = None,
    limit: int | None = None,
    repeat: int = 1,
    fail_on_leak: bool = False,
    run_ragas: bool = True,
) -> Path:
    """Execute the suite (optionally N times) and write a dated report.

    Returns the JSON report path; raises SystemExit(1) on ACL leakage when
    ``fail_on_leak`` is set.
    """
    from app.core.config import get_settings
    from app.core.logging import configure_logging, get_logger

    configure_logging()
    log = get_logger("eval")
    s = get_settings()

    if not golden_path and suite not in SUITES:
        raise typer.BadParameter(f"unknown suite {suite!r} — pick {sorted(SUITES)}")
    dataset_path = golden_path or SUITES[suite]
    users = {u["user_id"]: u for u in json.loads(Path(s.users_file).read_text())}
    entitlements = {uid: set(u["client_ids"]) for uid, u in users.items()}
    rows = _load_rows(dataset_path)[: limit or None]

    run_results = [
        _score_rows(rows, users, entitlements, run_ragas=run_ragas and i == 0)
        for i in range(repeat)
    ]
    result = run_results[0]
    result["version_snapshot"] = M.version_snapshot(dataset_path, len(rows))
    if repeat > 1:
        result["variance"] = M.variance_summary([r["summary"] for r in run_results])
        result["runs"] = repeat
    report_path = write_report(result)
    log.info(
        "eval.done",
        suite=suite,
        rows=len(rows),
        repeats=repeat,
        acl_leaks=result.get("acl_leaks", 0),
    )
    if fail_on_leak and result.get("acl_leaks"):
        log.error("eval.acl_leakage", leaks=result["acl_leaks"])
        raise SystemExit(1)
    return report_path


def main(
    suite: str = "golden",
    golden: Path | None = None,
    limit: int = 0,
    repeat: int = 1,
    fail_on_leak: bool = False,
    ragas: bool = True,
):
    """Run an eval suite and write the report (prints the JSON report path).

    CLI: ``--suite pr|dev|holdout|golden --limit N --repeat N
    --fail-on-leak --no-ragas``.
    """
    print(run_eval(suite, golden, limit or None, repeat, fail_on_leak, ragas))


if __name__ == "__main__":
    typer.run(main)
