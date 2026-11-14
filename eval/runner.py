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
import ragas_config as RC  # noqa: E402
from report import write_report  # noqa: E402

SCENARIOS = ROOT / "eval" / "scenarios"
SUITES = {
    "golden": ROOT / "eval" / "golden_set.jsonl",
    "pr": SCENARIOS / "pr.jsonl",
    "dev": SCENARIOS / "dev.jsonl",
    "holdout": SCENARIOS / "holdout.jsonl",
}

# Answer systems under test. ``agentic`` is the full LangGraph pipeline;
# the ``plain-*`` entries are the naive retrieve-then-answer baseline in
# src/app/baseline.py — (retrieval strategy, ACL pre-filter) variants so
# the comparison can isolate the agent-loop delta from retrieval/ACL.
SYSTEMS: dict[str, tuple[str, bool] | None] = {
    "agentic": None,
    "plain": ("dense", False),
    "plain-acl": ("dense", True),
    "plain-hybrid": ("hybrid_rerank", False),
    "plain-hybrid-acl": ("hybrid_rerank", True),
}


def _load_rows(path: Path) -> list[dict]:
    """Parse a JSONL scenario file into a list of question rows."""
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _score_rows(
    rows: list[dict],
    users: dict,
    entitlements: dict,
    run_ragas: bool,
    ragas_cfg: RC.RagasConfig | None = None,
    system: str = "agentic",
) -> dict:
    """Run every row through the chosen answer system and compute metrics."""
    from app.agent.graph import answer
    from app.core.logging import get_logger
    from app.security.authn import authenticate

    log = get_logger("eval")
    per_row, ragas_samples = [], []
    for row in rows:
        principal = authenticate(users[row["user_id"]]["api_key"], row["user_id"])
        request_id = f"eval-{row['id']}-{uuid.uuid4().hex[:6]}"
        if system == "agentic":
            output = answer(row["question"], principal, request_id=request_id)
        else:
            from app.baseline import plain_answer

            strategy, use_acl = SYSTEMS[system]
            output = plain_answer(
                row["question"],
                principal,
                request_id=request_id,
                strategy=strategy,
                use_acl=use_acl,
            )
        m = {
            **M.citation_metrics(output, row),
            **M.retrieval_metrics(output, row),
            **M.claim_support_metrics(output),
            "key_points_hit": M.key_points_metric(output, row),
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
            {f"ragas_{k}": v for k, v in M.ragas_metrics(ragas_samples, ragas_cfg).items()}
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
    ragas_config_path: Path | None = None,
    fail_on_thresholds: bool = False,
    gates_path: Path | None = None,
    system: str = "agentic",
) -> Path:
    """Execute the suite (optionally N times) and write a dated report.

    Returns the JSON report path; raises SystemExit(1) on ACL leakage when
    ``fail_on_leak`` is set, or when any configured Ragas threshold is
    missed under ``fail_on_thresholds``. ``system`` selects the answer
    pipeline — ``agentic`` (default) or a ``plain-*`` baseline variant.
    """
    from app.core.config import get_settings
    from app.core.logging import configure_logging, get_logger

    configure_logging()
    log = get_logger("eval")
    s = get_settings()

    ragas_cfg = RC.load_ragas_config(ragas_config_path)

    if not golden_path and suite not in SUITES:
        raise typer.BadParameter(f"unknown suite {suite!r} — pick {sorted(SUITES)}")
    if system not in SYSTEMS:
        raise typer.BadParameter(f"unknown system {system!r} — pick {sorted(SYSTEMS)}")
    dataset_path = golden_path or SUITES[suite]
    users = {u["user_id"]: u for u in json.loads(Path(s.users_file).read_text())}
    entitlements = {uid: set(u["client_ids"]) for uid, u in users.items()}
    rows = _load_rows(dataset_path)[: limit or None]

    run_results = [
        _score_rows(
            rows,
            users,
            entitlements,
            run_ragas=run_ragas and i == 0,
            ragas_cfg=ragas_cfg,
            system=system,
        )
        for i in range(repeat)
    ]
    result = run_results[0]
    result["system"] = system
    result["version_snapshot"] = M.version_snapshot(dataset_path, len(rows))
    result["version_snapshot"]["judge_model"] = ragas_cfg.model or s.anthropic_model
    if repeat > 1:
        result["variance"] = M.variance_summary([r["summary"] for r in run_results])
        result["runs"] = repeat
    if ragas_cfg.thresholds:
        result["threshold_checks"] = RC.threshold_checks(result["summary"], ragas_cfg.thresholds)
    gates = RC.load_gates(gates_path)
    if gates:
        result["gate_checks"] = RC.threshold_checks(result["summary"], gates, prefix="")
    report_path = write_report(result, tag=None if system == "agentic" else system)
    log.info(
        "eval.done",
        suite=suite,
        system=system,
        rows=len(rows),
        repeats=repeat,
        acl_leaks=result.get("acl_leaks", 0),
    )
    if fail_on_leak and result.get("acl_leaks"):
        log.error("eval.acl_leakage", leaks=result["acl_leaks"])
        raise SystemExit(1)
    if fail_on_thresholds:
        failed = {
            k: c
            for checks in (result.get("threshold_checks", {}), result.get("gate_checks", {}))
            for k, c in checks.items()
            if c["passed"] is False
        }
        if failed:
            log.error("eval.threshold_breach", failed=failed)
            raise SystemExit(1)
    return report_path


def main(
    suite: str = "golden",
    golden: Path | None = None,
    limit: int = 0,
    repeat: int = 1,
    fail_on_leak: bool = False,
    ragas: bool = True,
    ragas_config: Path | None = None,
    fail_on_thresholds: bool = False,
    gates: Path | None = None,
    system: str = "agentic",
):
    """Run an eval suite and write the report (prints the JSON report path).

    CLI: ``--suite pr|dev|holdout|golden --limit N --repeat N
    --fail-on-leak --no-ragas --ragas-config eval/ragas.yaml
    --gates eval/gates.yaml --fail-on-thresholds
    --system agentic|plain|plain-acl|plain-hybrid|plain-hybrid-acl|all``.

    ``--system all`` runs every variant, writes one tagged report each,
    then writes a side-by-side comparison report via eval/compare.py.
    """
    if system == "all":
        # Gate enforcement applies to the agentic run only: no-ACL baseline
        # variants leak *by design* (the leak count is the comparison's
        # result, not a failure) and must not abort the loop mid-run.
        paths = [
            run_eval(
                suite,
                golden,
                limit or None,
                repeat,
                fail_on_leak and name == "agentic",
                ragas,
                ragas_config,
                fail_on_thresholds and name == "agentic",
                gates,
                system=name,
            )
            for name in SYSTEMS
        ]
        import compare

        print(compare.main_compare(paths))
        return
    print(
        run_eval(
            suite,
            golden,
            limit or None,
            repeat,
            fail_on_leak,
            ragas,
            ragas_config,
            fail_on_thresholds,
            gates,
            system=system,
        )
    )


if __name__ == "__main__":
    typer.run(main)
