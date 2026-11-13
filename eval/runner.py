"""Eval runner — executes the copilot over the golden set and scores it."""

import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "eval"))

import metrics as M  # noqa: E402
from report import write_report  # noqa: E402


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_eval(golden_path: Path, limit: int | None = None) -> Path:
    from app.agent.graph import answer
    from app.core.config import get_settings
    from app.core.logging import configure_logging, get_logger
    from app.security.authn import authenticate

    configure_logging()
    log = get_logger("eval")
    s = get_settings()

    users = {u["user_id"]: u for u in json.loads(Path(s.users_file).read_text())}
    entitlements = {uid: set(u["client_ids"]) for uid, u in users.items()}

    rows = _load_rows(golden_path)[: limit or None]
    per_row, ragas_samples = [], []

    for row in rows:
        principal = authenticate(users[row["user_id"]]["api_key"], row["user_id"])
        output = answer(
            row["question"], principal, request_id=f"eval-{row['id']}-{uuid.uuid4().hex[:6]}"
        )
        m = {
            **M.citation_metrics(output, row),
            **M.retrieval_metrics(output, row),
            "refusal_correct": M.refusal_metric(output, row),
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

    ragas = M.ragas_metrics(ragas_samples)
    result = M.aggregate(per_row)
    result["summary"].update({f"ragas_{k}": v for k, v in ragas.items()})
    result["n_questions"] = len(per_row)
    return write_report(result)


if __name__ == "__main__":
    import typer

    typer.run(
        lambda golden=ROOT / "eval" / "golden_set.jsonl", limit=0: print(
            run_eval(Path(golden), limit or None)
        )
    )
