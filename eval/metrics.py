"""Eval metrics — Ragas (LLM-judged) + custom deterministic checks.

Ragas metrics (need ANTHROPIC_API_KEY + embeddings): faithfulness,
answer_relevancy, context_precision, context_recall, response_correctness.
Custom checks (always run): citation P/R, per-stage retrieval recall,
refusal accuracy, tool-selection correctness, ACL-leakage and
must_not_contain (zero tolerance).

Author: venkateshwara doijode
"""

from collections import defaultdict
from pathlib import Path


def _recall(actual_ids: list, required: set) -> float:
    """Fraction of required doc ids present among actual ids (1.0 when none required)."""
    if not required:
        return 1.0
    return len(set(actual_ids) & required) / len(required)


def citation_metrics(output: dict, row: dict) -> dict:
    """Citation precision/recall of the answer's cited doc ids vs. required ones.

    ``None`` (reported as ``n/a``, excluded from averages) marks undefined
    cases instead of fabricated scores: no required docs → nothing to
    measure against; no citations → precision over an empty set is
    undefined (the miss is still scored by recall).
    """
    cited = {c["doc_id"] for c in output.get("citations", [])}
    required = set(row.get("required_doc_ids", []))
    if not required:
        return {"citation_precision": None, "citation_recall": None}
    if not cited:
        return {"citation_precision": None, "citation_recall": 0.0}
    prec = len(cited & required) / len(cited)
    rec = len(cited & required) / len(required)
    return {"citation_precision": prec, "citation_recall": rec}


def key_points_metric(output: dict, row: dict) -> float | None:
    """Fraction of the row's ``key_points`` substrings present in the reply.

    A cheap deterministic correctness check — key points carry the facts
    that must appear in a correct answer (fees, statuses, dates, names).
    ``None`` when the row defines no key points or the answer
    refused/abstained (refusal correctness is scored separately).
    """
    key_points = row.get("key_points", [])
    if not key_points or output.get("refusal") or output.get("abstained"):
        return None
    reply = output.get("reply", "").casefold()
    return sum(1 for kp in key_points if str(kp).casefold() in reply) / len(key_points)


def retrieval_metrics(output: dict, row: dict) -> dict:
    """Per-stage recall (dense / BM25 / fused / reranked) against required doc ids."""
    required = set(row.get("required_doc_ids", []))
    diag = output.get("diagnostics", {})
    out = {}
    for stage in ("dense_ids", "bm25_ids", "fused_ids", "reranked_ids"):
        ids = []
        for coll_diag in diag.values():
            ids += coll_diag.get(stage, [])
        out[f"recall_{stage.replace('_ids', '')}"] = _recall(ids, required)
    return out


def refusal_metric(output: dict, row: dict) -> float:
    """1.0 when the refusal flag matches the row's ``must_refuse`` expectation."""
    if row.get("must_refuse"):
        return 1.0 if output.get("refusal") else 0.0
    return 1.0 if not output.get("refusal") else 0.0


def tool_selection_metric(output: dict, row: dict) -> float:
    """Fraction of required ops tools that were actually invoked."""
    required = set(row.get("required_tool_calls", []))
    if not required:
        return 1.0
    actual = {t["tool"] for t in output.get("tool_calls", [])}
    return len(required & actual) / len(required)


def acl_leakage_metric(output: dict, row: dict, entitlements: dict) -> float:
    """1.0 if no retrieved doc/tool row leaks a non-entitled, non-wildcard
    client. Zero-tolerance check — covers retrieved chunks, tool-call
    client scopes, and the reply text itself."""
    user_clients = set(entitlements.get(row["user_id"], set()))
    for r in output.get("retrieved", []):
        cid = r.get("client_id", "")
        if cid and cid not in user_clients:
            return 0.0
    for tc in output.get("tool_calls", []):
        cid = tc.get("client_id")
        if cid and cid not in user_clients:
            return 0.0
    # other clients' ids must not appear in the reply at all
    reply = output.get("reply", "")
    all_clients = {c for clients in entitlements.values() for c in clients}
    for cid in all_clients - user_clients:
        if cid and cid in reply:
            return 0.0
    return 1.0


def answerability_metric(output: dict, row: dict) -> float:
    """1.0 when the output's answerability matches the row's expectation.

    Tolerance: a ``must_refuse`` row accepts ``insufficient_evidence`` or
    ``unauthorized``; rows without an expectation score 1.0.
    """
    expected = row.get("expected_answerability")
    if not expected:
        return 1.0
    actual = output.get("answerability", "")
    if actual == expected:
        return 1.0
    if row.get("must_refuse") and actual in ("insufficient_evidence", "unauthorized"):
        return 1.0
    return 0.0


def claim_support_metrics(output: dict) -> dict:
    """Derive citation correctness/completeness + unsupported-claim rate
    from the validate node's claim-level record (n/a when no validation ran —
    a system without claim checking gets no credit here, not a perfect score)."""
    v = output.get("validation") or {}
    if not v:
        return {
            "citation_correctness": None,
            "citation_completeness": None,
            "unsupported_claim_rate": None,
        }
    claims = v.get("claims", [])
    citations = output.get("citations", [])
    invalid = len(v.get("invalid_citations", []))
    return {
        "citation_correctness": (
            1.0 - invalid / max(len(citations), 1) if citations or invalid else None
        ),
        "citation_completeness": (
            sum(1 for c in claims if c.get("cited")) / len(claims) if claims else None
        ),
        "unsupported_claim_rate": (
            len(v.get("unsupported_claims", [])) / len(claims) if claims else None
        ),
    }


def must_not_contain_metric(output: dict, row: dict) -> float:
    """0.0 when any banned substring appears in the reply, else 1.0."""
    banned = row.get("must_not_contain", [])
    if not banned:
        return 1.0
    reply = output.get("reply", "")
    return 0.0 if any(b in reply for b in banned) else 1.0


def ragas_metrics(samples: list[dict], cfg=None) -> dict:
    """Run Ragas over collected samples; returns {} if disabled/creds missing.

    ``cfg`` is a ``ragas_config.RagasConfig`` (defaults are loaded from
    ``eval/ragas.yaml`` when not supplied) — it selects the metric subset,
    judge model, run limits, and the env var the API key is read from.
    """
    if cfg is None:
        import ragas_config

        cfg = ragas_config.load_ragas_config()
    if not cfg.enabled:
        return {}
    try:
        from datasets import Dataset
        from langchain_anthropic import ChatAnthropic
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper
        from ragas.run_config import RunConfig

        try:
            from ragas.metrics.collections import (
                answer_correctness,
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )
        except ImportError:  # ragas < 0.4
            from ragas.metrics import (
                answer_correctness,
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

        from ragas.embeddings import LangchainEmbeddingsWrapper

        from app.core.config import get_settings
        from app.core.dependencies import get_embedder

        all_metrics = {
            "faithfulness": faithfulness,
            "answer_relevancy": answer_relevancy,
            "context_precision": context_precision,
            "context_recall": context_recall,
            "answer_correctness": answer_correctness,
        }
        chosen = [all_metrics[name] for name in cfg.metrics if name in all_metrics]
        if not chosen:
            return {}

        s = get_settings()
        api_key = cfg.api_key(fallback=s.anthropic_api_key)
        if not api_key:
            return {}
        judge_model = cfg.model or s.anthropic_model
        if judge_model == s.anthropic_model:
            from app.core.logging import get_logger

            get_logger("eval").warning(
                "eval.ragas_self_judge",
                model=judge_model,
                note="judge model == answer model — scores carry self-preference bias; "
                "set judge.model in eval/ragas.yaml to a different model",
            )
        ds = Dataset.from_list(samples)
        result = evaluate(
            ds,
            metrics=chosen,
            llm=LangchainLLMWrapper(ChatAnthropic(model=judge_model, api_key=api_key)),
            embeddings=LangchainEmbeddingsWrapper(get_embedder()),
            run_config=RunConfig(
                timeout=cfg.timeout_s,
                max_retries=cfg.max_retries,
                max_wait=cfg.max_wait_s,
                max_workers=cfg.max_workers,
                seed=cfg.seed,
            ),
        )
        return {k: float(v) for k, v in result.items()}
    except Exception as exc:  # ragas API drift, missing creds, etc.
        return {"ragas_error": str(exc)}


def _rel_dataset_path(dataset_path) -> str:
    """Repo-relative dataset path — committed reports must not leak the
    runner's absolute filesystem layout (username, checkout location)."""
    try:
        return str(Path(dataset_path).resolve().relative_to(Path(__file__).resolve().parents[1]))
    except ValueError:
        return Path(dataset_path).name


def version_snapshot(dataset_path, rows: int) -> dict:
    """Capture model/prompt/embedding/index/dataset versions for the report.

    Every eval run records exactly which components produced its scores so
    results are comparable across changes.
    """
    import hashlib
    import subprocess
    from pathlib import Path

    from app.core.config import get_settings

    s = get_settings()
    prompts_dir = Path(__file__).resolve().parents[1] / "src" / "app" / "agent" / "prompts"
    prompt_hashes = (
        {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()[:12]
            for p in sorted(prompts_dir.glob("*.md"))
        }
        if prompts_dir.is_dir()
        else {}
    )
    try:
        git_sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except Exception:
        git_sha = ""
    data = dataset_path.read_bytes() if dataset_path.exists() else b""
    return {
        "model": s.anthropic_model,
        "llm_configured": bool(s.anthropic_api_key),
        "embedding": {
            "provider": s.embedding_provider,
            "model": (
                s.local_embed_model if s.embedding_provider == "local" else s.bedrock_embed_model
            ),
            "dim": s.embedding_dim,
        },
        "prompts": prompt_hashes,
        "dataset": {
            "path": _rel_dataset_path(dataset_path),
            "rows": rows,
            "sha256": hashlib.sha256(data).hexdigest()[:16] if data else "",
        },
        "git_sha": git_sha,
    }


def variance_summary(run_summaries: list[dict]) -> dict:
    """Mean ± stdev + 95% CI per metric across repeated eval runs."""
    import math
    import statistics

    keys = {
        k
        for s in run_summaries
        for k, v in s.items()
        if isinstance(v, (int, float)) and not k.endswith("_n")
    }
    out = {}
    for k in sorted(keys):
        vals = [s[k] for s in run_summaries if isinstance(s.get(k), (int, float))]
        if not vals:
            continue
        mean = statistics.mean(vals)
        sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
        ci = 1.96 * sd / math.sqrt(len(vals)) if len(vals) > 1 else 0.0
        out[k] = {
            "mean": round(mean, 4),
            "stdev": round(sd, 4),
            "ci95": round(ci, 4),
            "n": len(vals),
        }
    return out


_LOWER_IS_BETTER = {"unsupported_claim_rate"}


def aggregate(per_row: list[dict]) -> dict:
    """Mean metrics overall + per category.

    ``None`` values are ``n/a`` — excluded from means so undefined metrics
    (e.g. citation precision on a refusal) neither inflate nor deflate
    the score. ``<metric>_n`` records how many rows each mean covers.
    """
    numeric_keys = {
        k for r in per_row for k, v in r["metrics"].items() if isinstance(v, (int, float))
    }

    def _mean(rows: list[dict], key: str) -> float | None:
        vals = [r["metrics"][key] for r in rows if isinstance(r["metrics"].get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {k: _mean(per_row, k) for k in numeric_keys}
    summary.update(
        {
            f"{k}_n": sum(1 for r in per_row if isinstance(r["metrics"].get(k), (int, float)))
            for k in numeric_keys
        }
    )
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in per_row:
        by_cat[r["category"]].append(r)
    by_category = {
        cat: {
            "n": len(rs),
            **{k: _mean(rs, k) for k in numeric_keys},
        }
        for cat, rs in by_cat.items()
    }

    # most metrics are higher-is-better (flag when < 1.0); rates like
    # unsupported_claim_rate are lower-is-better (flag when > 0)
    def _is_bad(key: str, v: float) -> bool:
        return v > 0 if key in _LOWER_IS_BETTER else v < 1.0

    failures = [
        {
            "id": r["id"],
            "category": r["category"],
            "question": r["question"],
            "reply": (r.get("output") or {}).get("reply", ""),
            "expected": {
                k: r[k]
                for k in (
                    "expected_answerability",
                    "must_refuse",
                    "key_points",
                    "required_doc_ids",
                    "must_not_contain",
                )
                if r.get(k) not in (None, "", [], False)
            },
            "actual": {
                "answerability": (r.get("output") or {}).get("answerability"),
                "refusal": (r.get("output") or {}).get("refusal"),
                "abstained": (r.get("output") or {}).get("abstained"),
                "cost_usd": ((r.get("output") or {}).get("budget") or {}).get("cost_usd"),
                "llm_calls": ((r.get("output") or {}).get("budget") or {}).get("llm_calls"),
            },
            "metrics": {
                k: v
                for k, v in r["metrics"].items()
                if isinstance(v, (int, float)) and _is_bad(k, v)
            },
        }
        for r in per_row
        if any(isinstance(v, (int, float)) and _is_bad(k, v) for k, v in r["metrics"].items())
    ]
    row_results = [
        {
            "id": r["id"],
            "category": r["category"],
            "question": r["question"],
            "answerability": (r.get("output") or {}).get("answerability"),
            "refusal": (r.get("output") or {}).get("refusal"),
            "abstained": (r.get("output") or {}).get("abstained"),
            "cost_usd": ((r.get("output") or {}).get("budget") or {}).get("cost_usd"),
            "llm_calls": ((r.get("output") or {}).get("budget") or {}).get("llm_calls"),
            "tokens_used": ((r.get("output") or {}).get("budget") or {}).get("tokens_used"),
            "failed_metrics": [
                k for k, v in r["metrics"].items() if isinstance(v, (int, float)) and _is_bad(k, v)
            ],
        }
        for r in per_row
    ]
    return {
        "summary": summary,
        "by_category": by_category,
        "failures": failures,
        "rows": row_results,
    }
