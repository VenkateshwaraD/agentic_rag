"""Eval metrics — Ragas (LLM-judged) + custom deterministic checks.

Ragas metrics (need ANTHROPIC_API_KEY + embeddings): faithfulness,
answer_relevancy, context_precision, context_recall, response_correctness.
Custom checks (always run): citation P/R, per-stage retrieval recall,
refusal accuracy, tool-selection correctness, ACL-leakage and
must_not_contain (zero tolerance).
"""

from collections import defaultdict


def _recall(actual_ids: list, required: set) -> float:
    if not required:
        return 1.0
    return len(set(actual_ids) & required) / len(required)


def citation_metrics(output: dict, row: dict) -> dict:
    cited = {c["doc_id"] for c in output.get("citations", [])}
    required = set(row.get("required_doc_ids", []))
    if not required:
        return {"citation_precision": 1.0 if not cited else 0.0, "citation_recall": 1.0}
    prec = len(cited & required) / len(cited) if cited else 0.0
    rec = len(cited & required) / len(required)
    return {"citation_precision": prec, "citation_recall": rec}


def retrieval_metrics(output: dict, row: dict) -> dict:
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
    if row.get("must_refuse"):
        return 1.0 if output.get("refusal") else 0.0
    return 1.0 if not output.get("refusal") else 0.0


def tool_selection_metric(output: dict, row: dict) -> float:
    required = set(row.get("required_tool_calls", []))
    if not required:
        return 1.0
    actual = {t["tool"] for t in output.get("tool_calls", [])}
    return len(required & actual) / len(required)


def acl_leakage_metric(output: dict, row: dict, entitlements: dict) -> float:
    """1.0 if no retrieved doc/tool row leaks a non-entitled, non-wildcard
    client. Zero-tolerance check."""
    user_clients = set(entitlements.get(row["user_id"], set()))
    for r in output.get("retrieved", []):
        cid = r.get("client_id", "")
        if cid and cid not in user_clients:
            return 0.0
    return 1.0


def must_not_contain_metric(output: dict, row: dict) -> float:
    banned = row.get("must_not_contain", [])
    if not banned:
        return 1.0
    reply = output.get("reply", "")
    return 0.0 if any(b in reply for b in banned) else 1.0


def ragas_metrics(samples: list[dict]) -> dict:
    """Run Ragas over collected samples; returns {} if ragas/creds missing."""
    try:
        from datasets import Dataset
        from langchain_anthropic import ChatAnthropic
        from ragas import evaluate
        from ragas.llms import LangchainLLMWrapper

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

        s = get_settings()
        if not s.anthropic_api_key:
            return {}
        ds = Dataset.from_list(samples)
        result = evaluate(
            ds,
            metrics=[
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_correctness,
            ],
            llm=LangchainLLMWrapper(
                ChatAnthropic(model=s.anthropic_model, api_key=s.anthropic_api_key)
            ),
            embeddings=LangchainEmbeddingsWrapper(get_embedder()),
        )
        return {k: float(v) for k, v in result.items()}
    except Exception as exc:  # ragas API drift, missing creds, etc.
        return {"ragas_error": str(exc)}


def aggregate(per_row: list[dict]) -> dict:
    """Mean metrics overall + per category."""
    numeric_keys = {
        k for r in per_row for k, v in r["metrics"].items() if isinstance(v, (int, float))
    }
    summary = {
        k: round(sum(r["metrics"].get(k, 0) for r in per_row) / max(len(per_row), 1), 4)
        for k in numeric_keys
    }
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in per_row:
        by_cat[r["category"]].append(r)
    by_category = {
        cat: {
            "n": len(rs),
            **{
                k: round(sum(r["metrics"].get(k, 0) for r in rs) / len(rs), 4) for k in numeric_keys
            },
        }
        for cat, rs in by_cat.items()
    }
    failures = [
        {
            "id": r["id"],
            "category": r["category"],
            "question": r["question"],
            "reply": r["output"].get("reply", "")[:300],
            "metrics": {
                k: v for k, v in r["metrics"].items() if isinstance(v, (int, float)) and v < 1.0
            },
        }
        for r in per_row
        if any(isinstance(v, (int, float)) and v < 1.0 for v in r["metrics"].values())
    ]
    return {"summary": summary, "by_category": by_category, "failures": failures}
