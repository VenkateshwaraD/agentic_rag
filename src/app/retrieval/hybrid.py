"""Hybrid retrieval: dense + BM25 → RRF fusion → cross-encoder rerank.

Every stage is independently measurable: diagnostics record per-leg ids,
counts, skipped legs, the resolved strategy, the classifier verdict (auto
mode), and expiry/dedup/parent-expansion counters — consumed by the eval
harness to compare dense-only / BM25-only / fused / reranked recall and
to evaluate RRF parameter choices.

Author: venkateshwara doijode
"""

from collections import defaultdict
from typing import Any

from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.core.logging import get_logger
from app.retrieval.freshness import is_expired
from app.retrieval.lexical import LexicalStore
from app.retrieval.query_classifier import MODE_TO_STRATEGY, classify
from app.retrieval.reranker import Reranker
from app.retrieval.vectorstore import VectorStore
from app.security.authz import ACLFilter

log = get_logger(__name__)

STRATEGIES = ("dense", "bm25", "hybrid", "hybrid_rerank", "auto")
_STALE_PENALTY = 0.3  # fusion-score multiplier for expired docs in down_rank mode


def rrf_fuse(
    rankings: list[list[dict]], k: int = 60, weights: list[float] | None = None
) -> list[dict]:
    """Weighted reciprocal rank fusion over multiple ranked result lists.

    ``weights[i]`` scales list ``i``'s contribution — used to evaluate the
    impact of dense-vs-BM25 weighting on fused recall. Items are keyed by
    ``(doc_id, section, chunk_id)``; expired items contribute a reduced
    score when ``stale`` is set by the store layer (down_rank mode).
    """
    weights = weights or [1.0] * len(rankings)
    scores: dict[tuple, float] = defaultdict(float)
    items: dict[tuple, dict] = {}
    for ranked, weight in zip(rankings, weights, strict=True):
        for rank, item in enumerate(ranked, start=1):
            key = (item["doc_id"], item["section"], item.get("chunk_id", ""))
            contribution = weight / (k + rank)
            if item.get("stale"):
                contribution *= _STALE_PENALTY
            scores[key] += contribution
            merged = items.setdefault(key, item)
            if item.get("stale"):
                merged["stale"] = True
    fused = [{**items[key], "rrf_score": score} for key, score in scores.items()]
    return sorted(fused, key=lambda c: c["rrf_score"], reverse=True)


class HybridRetriever:
    """Runs the configured retrieval strategy over dense + lexical stores."""

    def __init__(
        self,
        embedder: Embeddings,
        vectors: VectorStore,
        lexical: LexicalStore,
        reranker: Reranker,
        settings: Settings,
    ):
        """Store the pipeline components and retrieval tuning settings."""
        self._embedder = embedder
        self._vectors = vectors
        self._lexical = lexical
        self._reranker = reranker
        self._s = settings

    def _resolve(self, query: str, strategy: str | None) -> tuple[str, str | None]:
        """Resolve the effective strategy: explicit param > classifier (auto) > settings."""
        s = (strategy or self._s.retrieval_strategy or "hybrid_rerank").lower()
        if s not in STRATEGIES:
            s = "hybrid_rerank"
        if s == "auto":
            verdict = classify(query)
            return MODE_TO_STRATEGY[verdict], verdict.value
        return s, None

    def search(
        self,
        query: str,
        collection: str,
        acl: ACLFilter | None,
        focus_client: str | None = None,
        strategy: str | None = None,
    ) -> dict[str, Any]:
        """Execute the resolved retrieval strategy for one query + collection.

        Runs only the legs the strategy needs (dense-only skips BM25 etc.),
        fuses via weighted RRF when both legs ran, reranks only under
        ``hybrid_rerank``, expands parent passages for child chunks when
        ``parent_child_enabled``, and returns top-k results plus full
        per-stage diagnostics.
        """
        resolved, verdict = self._resolve(query, strategy)
        use_dense = resolved in ("dense", "hybrid", "hybrid_rerank")
        use_bm25 = resolved in ("bm25", "hybrid", "hybrid_rerank")

        dense: list[dict] = []
        bm25: list[dict] = []
        if use_dense:
            qvec = self._embedder.embed_query(query)
            dense = self._vectors.search(qvec, collection, acl, self._s.dense_top_k, focus_client)
        if use_bm25:
            bm25 = self._lexical.search(query, collection, acl, self._s.lexical_top_k, focus_client)

        if resolved == "dense":
            candidates = dense
            fused: list[dict] = []
        elif resolved == "bm25":
            candidates = bm25
            fused = []
        else:
            fused = rrf_fuse(
                [dense, bm25],
                k=self._s.rrf_k,
                weights=[self._s.rrf_dense_weight, self._s.rrf_bm25_weight],
            )
            candidates = fused[: self._s.rerank_candidates]

        if resolved == "hybrid_rerank":
            reranked = self._reranker.rerank(query, candidates)
        else:
            reranked = candidates

        results = reranked[: self._s.final_top_k]
        parent_expanded = self._expand_parents(results)
        for r in results:
            if is_expired(r):
                r["stale"] = True

        skipped = [leg for leg, used in (("dense", use_dense), ("bm25", use_bm25)) if not used]
        if resolved != "hybrid_rerank":
            skipped.append("rerank")
        return {
            "results": results,
            "diagnostics": {
                "strategy": resolved,
                "classifier_verdict": verdict,
                "skipped": skipped,
                "dense_ids": [r["doc_id"] for r in dense],
                "bm25_ids": [r["doc_id"] for r in bm25],
                "fused_ids": [r["doc_id"] for r in fused],
                "reranked_ids": [r["doc_id"] for r in reranked],
                "counts": {
                    "dense": len(dense),
                    "bm25": len(bm25),
                    "fused": len(fused),
                    "stale": sum(1 for r in results if r.get("stale")),
                    "parent_expanded": parent_expanded,
                },
            },
        }

    def _expand_parents(self, results: list[dict]) -> int:
        """Attach parent-section text to child hits when parent-child is enabled.

        Long agreements index paragraph children under section parents;
        a child hit carries the parent's full text in ``parent_text`` so the
        synthesizer sees enough context. Flagged in diagnostics for eval.
        """
        if not self._s.parent_child_enabled:
            return 0
        expanded = 0
        for r in results:
            parent_id = r.get("parent_id")
            if not parent_id:
                continue
            try:
                parent = self._vectors.fetch_chunk(parent_id)
            except Exception as exc:
                log.warning("hybrid.parent_fetch_failed", parent_id=parent_id, error=str(exc)[:200])
                continue
            if parent:
                r["parent_text"] = parent.get("text", "")
                r["parent_expanded"] = True
                expanded += 1
        return expanded
