"""Hybrid retrieval: dense + BM25 → RRF fusion → cross-encoder rerank.

Returns top-m results plus per-stage diagnostics (used by the eval harness
to measure dense-only / BM25-only / fused / reranked recall separately).

Author: venkateshwara doijode
"""

from collections import defaultdict
from typing import Any

from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.core.logging import get_logger
from app.retrieval.lexical import LexicalStore
from app.retrieval.reranker import Reranker
from app.retrieval.vectorstore import VectorStore
from app.security.authz import ACLFilter

log = get_logger(__name__)


def rrf_fuse(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    """Reciprocal rank fusion over multiple ranked result lists.
    Items are keyed by (doc_id, section)."""
    scores: dict[tuple, float] = defaultdict(float)
    items: dict[tuple, dict] = {}
    for ranked in rankings:
        for rank, item in enumerate(ranked, start=1):
            key = (item["doc_id"], item["section"])
            scores[key] += 1.0 / (k + rank)
            items.setdefault(key, item)
    fused = [{**items[key], "rrf_score": score} for key, score in scores.items()]
    return sorted(fused, key=lambda c: c["rrf_score"], reverse=True)


class HybridRetriever:
    """Runs dense + lexical search, fuses via RRF, and reranks the candidates."""

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

    def search(
        self,
        query: str,
        collection: str,
        acl: ACLFilter,
        focus_client: str | None = None,
    ) -> dict[str, Any]:
        """Execute the full hybrid pipeline for one query in one collection.

        Embeds the query, runs ACL-filtered dense and BM25 searches, fuses
        them with RRF, reranks the top candidates, and returns the final
        top-k results plus per-stage diagnostics.
        """
        qvec = self._embedder.embed_query(query)
        dense = self._vectors.search(qvec, collection, acl, self._s.dense_top_k, focus_client)
        bm25 = self._lexical.search(query, collection, acl, self._s.lexical_top_k, focus_client)
        fused = rrf_fuse([dense, bm25], k=self._s.rrf_k)
        candidates = fused[: self._s.rerank_candidates]
        reranked = self._reranker.rerank(query, candidates)

        return {
            "results": reranked[: self._s.final_top_k],
            "diagnostics": {
                "dense_ids": [r["doc_id"] for r in dense],
                "bm25_ids": [r["doc_id"] for r in bm25],
                "fused_ids": [r["doc_id"] for r in fused],
                "reranked_ids": [r["doc_id"] for r in reranked],
                "counts": {
                    "dense": len(dense),
                    "bm25": len(bm25),
                    "fused": len(fused),
                },
            },
        }
