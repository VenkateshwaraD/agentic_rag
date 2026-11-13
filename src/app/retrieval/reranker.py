"""Local cross-encoder reranker (BGE-reranker via sentence-transformers).
Lazy singleton — model loads on first use."""

from app.core.logging import get_logger

log = get_logger(__name__)


class Reranker:
    def __init__(self, model_name: str):
        self._model_name = model_name
        self._model = None  # lazy: sentence_transformers loads on first use

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            log.info("reranker.loading", model=self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(self, query: str, candidates: list[dict]) -> list[dict]:
        """Score (query, text) pairs; returns candidates sorted desc by score."""
        if not candidates:
            return []
        pairs = [(query, c["text"]) for c in candidates]
        scores = self.model.predict(pairs)
        for c, s in zip(candidates, scores, strict=True):
            c["rerank_score"] = float(s)
        return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
