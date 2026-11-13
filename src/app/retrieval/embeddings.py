"""Embedding providers — Titan via Bedrock (prod) or local sentence-transformers
(EMBEDDING_PROVIDER=local for offline dev/tests)."""

from langchain_core.embeddings import Embeddings

from app.core.config import Settings
from app.core.logging import get_logger

log = get_logger(__name__)


class LocalEmbedder(Embeddings):
    """sentence-transformers backed embedder conforming to the LC interface."""

    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def build_embedder(settings: Settings) -> Embeddings:
    if settings.embedding_provider == "local":
        log.info("embeddings.provider", provider="local", model=settings.local_embed_model)
        return LocalEmbedder(settings.local_embed_model)

    from langchain_aws import BedrockEmbeddings

    kwargs: dict = {"model_id": settings.bedrock_embed_model, "region_name": settings.aws_region}
    if settings.aws_profile:
        import boto3

        kwargs["client"] = boto3.Session(
            profile_name=settings.aws_profile, region_name=settings.aws_region
        ).client("bedrock-runtime")
    log.info("embeddings.provider", provider="bedrock", model=settings.bedrock_embed_model)
    return BedrockEmbeddings(**kwargs)


def probe_dimension(embedder: Embeddings) -> int:
    return len(embedder.embed_query("dimension probe"))
