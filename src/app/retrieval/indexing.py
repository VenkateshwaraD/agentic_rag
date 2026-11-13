"""Indexing pipeline: load → chunk → redact → embed → write to both stores.

Author: venkateshwara doijode
"""

from pathlib import Path

from app.core.config import get_settings
from app.core.constants import COLLECTIONS
from app.core.dependencies import (
    get_embedder,
    get_opensearch,
    get_vectorstore,
)
from app.core.logging import get_logger
from app.data.ingest.chunking import Chunk, chunk_directory
from app.data.ingest.loaders import iter_documents
from app.data.ingest.redact import redact_chunk_text
from app.retrieval.embeddings import probe_dimension

log = get_logger(__name__)


def _collection_dirs(settings) -> list[tuple[Path, str]]:
    """Map filesystem dirs to collection names (synthetic + real drop-in)."""
    pairs: list[tuple[Path, str]] = []
    for coll in COLLECTIONS:
        for root in (settings.synthetic_dir / coll, settings.real_dir / coll):
            if root.is_dir():
                pairs.append((root, coll))
    return pairs


def gather_chunks() -> dict[str, list[Chunk]]:
    """Load, chunk, and PII-redact every corpus doc, grouped by collection."""
    settings = get_settings()
    by_collection: dict[str, list[Chunk]] = {}
    for root, coll in _collection_dirs(settings):
        chunks = chunk_directory(root, coll, iter_documents)
        for c in chunks:
            c.text = redact_chunk_text(c.text)  # ingest-time PII redaction
        by_collection.setdefault(coll, []).extend(chunks)
        log.info("index.chunks_built", dir=str(root), collection=coll, n=len(chunks))
    return by_collection


def run_index(rebuild: bool = False) -> dict[str, int]:
    """Run the full indexing pipeline and return per-collection chunk counts.

    Ensures the pgvector schema (probing the embed dimension), optionally
    clears both stores when ``rebuild`` is set, then embeds and writes all
    gathered chunks to pgvector and OpenSearch.
    """
    embedder = get_embedder()
    vectors = get_vectorstore()
    lexical = get_opensearch()

    vectors.ensure_schema(probe_dimension(embedder))
    if rebuild:
        vectors.clear()
        for coll in COLLECTIONS:
            lexical.clear(coll)

    counts: dict[str, int] = {}
    for coll, chunks in gather_chunks().items():
        lexical.ensure_index(coll)
        embeddings = embedder.embed_documents([c.text for c in chunks])
        counts[coll] = vectors.upsert(chunks, embeddings)
        lexical.bulk_index(coll, chunks)
        log.info("index.collection_done", collection=coll, chunks=len(chunks))
    return counts
