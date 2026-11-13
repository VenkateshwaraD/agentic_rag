"""Ingestion pipeline — the independently testable stage sequence:

    parse → validate → normalize → metadata-extract → chunk → redact
    → embed → index

Each stage is a small pure-ish function; ``ingest_document`` composes them
per file and returns a structured ``IngestResult`` instead of raising, so
one bad document never kills a job — failures carry the failed stage name
for dead-lettering and metrics.

Author: venkateshwara doijode
"""

from dataclasses import dataclass, field
from pathlib import Path

from app.core.config import get_settings
from app.core.errors import AppError
from app.data.ingest.chunking import Chunk, chunk_document
from app.data.ingest.loaders import (
    CorruptedDocumentError,
    RawDocument,
    parse_document,
)
from app.data.ingest.normalize import normalize_text
from app.data.ingest.redact import redact_chunk_text
from app.data.ingest.validate import validate_document, validate_path

# stages — recorded on IngestResult when that stage fails
STAGES = ("validate", "parse", "normalize", "chunk", "redact", "embed", "index")


@dataclass
class IngestResult:
    """Structured per-document outcome returned by ``ingest_document``."""

    path: str
    collection: str
    status: str  # indexed | skipped | failed
    doc_id: str = ""
    reason: str = ""  # unsupported | empty | corrupted | unchanged | <stage>_failed
    stage: str = ""  # stage that produced a failure
    chunk_ids: list[str] = field(default_factory=list)
    doc_version: str = ""
    content_hash: str = ""
    errors: list[str] = field(default_factory=list)


def _fail(
    path: Path, collection: str, stage: str, reason: str, exc: Exception | str
) -> IngestResult:
    """Build a failed IngestResult recording the stage and sanitized error."""
    return IngestResult(
        path=str(path),
        collection=collection,
        status="failed",
        reason=f"{stage}_failed",
        stage=stage,
        errors=[str(exc)[:300]],
    )


def ingest_document(
    path: Path,
    collection: str,
    embedder=None,
    vectors=None,
    lexical=None,
) -> IngestResult:
    """Run the full stage pipeline for one file; never raises.

    ``embedder``/``vectors``/``lexical`` are injectable so stages can be
    tested in isolation with fakes — when omitted they resolve from the
    cached app dependencies.
    """
    from app.core.dependencies import get_embedder, get_opensearch, get_vectorstore

    settings = get_settings()
    embedder = embedder or get_embedder()
    vectors = vectors or get_vectorstore()
    lexical = lexical or get_opensearch()

    # stage: validate (path-level pre-checks)
    v = validate_path(path)
    if v.status != "ok":
        return IngestResult(
            path=str(path),
            collection=collection,
            status="skipped",
            reason=v.status,
            stage="validate",
            errors=[v.detail],
        )

    # stage: parse
    try:
        raw: RawDocument = parse_document(path)
    except CorruptedDocumentError as exc:
        return _fail(path, collection, "parse", "corrupted", exc)
    except AppError as exc:  # race: extension ok'd but loader rejected
        return IngestResult(
            path=str(path),
            collection=collection,
            status="skipped",
            reason="unsupported",
            stage="parse",
            errors=[str(exc)[:200]],
        )

    # stage: validate (content)
    if (vc := validate_document(raw)).status != "ok":
        return IngestResult(
            path=str(path),
            collection=collection,
            status="skipped",
            reason=vc.status,
            stage="validate",
            errors=[vc.detail],
        )

    # stage: normalize
    text = normalize_text(raw.text)

    # stage: chunk (metadata extraction happens inside chunk_document)
    try:
        chunks: list[Chunk] = chunk_document(
            text, path, collection, parent_child=settings.parent_child_enabled
        )
    except Exception as exc:
        return _fail(path, collection, "chunk", "chunk", exc)
    if not chunks:
        return IngestResult(
            path=str(path),
            collection=collection,
            status="skipped",
            reason="empty",
            stage="chunk",
            errors=["document produced no chunks"],
        )

    # stage: redact (per-chunk — placeholders preserve lookup ids by policy)
    try:
        for c in chunks:
            c.text = redact_chunk_text(c.text)
    except Exception as exc:
        return _fail(path, collection, "redact", "redact", exc)

    # stage: embed
    try:
        embeddings = embedder.embed_documents([c.text for c in chunks])
    except Exception as exc:
        return _fail(path, collection, "embed", "embedding", exc)

    # stamp the *actual* embedding metadata (local ≠ configured dim)
    model = (
        settings.local_embed_model
        if settings.embedding_provider == "local"
        else settings.bedrock_embed_model
    )
    actual_dim = len(embeddings[0]) if embeddings else settings.embedding_dim
    for c in chunks:
        c.embed_model = model
        c.embed_dim = actual_dim

    # stage: index (both stores — pgvector first, then OpenSearch)
    try:
        vectors.upsert(chunks, embeddings)
    except Exception as exc:
        return _fail(path, collection, "index", "indexing_pg", exc)
    try:
        lexical.ensure_index(collection)
        lexical.bulk_index(collection, chunks)
    except Exception as exc:
        return _fail(path, collection, "index", "indexing_os", exc)

    return IngestResult(
        path=str(path),
        collection=collection,
        status="indexed",
        doc_id=chunks[0].doc_id,
        chunk_ids=[c.chunk_id for c in chunks],
        doc_version=chunks[0].doc_version,
        content_hash=chunks[0].content_hash,
    )
