"""Indexing pipeline: per-document staged ingest → embed → write both stores.

Orchestrates the testable stages in ``data/ingest/pipeline.py`` behind a
job tracker: incremental indexing via the document-registry manifest,
per-document retry + dead-lettering, version-policy re-index deletes,
manifest pruning, and cross-store consistency verification.

Author: venkateshwara doijode
"""

import hashlib
import re
import time
from collections import defaultdict
from pathlib import Path

from app.core.config import get_settings
from app.core.constants import COLLECTIONS
from app.core.dependencies import (
    get_embedder,
    get_opensearch,
    get_vectorstore,
)
from app.core.logging import get_logger
from app.data.ingest import jobs
from app.data.ingest.loaders import iter_documents
from app.data.ingest.pipeline import ingest_document
from app.retrieval.embeddings import probe_dimension

log = get_logger(__name__)

_DOC_ID_RE = re.compile(r"doc_id:\s*([A-Z0-9-]+)")


def _collection_dirs(settings) -> list[tuple[Path, str]]:
    """Map filesystem dirs to collection names (synthetic + real drop-in)."""
    pairs: list[tuple[Path, str]] = []
    for coll in COLLECTIONS:
        for root in (settings.synthetic_dir / coll, settings.real_dir / coll):
            if root.is_dir():
                pairs.append((root, coll))
    return pairs


def _sniff_doc(path: Path) -> tuple[str, str]:
    """Cheap (doc_id, version) probe from the file header for pre-delete calls.

    Reads the first few KB for the metadata comment; falls back to the file
    stem and ``1.0`` (the same fallbacks ``chunk_document`` uses).
    """
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")[:4096]
        doc_m = _DOC_ID_RE.search(head)
        ver_m = re.search(r"version:\s*([\d.]+)", head)
        return (doc_m.group(1) if doc_m else path.stem, ver_m.group(1) if ver_m else "1.0")
    except OSError:
        return path.stem, "1.0"


def _file_hash(path: Path) -> str:
    """sha256 of file bytes — the incremental manifest's change key."""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:32]


def _ingest_with_retry(path: Path, coll: str, settings, job_id: str):
    """Run ``ingest_document`` with bounded retries; returns the final result.

    Corrupted/parse failures are unfixable — they go straight to the dead
    letter without retrying. Embed/index failures retry with small backoff.
    """
    result = None
    for attempt in range(1, settings.ingest_max_attempts + 1):
        result = ingest_document(path, coll)
        if result.status != "failed" or result.stage in ("validate", "parse"):
            return result
        if attempt < settings.ingest_max_attempts:
            time.sleep(min(2 ** (attempt - 1), 4))
    return result


def run_index(rebuild: bool = False) -> dict[str, int]:
    """Run the tracked indexing pipeline; return per-collection chunk counts.

    - ensures schemas, optionally wipes both stores + registry on rebuild;
    - ``index_mode=incremental``: file-hash vs registry → unchanged files
      skip, changed docs propagate deletes before re-index;
    - ``doc_version_policy=latest``: a doc_id seen under another path (or
      changed content) deletes its old chunks before indexing;
    - per-document retry → dead-letter on exhaustion; vanished files are
      pruned from registry and both stores;
    - records embedding-model metadata per collection (``index_meta``).
    """
    settings = get_settings()
    embedder = get_embedder()
    vectors = get_vectorstore()
    lexical = get_opensearch()
    incremental = settings.index_mode == "incremental" and not rebuild
    started = time.monotonic()

    jobs.ensure_tables()
    job_id = jobs.start_job(
        {
            "rebuild": rebuild,
            "index_mode": settings.index_mode,
            "doc_version_policy": settings.doc_version_policy,
        }
    )
    vectors.ensure_schema(probe_dimension(embedder))
    if rebuild:
        vectors.clear()
        jobs.clear_registry()
        for coll in COLLECTIONS:
            lexical.clear(coll)

    candidates = [
        (p, coll) for root, coll in _collection_dirs(settings) for p in iter_documents(root)
    ]
    jobs.set_docs_total(job_id, len(candidates))
    seen_paths: set[str] = set()
    counts: dict[str, int] = defaultdict(int)
    dead = 0

    for path, coll in candidates:
        sp = str(path)
        seen_paths.add(sp)
        file_hash = _file_hash(path)
        reg = jobs.registry_by_path(sp) if incremental else None
        if reg and reg["content_hash"] == file_hash:
            jobs.record_progress(job_id, skipped=1)
            counts[coll] += len(reg.get("chunk_ids") or [])
            continue

        # changed content at this path always replaces the path's old doc;
        # a doc_id living at another path is deleted under 'latest', or
        # under 'keep_all' only when its version collides (chunk_ids are
        # version-scoped so distinct versions coexist cleanly)
        to_delete: set[str] = set()
        if reg and reg.get("doc_id"):
            to_delete.add(reg["doc_id"])
        doc_id_guess, version_guess = _sniff_doc(path)
        known = jobs.registry_by_doc(doc_id_guess)
        if (
            known
            and known["source_path"] != sp
            and (settings.doc_version_policy == "latest" or known["doc_version"] == version_guess)
        ):
            to_delete.add(known["doc_id"])
        for old_doc in to_delete:
            delete_document(old_doc)

        result = _ingest_with_retry(path, coll, settings, job_id)
        if result.status == "failed":
            jobs.dead_letter(
                job_id, sp, result.stage, "; ".join(result.errors), settings.ingest_max_attempts
            )
            dead += 1
            jobs.record_progress(
                job_id,
                dead=1,
                embed_fail=1 if result.stage == "embed" else 0,
                index_fail=1 if result.stage == "index" else 0,
            )
        elif result.status == "skipped":
            jobs.record_progress(job_id, skipped=1)
            jobs.clear_dead_letter(sp)
        else:
            counts[coll] += len(result.chunk_ids)
            jobs.record_document(
                sp,
                result.doc_id,
                coll,
                result.doc_version,
                file_hash,
                result.chunk_ids,
                job_id,
            )
            jobs.record_progress(job_id, done=1, chunks=len(result.chunk_ids))
            jobs.clear_dead_letter(sp)
            log.info("index.doc_done", doc_id=result.doc_id, chunks=len(result.chunk_ids))

    # incremental manifest prune — files that vanished since last run
    if incremental:
        for old_path in jobs.registry_paths() - seen_paths:
            reg = jobs.registry_by_path(old_path)
            if reg:
                delete_document(reg["doc_id"])
                jobs.delete_registry_path(old_path)
                log.info("index.pruned", path=old_path, doc_id=reg["doc_id"])

    # embedding-model registry (both surfaces)
    model = (
        settings.local_embed_model
        if settings.embedding_provider == "local"
        else settings.bedrock_embed_model
    )
    for coll in COLLECTIONS:
        vectors.record_index_meta(
            coll, model, "", settings.embedding_dim, settings.embedding_provider
        )
        lexical.record_index_meta(
            coll,
            {
                "embed_model": model,
                "embed_version": "",
                "embed_dim": settings.embedding_dim,
                "provider": settings.embedding_provider,
            },
        )

    jobs.finish_job(job_id, "partial" if dead else "succeeded", started)
    log.info("index.job_done", job_id=job_id, dead_letters=dead, counts=dict(counts))
    return dict(counts)


def delete_document(doc_id: str) -> dict[str, int]:
    """Delete a document's chunks from both stores + registry (propagation).

    Returns ``{pg_deleted, os_deleted}`` so callers/auditors can verify the
    removal actually propagated. Exposed via ``copilot docs delete`` and
    ``DELETE /admin/documents/{id}``.
    """
    vectors = get_vectorstore()
    lexical = get_opensearch()
    pg_deleted = vectors.delete(doc_id)
    os_deleted = lexical.delete(doc_id)
    jobs.delete_registry_doc(doc_id)
    log.info("index.document_deleted", doc_id=doc_id, pg=pg_deleted, os=os_deleted)
    return {"pg_deleted": pg_deleted, "os_deleted": os_deleted}


def verify_consistency() -> dict[str, dict]:
    """Compare chunk_id sets across pgvector ↔ OpenSearch ↔ registry.

    Per collection: ``{in_sync, missing_in_pg, missing_in_os, untracked,
    hash_mismatches, ok}`` — ``untracked`` = chunks present in a store but
    absent from the registry manifest. ``copilot docs verify`` /
    ``GET /admin/consistency`` surface this.
    """
    vectors = get_vectorstore()
    lexical = get_opensearch()
    report: dict[str, dict] = {}
    for coll in COLLECTIONS:
        pg_ids = vectors.list_chunk_ids(coll)
        os_ids = lexical.list_chunk_ids(coll)
        reg_ids = jobs.registry_chunk_ids(coll)
        store_ids = pg_ids | os_ids
        report[coll] = {
            "in_sync": len(pg_ids & os_ids & reg_ids),
            "missing_in_os": sorted(pg_ids - os_ids)[:50],
            "missing_in_pg": sorted(os_ids - pg_ids)[:50],
            "untracked": sorted(store_ids - reg_ids)[:50],
            "ok": pg_ids == os_ids == reg_ids,
        }
    return report


def index_info() -> list[dict]:
    """Return the pg-side ``index_meta`` registry rows (model/version/dim)."""
    return get_vectorstore().index_meta()
