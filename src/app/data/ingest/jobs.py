"""Ingestion job tracking — status/progress, doc↔chunk registry, dead letters.

Three ragdb tables (self-DDL, mirroring ``audit_events``):
- ``ingestion_jobs``: one row per index run — status, progress counters,
  and the required metrics (duration, doc count, chunk count, embedding
  failures, indexing failures);
- ``document_registry``: source_path ↔ doc_id ↔ chunk_ids mapping —
  doubles as the incremental-indexing manifest and the consistency-check
  baseline;
- ``ingestion_dead_letter``: per-document failures after retry exhaustion,
  recording the stage that failed for targeted recovery.

Author: venkateshwara doijode
"""

import json
import time
import uuid
from typing import Any

from sqlalchemy import text

from app.core.logging import get_logger
from app.db.session import rag_engine

log = get_logger(__name__)

_DDL = """
CREATE TABLE IF NOT EXISTS ingestion_jobs (
    job_id            TEXT PRIMARY KEY,
    status            TEXT NOT NULL DEFAULT 'running',
    params            JSONB NOT NULL DEFAULT '{}',
    started_at        TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at       TIMESTAMPTZ,
    duration_ms       DOUBLE PRECISION DEFAULT 0,
    docs_total        INT DEFAULT 0,
    docs_done         INT DEFAULT 0,
    docs_skipped      INT DEFAULT 0,
    chunks_indexed    INT DEFAULT 0,
    embedding_failures INT DEFAULT 0,
    indexing_failures INT DEFAULT 0,
    dead_letters      INT DEFAULT 0
);
CREATE TABLE IF NOT EXISTS document_registry (
    source_path   TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    collection    TEXT NOT NULL,
    doc_version   TEXT DEFAULT '1.0',
    content_hash  TEXT DEFAULT '',
    chunk_ids     JSONB NOT NULL DEFAULT '[]',
    status        TEXT DEFAULT 'indexed',
    job_id        TEXT DEFAULT '',
    indexed_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_registry_doc ON document_registry(doc_id);
CREATE TABLE IF NOT EXISTS ingestion_dead_letter (
    id              SERIAL PRIMARY KEY,
    job_id          TEXT NOT NULL,
    path            TEXT NOT NULL,
    stage           TEXT NOT NULL,
    error           TEXT DEFAULT '',
    attempts        INT DEFAULT 0,
    last_attempt_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_dlq_path ON ingestion_dead_letter(path);
"""


def ensure_tables() -> None:
    """Create the three ingestion tables if they don't exist."""
    with rag_engine().begin() as conn:
        for stmt in _DDL.split(";"):
            if stmt.strip():
                conn.execute(text(stmt))


def start_job(params: dict) -> str:
    """Insert a running job row and return its generated job_id."""
    job_id = uuid.uuid4().hex[:16]
    ensure_tables()
    with rag_engine().begin() as conn:
        conn.execute(
            text("INSERT INTO ingestion_jobs (job_id, params) VALUES (:j, CAST(:p AS jsonb))"),
            {"j": job_id, "p": json.dumps(params)},
        )
    return job_id


def _update(job_id: str, sets: str, params: dict) -> None:
    """Run an UPDATE on ingestion_jobs with the given SET fragment."""
    params["j"] = job_id
    with rag_engine().begin() as conn:
        conn.execute(text(f"UPDATE ingestion_jobs SET {sets} WHERE job_id = :j"), params)


def set_docs_total(job_id: str, total: int) -> None:
    """Record the total document count discovered for this job."""
    _update(job_id, "docs_total = :n", {"n": total})


def record_progress(
    job_id: str,
    done: int = 0,
    skipped: int = 0,
    chunks: int = 0,
    embed_fail: int = 0,
    index_fail: int = 0,
    dead: int = 0,
) -> None:
    """Increment the job's progress/metric counters by the given deltas."""
    _update(
        job_id,
        "docs_done = docs_done + :d, docs_skipped = docs_skipped + :s, "
        "chunks_indexed = chunks_indexed + :c, embedding_failures = embedding_failures + :e, "
        "indexing_failures = indexing_failures + :i, dead_letters = dead_letters + :dl",
        {"d": done, "s": skipped, "c": chunks, "e": embed_fail, "i": index_fail, "dl": dead},
    )


def finish_job(job_id: str, status: str, started_monotonic: float) -> None:
    """Mark the job finished (succeeded|partial|failed) with wall-clock duration."""
    duration = (time.monotonic() - started_monotonic) * 1000
    _update(
        job_id,
        "status = :s, finished_at = now(), duration_ms = :d",
        {"s": status, "d": duration},
    )


def get_job(job_id: str) -> dict[str, Any] | None:
    """Return one job row as a dict, or None when unknown."""
    with rag_engine().begin() as conn:
        row = (
            conn.execute(text("SELECT * FROM ingestion_jobs WHERE job_id = :j"), {"j": job_id})
            .mappings()
            .first()
        )
    return dict(row) if row else None


def list_jobs(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent jobs newest-first."""
    with rag_engine().begin() as conn:
        rows = conn.execute(
            text("SELECT * FROM ingestion_jobs ORDER BY started_at DESC LIMIT :l"),
            {"l": limit},
        ).mappings()
    return [dict(r) for r in rows]


# --- document registry -----------------------------------------------------


def registry_by_path(source_path: str) -> dict[str, Any] | None:
    """Return the registry row for a source path (incremental manifest lookup)."""
    with rag_engine().begin() as conn:
        row = (
            conn.execute(
                text("SELECT * FROM document_registry WHERE source_path = :p"),
                {"p": source_path},
            )
            .mappings()
            .first()
        )
    return dict(row) if row else None


def registry_by_doc(doc_id: str) -> dict[str, Any] | None:
    """Return the registry row for a document id (mapping/consistency lookup)."""
    with rag_engine().begin() as conn:
        row = (
            conn.execute(text("SELECT * FROM document_registry WHERE doc_id = :d"), {"d": doc_id})
            .mappings()
            .first()
        )
    return dict(row) if row else None


def clear_registry() -> None:
    """Drop every registry row (full rebuild — stores are wiped anyway)."""
    with rag_engine().begin() as conn:
        conn.execute(text("DELETE FROM document_registry"))


def registry_paths() -> set[str]:
    """Return every registered source path (manifest prune baseline)."""
    with rag_engine().begin() as conn:
        return {r[0] for r in conn.execute(text("SELECT source_path FROM document_registry"))}


def record_document(
    source_path: str,
    doc_id: str,
    collection: str,
    doc_version: str,
    content_hash: str,
    chunk_ids: list[str],
    job_id: str,
) -> None:
    """Upsert the doc↔chunk mapping + manifest row for an indexed document."""
    with rag_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO document_registry (source_path, doc_id, collection, doc_version, "
                "content_hash, chunk_ids, job_id) "
                "VALUES (:p, :d, :c, :v, :h, CAST(:ids AS jsonb), :j) "
                "ON CONFLICT (source_path) DO UPDATE SET doc_id = EXCLUDED.doc_id, "
                "collection = EXCLUDED.collection, doc_version = EXCLUDED.doc_version, "
                "content_hash = EXCLUDED.content_hash, chunk_ids = EXCLUDED.chunk_ids, "
                "job_id = EXCLUDED.job_id, indexed_at = now(), status = 'indexed'"
            ),
            {
                "p": source_path,
                "d": doc_id,
                "c": collection,
                "v": doc_version,
                "h": content_hash,
                "ids": json.dumps(chunk_ids),
                "j": job_id,
            },
        )


def delete_registry_path(source_path: str) -> None:
    """Remove a registry row (file vanished / document deleted)."""
    with rag_engine().begin() as conn:
        conn.execute(
            text("DELETE FROM document_registry WHERE source_path = :p"), {"p": source_path}
        )


def delete_registry_doc(doc_id: str) -> None:
    """Remove registry rows for a document id (deletion propagation)."""
    with rag_engine().begin() as conn:
        conn.execute(text("DELETE FROM document_registry WHERE doc_id = :d"), {"d": doc_id})


def registry_chunk_ids(collection: str) -> set[str]:
    """Return all registered chunk_ids for a collection (consistency baseline)."""
    with rag_engine().begin() as conn:
        rows = conn.execute(
            text("SELECT chunk_ids FROM document_registry WHERE collection = :c"),
            {"c": collection},
        )
        ids: set[str] = set()
        for (chunk_ids,) in rows:
            ids.update(chunk_ids or [])
        return ids


# --- dead letters -----------------------------------------------------------


def dead_letter(job_id: str, path: str, stage: str, error: str, attempts: int) -> None:
    """Record a document that exhausted retries into the dead-letter table."""
    with rag_engine().begin() as conn:
        conn.execute(
            text(
                "INSERT INTO ingestion_dead_letter (job_id, path, stage, error, attempts) "
                "VALUES (:j, :p, :s, :e, :a)"
            ),
            {"j": job_id, "p": path, "s": stage, "e": error[:500], "a": attempts},
        )


def dead_letters(job_id: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
    """Return dead-letter rows, optionally scoped to one job."""
    sql = "SELECT * FROM ingestion_dead_letter"
    params: dict[str, Any] = {"l": limit}
    if job_id:
        sql += " WHERE job_id = :j"
        params["j"] = job_id
    sql += " ORDER BY last_attempt_at DESC LIMIT :l"
    with rag_engine().begin() as conn:
        return [dict(r) for r in conn.execute(text(sql), params).mappings()]


def clear_dead_letter(path: str) -> None:
    """Drop dead-letter rows for a path that has since succeeded."""
    with rag_engine().begin() as conn:
        conn.execute(text("DELETE FROM ingestion_dead_letter WHERE path = :p"), {"p": path})
