"""Admin endpoints — reindex, review queue, ingestion jobs, doc ops.

All routes require the ``X-Admin-Token`` header (router-level dependency).

Author: venkateshwara doijode
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from app.api.deps import get_admin
from app.api.schemas.admin import ReindexResponse
from app.core.errors import AppError, NotFoundError
from app.data.ingest import jobs
from app.observability import review
from app.retrieval.indexing import (
    delete_document,
    index_info,
    run_index,
    verify_consistency,
)

router = APIRouter(dependencies=[Depends(get_admin)])


class IndexRequest(BaseModel):
    """Body for ``POST /admin/index`` — incremental by default, rebuild optional."""

    rebuild: bool = False


class ReviewDecision(BaseModel):
    """Body for ``POST /admin/review/{id}/decision`` — reviewer resolution."""

    status: str = Field(pattern="^(approved|rejected|edited)$")
    note: str = Field(default="", max_length=2000)


@router.post("/admin/reindex", response_model=ReindexResponse)
def reindex():
    """Trigger a full rebuild of the vector and lexical indexes."""
    return ReindexResponse(collections=run_index(rebuild=True))


@router.post("/admin/index", response_model=ReindexResponse)
def index_run(body: IndexRequest):
    """Run the ingestion pipeline as a tracked job.

    Default ``index_mode`` applies (incremental unless configured otherwise);
    ``rebuild=true`` wipes both stores, the registry, and re-indexes.
    """
    return ReindexResponse(collections=run_index(rebuild=body.rebuild))


@router.get("/admin/ingest/jobs")
def ingest_jobs(limit: int = 20):
    """List recent ingestion jobs with status/progress/metrics."""
    return {"jobs": jobs.list_jobs(limit)}


@router.get("/admin/ingest/jobs/{job_id}")
def ingest_job(job_id: str):
    """Return one ingestion job's status/progress row, or 404."""
    job = jobs.get_job(job_id)
    if not job:
        raise NotFoundError(f"unknown ingestion job {job_id}")
    return job


@router.get("/admin/ingest/dlq")
def ingest_dlq(job_id: str | None = None, limit: int = 100):
    """List dead-lettered documents (optionally scoped to one job)."""
    return {"dead_letters": jobs.dead_letters(job_id=job_id, limit=limit)}


@router.post("/admin/ingest/dlq/{path:path}/retry")
def ingest_dlq_retry(path: str):
    """Re-run the pipeline for one dead-lettered file; clear the letter on success."""
    from pathlib import Path

    from app.core.config import get_settings
    from app.data.ingest.pipeline import ingest_document
    from app.retrieval.indexing import _collection_dirs

    p = Path(path).resolve()
    coll = next(
        (c for root, c in _collection_dirs(get_settings()) if p.is_relative_to(root.resolve())),
        None,
    )
    if coll is None:
        raise NotFoundError(f"{path} is not under a known collection directory")
    result = ingest_document(p, coll)
    if result.status == "failed":
        raise AppError(
            f"retry failed at stage {result.stage}: {'; '.join(result.errors)}",
            code="retry_failed",
        )
    jobs.clear_dead_letter(path)
    return {"status": result.status, "doc_id": result.doc_id, "chunks": len(result.chunk_ids)}


@router.delete("/admin/documents/{doc_id}")
def delete_doc(doc_id: str):
    """Delete a document from pgvector, OpenSearch, and the registry."""
    return {"doc_id": doc_id, **delete_document(doc_id)}


@router.get("/admin/consistency")
def consistency():
    """Report pgvector↔OpenSearch↔registry chunk_id parity per collection."""
    return {"collections": verify_consistency()}


@router.get("/admin/index-info")
def index_meta():
    """Return the embedding-model registry (name/version/dimensions) per index."""
    return {"indexes": index_info()}


@router.get("/admin/ops-health")
def ops_health():
    """Health-check each operational store (pooled SELECT 1 + latency)."""
    from app.core.dependencies import (
        get_accounts_store,
        get_tickets_store,
        get_trades_store,
    )

    stores = [get_accounts_store(), get_trades_store(), get_tickets_store()]
    checks = [s.healthcheck().model_dump() for s in stores]
    return {"status": "healthy" if all(c["ok"] for c in checks) else "degraded", "stores": checks}


# --- human review queue ----------------------------------------------------


@router.get("/admin/review")
def review_pending(limit: int = 50):
    """List pending human-review items newest-first."""
    try:
        review.ensure_table()
    except Exception as exc:
        raise AppError(f"review store unavailable: {exc}", code="store_unavailable") from exc
    return {"pending": review.list_pending(limit)}


@router.post("/admin/review/{review_id}/decision")
def review_decision(review_id: int, body: ReviewDecision):
    """Resolve a review item (approved/rejected/edited) with an optional note."""
    row = review.resolve(review_id, body.status, body.note)
    if not row:
        raise NotFoundError(f"unknown review id {review_id}")
    return row
