"""POST /admin/reindex — rebuild both stores (admin token required)."""

from fastapi import APIRouter, Depends

from app.api.deps import get_admin
from app.api.schemas.admin import ReindexResponse
from app.retrieval.indexing import run_index

router = APIRouter(dependencies=[Depends(get_admin)])


@router.post("/admin/reindex", response_model=ReindexResponse)
def reindex():
    return ReindexResponse(collections=run_index(rebuild=True))
