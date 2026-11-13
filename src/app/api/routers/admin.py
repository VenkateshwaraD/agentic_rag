"""POST /admin/reindex — rebuild both stores (admin token required).

Author: venkateshwara doijode
"""

from fastapi import APIRouter, Depends

from app.api.deps import get_admin
from app.api.schemas.admin import ReindexResponse
from app.retrieval.indexing import run_index

router = APIRouter(dependencies=[Depends(get_admin)])


@router.post("/admin/reindex", response_model=ReindexResponse)
def reindex():
    """Trigger a full rebuild of the vector and lexical indexes; returns per-collection doc counts."""
    return ReindexResponse(collections=run_index(rebuild=True))
