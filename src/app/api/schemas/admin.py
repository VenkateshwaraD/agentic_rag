"""Pydantic schemas for the admin endpoints (reindex, audit lookup).

Author: venkateshwara doijode
"""

from pydantic import BaseModel


class ReindexResponse(BaseModel):
    """Response body for ``POST /admin/reindex`` — per-collection doc counts."""

    collections: dict[str, int]
