"""GET /health (liveness) and /ready (dependency checks)."""

import psycopg
from fastapi import APIRouter
from opensearchpy import OpenSearch

from app.core.config import get_settings

router = APIRouter()


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    s = get_settings()
    checks: dict[str, bool] = {}
    try:
        psycopg.connect(s.database_url, connect_timeout=3).close()
        checks["postgres"] = True
    except Exception:
        checks["postgres"] = False
    try:
        checks["opensearch"] = bool(OpenSearch([s.opensearch_url], timeout=3).ping())
    except Exception:
        checks["opensearch"] = False
    ok = all(checks.values())
    return {"status": "ready" if ok else "degraded", "checks": checks}
