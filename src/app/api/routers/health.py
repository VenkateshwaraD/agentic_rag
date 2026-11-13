"""Health endpoints — liveness, readiness, and dependency health.

Mounted at the root (k8s probes) and under /api/v1 for the public surface.
``/ready`` and ``/deps`` return HTTP 503 when a dependency is down so
probes and load balancers route around the replica.

Author: venkateshwara doijode
"""

import time

import psycopg
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from opensearchpy import OpenSearch

from app.core.config import get_settings

router = APIRouter()


def _probe_postgres(url: str) -> dict:
    """Return {ok, latency_ms} for a Postgres connect round-trip."""
    t = time.monotonic()
    try:
        psycopg.connect(url, connect_timeout=3).close()
        return {"ok": True, "latency_ms": round((time.monotonic() - t) * 1000, 1)}
    except Exception:
        return {"ok": False, "latency_ms": round((time.monotonic() - t) * 1000, 1)}


def _probe_opensearch(url: str) -> dict:
    """Return {ok, latency_ms} for an OpenSearch ping."""
    t = time.monotonic()
    try:
        ok = bool(OpenSearch([url], timeout=3).ping())
        return {"ok": ok, "latency_ms": round((time.monotonic() - t) * 1000, 1)}
    except Exception:
        return {"ok": False, "latency_ms": round((time.monotonic() - t) * 1000, 1)}


def _dependency_checks() -> dict[str, dict]:
    """Run every dependency probe and return the per-dependency results."""
    s = get_settings()
    checks = {
        "postgres": _probe_postgres(s.database_url),
        "opensearch": _probe_opensearch(s.opensearch_url),
        "ops_postgres": _probe_postgres(s.ops_database_url),
    }
    for name, store_getter in (
        ("ops_accounts", "get_accounts_store"),
        ("ops_trades", "get_trades_store"),
        ("ops_tickets", "get_tickets_store"),
    ):
        t = time.monotonic()
        try:
            from app.core import dependencies

            health = getattr(dependencies, store_getter)().healthcheck()
            checks[name] = {
                "ok": health.ok,
                "latency_ms": health.latency_ms or round((time.monotonic() - t) * 1000, 1),
            }
        except Exception:
            checks[name] = {"ok": False}
    return checks


@router.get("/health")
def health():
    """Liveness probe — always returns ``{"status": "ok"}`` when the app is up."""
    return {"status": "ok"}


@router.get("/ready")
def ready():
    """Readiness probe — 200 when every dependency is up, else 503 + details."""
    checks = _dependency_checks()
    ok = all(c["ok"] for c in checks.values())
    body = {"status": "ready" if ok else "degraded", "checks": checks}
    return body if ok else JSONResponse(status_code=503, content=body)


@router.get("/deps")
def deps():
    """Dependency health — per-dependency status + latency, 503 on failure."""
    checks = _dependency_checks()
    ok = all(c["ok"] for c in checks.values())
    body = {"status": "healthy" if ok else "degraded", "checks": checks}
    return body if ok else JSONResponse(status_code=503, content=body)
