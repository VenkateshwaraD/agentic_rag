"""API contract tests — versioning, error envelope, auth, idempotency.

Runs without external services (no Postgres/OpenSearch/LLM): auth and
validation failures are exercised directly; the answer path itself is
integration-tier.

Author: venkateshwara doijode
"""

import json

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

AUTH = {"X-API-Key": "dev-key-agent1", "X-User-Id": "u_agent1"}


def test_health_legacy():
    """Legacy liveness endpoint returns ``{"status": "ok"}``."""
    assert client.get("/health").json() == {"status": "ok"}


def test_health_v1():
    """Versioned liveness endpoint returns ``{"status": "ok"}``."""
    assert client.get("/api/v1/health").json() == {"status": "ok"}


def test_answer_requires_auth():
    """POST /answer without auth headers is rejected with 401."""
    resp = client.post("/answer", json={"question": "wire fee?"})
    assert resp.status_code == 401


def test_answer_rejects_bad_key():
    """POST /answer with an invalid API key is rejected with 401."""
    resp = client.post(
        "/answer",
        json={"question": "wire fee?"},
        headers={"X-API-Key": "nope", "X-User-Id": "u_agent1"},
    )
    assert resp.status_code == 401


def test_v1_error_envelope_shape():
    """401s under /api/v1 carry the stable {"error": {code,message,request_id}} body."""
    resp = client.post("/api/v1/answer", json={"question": "wire fee?"})
    assert resp.status_code == 401
    err = resp.json()["error"]
    assert err["code"] == "unauthorized"
    assert err["request_id"]


def test_v1_validation_error_envelope():
    """A malformed body (empty question) → 422 with the error envelope."""
    resp = client.post("/api/v1/answer", json={"question": ""}, headers=AUTH)
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "invalid_input"


def test_request_id_echoed():
    """X-Request-Id supplied by the client is echoed back in header + body."""
    resp = client.post(
        "/api/v1/answer",
        json={"question": "wire fee?"},
        headers={"X-Request-Id": "req-test-123"},
    )
    assert resp.headers["X-Request-Id"] == "req-test-123"
    assert resp.json()["error"]["request_id"] == "req-test-123"


def test_deps_endpoint_reports_status():
    """/api/v1/deps returns per-dependency results (503 offline is acceptable)."""
    resp = client.get("/api/v1/deps")
    assert resp.status_code in (200, 503)
    body = resp.json()
    assert body["status"] in ("healthy", "degraded")
    assert "postgres" in body["checks"]
    assert "opensearch" in body["checks"]


def test_idempotency_lookup_conflict():
    """Replaying an Idempotency-Key with a different body raises a conflict."""
    from app.api import idempotency
    from app.core.errors import IdempotencyConflictError

    idempotency.clear()
    idempotency.store("u1", "k1", "fee?", None, {"reply": "r"})
    assert idempotency.lookup("u1", "k1", "fee?", None) == {"reply": "r"}
    try:
        idempotency.lookup("u1", "k1", "different question", None)
        raise AssertionError("expected IdempotencyConflictError")
    except IdempotencyConflictError:
        pass
    idempotency.clear()


def test_admin_requires_token():
    """Admin endpoints reject requests without the X-Admin-Token header."""
    resp = client.get("/api/v1/admin/review")
    assert resp.status_code == 403
    assert resp.json()["error"]["code"] == "forbidden"


def test_api_key_cannot_impersonate_another_user():
    """A valid key presented with a different X-User-Id is rejected — the
    key↔user binding is enforced, not just key validity."""
    resp = client.post(
        "/api/v1/answer",
        json={"question": "wire fee?"},
        headers={"X-API-Key": "dev-key-agent1", "X-User-Id": "u_agent2"},
    )
    assert resp.status_code == 401


def test_request_id_unsafe_charset_replaced():
    """A caller-supplied X-Request-Id outside the safe charset is dropped —
    attacker-controlled strings must not land in logs/audit verbatim."""
    bad = "bad id<script>stuff"
    resp = client.post(
        "/api/v1/answer",
        json={"question": "x"},
        headers={"X-Request-Id": bad},
    )
    assert resp.headers["X-Request-Id"] != bad


def test_users_fixture_rejects_shared_keys(tmp_path):
    """load_users fails closed when one api_key maps to two users — a shared
    key lets its holder impersonate either record."""
    from app.core.errors import UnauthorizedError
    from app.security.authn import load_users

    fixture = tmp_path / "users.json"
    fixture.write_text(
        json.dumps(
            [
                {"user_id": "u1", "api_key": "shared", "role": "agent", "client_ids": ["acme"]},
                {"user_id": "u2", "api_key": "shared", "role": "admin", "client_ids": ["acme"]},
            ]
        )
    )
    with pytest.raises(UnauthorizedError):
        load_users(fixture)


def test_production_guard_rejects_dev_defaults():
    """ENVIRONMENT=production + published dev secrets → startup refuses to boot."""
    from app.core.config import Settings

    with pytest.raises(RuntimeError, match="unsafe production"):
        Settings(environment="production").assert_safe_for_production()
    Settings().assert_safe_for_production()  # development — no-op
