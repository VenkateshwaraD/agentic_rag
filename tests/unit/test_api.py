"""API smoke tests — no external services required."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_answer_requires_auth():
    resp = client.post("/answer", json={"question": "wire fee?"})
    assert resp.status_code == 401


def test_answer_rejects_bad_key():
    resp = client.post(
        "/answer",
        json={"question": "wire fee?"},
        headers={"X-API-Key": "nope", "X-User-Id": "u_agent1"},
    )
    assert resp.status_code == 401
