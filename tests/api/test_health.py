"""API health endpoint tests."""

from app.main import app
from fastapi.testclient import TestClient


def test_health_returns_ok() -> None:
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data
    assert "version" in data
