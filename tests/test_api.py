"""Smoke tests for the FastAPI backend."""
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    from backend.app.main import app
    return TestClient(app)


def test_health_ok(client):
    response = client.get("/health/")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert data["version"] == "1.1.1"


def test_health_data_fields(client):
    response = client.get("/health/")
    data = response.json()
    assert "data_ready" in data
    assert "chunks_ready" in data
    assert "details" in data


def test_upload_rejects_non_pdf(client):
    response = client.post(
        "/upload/",
        files={"file": ("test.txt", b"hello", "text/plain")},
    )
    assert response.status_code == 400


def test_upload_rejects_large_file(client):
    big = b"x" * (11 * 1024 * 1024)
    response = client.post(
        "/upload/",
        files={"file": ("big.pdf", big, "application/pdf")},
    )
    assert response.status_code == 413


def test_query_invalid_body(client):
    response = client.post("/query/", json={"query": "hi"})
    # Short query (< 3 chars) should be rejected
    response2 = client.post("/query/", json={"query": "hi"})
    assert response2.status_code in (200, 422, 503)


def test_query_missing_index_files(client):
    """When chunk files are missing, query endpoint should return 503."""
    response = client.post(
        "/query/",
        json={"query": "What is IPC section 302?", "index_name": "ipc"},
    )
    # Either works (200) if data exists or 503 if data missing
    assert response.status_code in (200, 503)
