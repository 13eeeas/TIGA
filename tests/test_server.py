"""
tests/test_server.py — FastAPI server endpoint tests.

Uses FastAPI TestClient with dependency_overrides for the DB connection.
Ollama + search + compose_answer are mocked — no external services needed.

Run with: pytest tests/test_server.py -v
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from fastapi.testclient import TestClient

from config import load_config
from core.compose import ComposeResult, ResultView
from core.db import get_connection


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def db(tmp_path: Path):
    conn = get_connection(tmp_path / "db" / "tiga.db")
    yield conn
    conn.close()


@pytest.fixture
def client(db):
    """TestClient with DB dependency overridden to use the tmp SQLite DB."""
    from server import app, get_db
    app.dependency_overrides[get_db] = lambda: db
    # Suppress lifespan Ollama check noise during tests
    with patch("server.ollama_available", return_value=False):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c
    app.dependency_overrides.clear()


def _fake_compose_result(answer: str = "Hospital brief covers design requirements.") -> ComposeResult:
    v = ResultView(
        title       = "brief",
        rel_path    = "proj/brief.txt",
        citation    = "proj/brief.txt#p1",
        snippet     = "hospital architecture brief design",
        project_id  = "2023_HOSP",
        typology    = "healthcare",
        ext         = ".txt",
        final_score = 0.8,
    )
    return ComposeResult(
        answer_summary = answer,
        follow_ups     = ["What is the structural system?", "Who is the client?", "What is the budget?"],
        confidence     = 0.8,
        results        = [v],
        latency_ms     = 220.0,
    )


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

def test_health_endpoint_returns_ok(client) -> None:
    """Ollama reachable → {status: 'ok', ollama: true}."""
    with patch("server.ollama_available", return_value=True):
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["ollama"] is True


def test_health_endpoint_ollama_down(client) -> None:
    """Ollama unreachable → {status: 'degraded', ollama: false}."""
    with patch("server.ollama_available", return_value=False):
        resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "degraded"
    assert data["ollama"] is False


# ---------------------------------------------------------------------------
# POST /api/query
# ---------------------------------------------------------------------------

def test_query_endpoint_returns_answer_payload(client) -> None:
    """Query endpoint returns answer_summary, follow_ups, confidence, results, session_id."""
    with (
        patch("server.search", return_value=[]),
        patch("server.compose_answer", return_value=_fake_compose_result()),
    ):
        resp = client.post("/api/query", json={"query": "hospital brief"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["answer_summary"] == "Hospital brief covers design requirements."
    assert len(data["follow_up_prompts"]) == 3
    assert "confidence" in data
    assert "session_id" in data
    assert "latency_ms" in data
    assert len(data["results"]) == 1
    r = data["results"][0]
    assert r["citation"] == "proj/brief.txt#p1"
    assert r["project_id"] == "2023_HOSP"


def test_query_creates_session_if_none_provided(client) -> None:
    """No session_id in request → server generates a UUID and echoes it back."""
    with (
        patch("server.search", return_value=[]),
        patch("server.compose_answer", return_value=_fake_compose_result()),
    ):
        resp = client.post("/api/query", json={"query": "hospital"})

    assert resp.status_code == 200
    session_id = resp.json()["session_id"]
    assert session_id
    uuid.UUID(session_id)   # raises ValueError if not a valid UUID


def test_query_preserves_provided_session_id(client) -> None:
    """Explicit session_id is echoed back unchanged."""
    sid = "my-known-session-abc"
    with (
        patch("server.search", return_value=[]),
        patch("server.compose_answer", return_value=_fake_compose_result()),
    ):
        resp = client.post("/api/query", json={"query": "school", "session_id": sid})

    assert resp.status_code == 200
    assert resp.json()["session_id"] == sid


def test_query_empty_string_returns_400(client) -> None:
    """Blank query string → HTTP 400."""
    resp = client.post("/api/query", json={"query": "   "})
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------

def test_status_endpoint_returns_all_keys(client) -> None:
    """Status endpoint includes all required index-stat fields."""
    with patch("server.ollama_available", return_value=True):
        resp = client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    for key in (
        "files_discovered", "files_extracted", "files_embedded",
        "files_indexed", "files_skipped",
        "chunks_total", "embedded_chunks",
        "ollama_available", "last_indexed_at",
    ):
        assert key in data, f"Missing key in /api/status: {key}"


def test_status_last_indexed_at_null_on_empty_db(client) -> None:
    """No indexed files → last_indexed_at is null."""
    with patch("server.ollama_available", return_value=False):
        resp = client.get("/api/status")
    assert resp.status_code == 200
    assert resp.json()["last_indexed_at"] is None


# ---------------------------------------------------------------------------
# GET /api/projects
# ---------------------------------------------------------------------------

def test_projects_endpoint_returns_list(client, db) -> None:
    """Seeded project_id appears in /api/projects with correct file_count."""
    db.execute(
        "INSERT OR IGNORE INTO files "
        "(file_id, file_path, file_name, extension, status, project_id) "
        "VALUES ('fid1', '/a/file.txt', 'file.txt', '.txt', 'INDEXED', '2023_HOSP')"
    )
    db.commit()

    resp = client.get("/api/projects")
    assert resp.status_code == 200
    projects = resp.json()
    assert isinstance(projects, list)
    assert any(p["project_id"] == "2023_HOSP" and p["file_count"] == 1 for p in projects)


def test_projects_empty_on_empty_db(client) -> None:
    """Empty DB → empty projects list (or Unknown bucket only)."""
    resp = client.get("/api/projects")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


# ---------------------------------------------------------------------------
# POST /api/session + GET /api/session/{id}
# ---------------------------------------------------------------------------

def test_create_session_returns_valid_uuid(client) -> None:
    """POST /api/session returns a {session_id} that is a valid UUID."""
    resp = client.post("/api/session")
    assert resp.status_code == 200
    sid = resp.json()["session_id"]
    uuid.UUID(sid)   # raises if invalid


def test_get_session_returns_message_history(client, db) -> None:
    """Seeded messages appear in GET /api/session/{id}."""
    sid = "sess-server-001"
    db.execute("INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (sid,))
    db.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, 'user', 'hello archive')",
        (sid,),
    )
    db.commit()

    resp = client.get(f"/api/session/{sid}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == sid
    assert len(data["messages"]) == 1
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "hello archive"


def test_get_session_empty_for_unknown_id(client) -> None:
    """Unknown session_id returns empty message list (not 404)."""
    resp = client.get("/api/session/no-such-session")
    assert resp.status_code == 200
    assert resp.json()["messages"] == []
