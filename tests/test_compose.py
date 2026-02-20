"""
tests/test_compose.py — compose_answer() unit tests.

All Ollama HTTP calls are mocked via urllib.request.urlopen.
All tests use a real in-memory/tmp SQLite DB.

Run with: pytest tests/test_compose.py -v
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from config import load_config
from core.db import get_connection
from core.compose import (
    ComposeResult,
    ResultView,
    _confidence,
    compose_answer,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.dump({"index_roots": [str(tmp_path)]}),
        encoding="utf-8",
    )
    return load_config(config_file=cfg_file, work_dir=tmp_path)


@pytest.fixture
def conn(tmp_path: Path):
    db = get_connection(tmp_path / "db" / "tiga.db")
    yield db
    db.close()


def _fake_result(score: float = 0.8) -> dict:
    return {
        "chunk_id":     "cid-001",
        "file_id":      "fid-001",
        "ref_value":    "p1",
        "rel_path":     "proj/brief.txt",
        "file_name":    "brief.txt",
        "project_id":   "2023_HOSP",
        "typology":     "healthcare",
        "snippet":      "hospital architecture brief design",
        "bm25_score":   score,
        "vector_score": score,
        "final_score":  score,
        "citation":     "proj/brief.txt#p1",
    }


def _make_urlopen_mock(content: str) -> MagicMock:
    """Return a mock for urllib.request.urlopen that yields JSON {message:{content}}."""
    body = json.dumps({"message": {"content": content}}).encode()
    mock_resp = MagicMock()
    mock_resp.read.return_value = body
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    return MagicMock(return_value=mock_resp)


# ---------------------------------------------------------------------------
# Unit tests — ResultView, _confidence
# ---------------------------------------------------------------------------

def test_result_view_from_search_result() -> None:
    """ResultView.from_search_result maps SearchResult keys correctly."""
    v = ResultView.from_search_result(_fake_result(0.75))
    assert v.title      == "brief"
    assert v.rel_path   == "proj/brief.txt"
    assert v.citation   == "proj/brief.txt#p1"
    assert v.ext        == ".txt"
    assert v.project_id == "2023_HOSP"
    assert v.typology   == "healthcare"
    assert abs(v.final_score - 0.75) < 1e-9


def test_confidence_zero_when_no_results() -> None:
    """Empty result list → confidence = 0.0."""
    assert _confidence([]) == 0.0


def test_confidence_with_single_result() -> None:
    """Single result → confidence equals its final_score (no normalisation needed)."""
    v = ResultView.from_search_result(_fake_result(0.7))
    assert abs(_confidence([v]) - 0.7) < 1e-9


def test_confidence_with_multiple_results() -> None:
    """Multiple results → confidence in [0, 1]."""
    scores = [0.9, 0.6, 0.3]
    views = [ResultView.from_search_result(_fake_result(s)) for s in scores]
    c = _confidence(views)
    assert 0.0 <= c <= 1.0


# ---------------------------------------------------------------------------
# Integration tests — compose_answer
# ---------------------------------------------------------------------------

def test_fallback_used_when_ollama_unavailable(tmp_path: Path, conn) -> None:
    """Ollama down → fallback prefix + raw snippets, no crash."""
    cfg_obj = _cfg(tmp_path)
    with patch("urllib.request.urlopen", side_effect=Exception("connection refused")):
        result = compose_answer("hospital brief", [_fake_result()],
                                conn=conn, cfg_obj=cfg_obj)
    assert isinstance(result, ComposeResult)
    assert result.answer_summary.startswith("[Ollama unavailable")
    assert "hospital architecture brief" in result.answer_summary


def test_follow_ups_empty_on_ollama_failure(tmp_path: Path, conn) -> None:
    """Ollama down → follow_ups is empty list, no crash."""
    cfg_obj = _cfg(tmp_path)
    with patch("urllib.request.urlopen", side_effect=Exception("timeout")):
        result = compose_answer("hospital brief", [_fake_result()],
                                conn=conn, cfg_obj=cfg_obj)
    assert result.follow_ups == []


def test_ollama_answer_returned_in_summary(tmp_path: Path, conn) -> None:
    """Ollama reachable → answer_summary equals Ollama's response content."""
    cfg_obj = _cfg(tmp_path)
    expected = "The hospital brief covers structural and design requirements."
    mock_urlopen = _make_urlopen_mock(expected)
    with patch("urllib.request.urlopen", mock_urlopen):
        result = compose_answer("hospital brief", [_fake_result()],
                                conn=conn, cfg_obj=cfg_obj)
    assert result.answer_summary == expected


def test_confidence_nonzero_in_compose_result(tmp_path: Path, conn) -> None:
    """Non-empty results → confidence > 0."""
    cfg_obj = _cfg(tmp_path)
    with patch("urllib.request.urlopen", side_effect=Exception("down")):
        result = compose_answer("hospital", [_fake_result(0.8)],
                                conn=conn, cfg_obj=cfg_obj)
    assert result.confidence > 0.0


def test_confidence_zero_when_no_results(tmp_path: Path, conn) -> None:
    """Empty results list → confidence = 0.0."""
    cfg_obj = _cfg(tmp_path)
    with patch("urllib.request.urlopen", side_effect=Exception("down")):
        result = compose_answer("hospital", [], conn=conn, cfg_obj=cfg_obj)
    assert result.confidence == 0.0
    assert result.results == []


def test_session_persistence_saves_user_message(tmp_path: Path, conn) -> None:
    """With session_id, user query is saved to messages table."""
    cfg_obj = _cfg(tmp_path)
    sid = "sess-compose-001"
    with patch("urllib.request.urlopen", side_effect=Exception("down")):
        compose_answer("hospital brief", [_fake_result()],
                       session_id=sid, conn=conn, cfg_obj=cfg_obj)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY message_id",
        (sid,),
    ).fetchall()
    assert any(r["role"] == "user" and "hospital" in r["content"] for r in rows), \
        f"User query not found in messages: {list(rows)}"


def test_session_persistence_saves_assistant_message(tmp_path: Path, conn) -> None:
    """With session_id, assistant answer is saved to messages table."""
    cfg_obj = _cfg(tmp_path)
    sid = "sess-compose-002"
    mock_urlopen = _make_urlopen_mock("Design brief summary answer.")
    with patch("urllib.request.urlopen", mock_urlopen):
        compose_answer("hospital brief", [_fake_result()],
                       session_id=sid, conn=conn, cfg_obj=cfg_obj)
    rows = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=? ORDER BY message_id",
        (sid,),
    ).fetchall()
    assert any(r["role"] == "assistant" for r in rows), \
        f"Assistant message not found: {list(rows)}"


def test_session_citations_saved_as_json(tmp_path: Path, conn) -> None:
    """Assistant message citations column contains JSON list of citation strings."""
    cfg_obj = _cfg(tmp_path)
    sid = "sess-compose-003"
    mock_urlopen = _make_urlopen_mock("Answer text.")
    with patch("urllib.request.urlopen", mock_urlopen):
        compose_answer("query", [_fake_result()],
                       session_id=sid, conn=conn, cfg_obj=cfg_obj)
    row = conn.execute(
        "SELECT citations FROM messages WHERE session_id=? AND role='assistant'",
        (sid,),
    ).fetchone()
    assert row is not None
    import json as _json
    citations = _json.loads(row["citations"])
    assert isinstance(citations, list)
    assert any("proj/brief.txt#p1" in c for c in citations)


def test_latency_logged_to_events(tmp_path: Path, conn) -> None:
    """compose_answer logs a compose_answer event to the events table."""
    cfg_obj = _cfg(tmp_path)
    mock_urlopen = _make_urlopen_mock("Answer.")
    with patch("urllib.request.urlopen", mock_urlopen):
        result = compose_answer("query", [_fake_result()], conn=conn, cfg_obj=cfg_obj)
    events = conn.execute(
        "SELECT event_type, detail FROM events WHERE event_type='compose_answer'"
    ).fetchall()
    assert len(events) >= 1
    assert "ms" in events[0]["detail"]
    assert result.latency_ms > 0


def test_results_list_populated(tmp_path: Path, conn) -> None:
    """compose_answer.results contains ResultView objects from the input."""
    cfg_obj = _cfg(tmp_path)
    with patch("urllib.request.urlopen", side_effect=Exception("down")):
        result = compose_answer("hospital", [_fake_result()],
                                conn=conn, cfg_obj=cfg_obj)
    assert len(result.results) == 1
    v = result.results[0]
    assert isinstance(v, ResultView)
    assert v.citation == "proj/brief.txt#p1"
    assert v.project_id == "2023_HOSP"


def test_no_session_id_no_messages_saved(tmp_path: Path, conn) -> None:
    """Without session_id, nothing is written to messages table."""
    cfg_obj = _cfg(tmp_path)
    with patch("urllib.request.urlopen", side_effect=Exception("down")):
        compose_answer("query", [_fake_result()], conn=conn, cfg_obj=cfg_obj)
    count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    assert count == 0
