"""
tests/test_db.py — SQLite schema and helper unit tests.

Run with: pytest tests/test_db.py -v
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

import pytest

from core.db import (
    add_message,
    compute_fingerprint,
    create_session,
    file_id_from_path,
    fts_search,
    get_connection,
    get_stats,
    log_event,
    set_file_status,
    upsert_chunk,
    upsert_file,
    upsert_project,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def conn(tmp_path: Path):
    db_path = tmp_path / "db" / "tiga.db"
    c = get_connection(db_path)
    yield c
    c.close()


def _file(file_path: str = "/archive/test.pdf", **kwargs) -> dict:
    fid = file_id_from_path(file_path)
    base = {
        "file_id":   fid,
        "file_path": file_path,
        "file_name": Path(file_path).name,
        "extension": Path(file_path).suffix,
        "lane":      "TEXT_EXTRACTABLE",
        "status":    "DISCOVERED",
    }
    return {**base, **kwargs}


def _chunk(file_id: str, ref_value: str = "p1", text: str = "hello world") -> dict:
    chunk_id = hashlib.sha256(f"{file_id}::{ref_value}".encode()).hexdigest()
    return {
        "chunk_id":  chunk_id,
        "file_id":   file_id,
        "ref_value": ref_value,
        "text":      text,
    }


# ---------------------------------------------------------------------------
# Schema / connection tests
# ---------------------------------------------------------------------------

def test_wal_mode_enabled(tmp_path: Path) -> None:
    """get_connection() must enable WAL journal mode."""
    db_path = tmp_path / "db" / "tiga.db"
    c = get_connection(db_path)
    row = c.execute("PRAGMA journal_mode").fetchone()
    assert row[0] == "wal"
    c.close()


def test_foreign_keys_enabled(conn) -> None:
    fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
    assert fk == 1


def test_all_tables_exist(conn) -> None:
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    for expected in ("files", "chunks", "projects", "sessions", "messages", "events"):
        assert expected in tables, f"Missing table: {expected}"


def test_fts_table_exists(conn) -> None:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='chunks_fts'"
    ).fetchone()
    assert row is not None


# ---------------------------------------------------------------------------
# files helpers
# ---------------------------------------------------------------------------

def test_upsert_file_idempotent(conn) -> None:
    """Upserting the same file twice produces exactly one row."""
    upsert_file(conn, _file())
    upsert_file(conn, _file())
    count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    assert count == 1


def test_upsert_file_status_updated(conn) -> None:
    """Second upsert with a new status field is accepted."""
    upsert_file(conn, _file(status="DISCOVERED"))
    # update to EXTRACTED via set_file_status
    fid = file_id_from_path("/archive/test.pdf")
    set_file_status(conn, fid, "EXTRACTED")
    row = conn.execute("SELECT status FROM files WHERE file_id=?", (fid,)).fetchone()
    assert row["status"] == "EXTRACTED"


def test_set_file_status_with_error(conn) -> None:
    upsert_file(conn, _file())
    fid = file_id_from_path("/archive/test.pdf")
    set_file_status(conn, fid, "FAILED", error_code="EXTRACT_ERR", error_detail="corrupt")
    row = conn.execute("SELECT * FROM files WHERE file_id=?", (fid,)).fetchone()
    assert row["status"] == "FAILED"
    assert row["error_code"] == "EXTRACT_ERR"


# ---------------------------------------------------------------------------
# chunks helpers
# ---------------------------------------------------------------------------

def test_upsert_chunk_creates_row(conn) -> None:
    upsert_file(conn, _file())
    fid = file_id_from_path("/archive/test.pdf")
    upsert_chunk(conn, _chunk(fid))
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == 1


def test_upsert_chunk_idempotent(conn) -> None:
    upsert_file(conn, _file())
    fid = file_id_from_path("/archive/test.pdf")
    ch = _chunk(fid)
    upsert_chunk(conn, ch)
    upsert_chunk(conn, ch)
    count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    assert count == 1


def test_chunk_cascade_delete(conn) -> None:
    """Deleting a file cascades to its chunks."""
    upsert_file(conn, _file())
    fid = file_id_from_path("/archive/test.pdf")
    upsert_chunk(conn, _chunk(fid, "p1"))
    upsert_chunk(conn, _chunk(fid, "p2"))
    conn.execute("DELETE FROM files WHERE file_id=?", (fid,))
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM chunks WHERE file_id=?", (fid,)).fetchone()[0]
    assert count == 0


# ---------------------------------------------------------------------------
# FTS search
# ---------------------------------------------------------------------------

def test_fts_search_matches_text(conn) -> None:
    upsert_file(conn, _file())
    fid = file_id_from_path("/archive/test.pdf")
    upsert_chunk(conn, _chunk(fid, "p1", "hospital tender submission Singapore"))
    results = fts_search(conn, "hospital", top_k=5)
    assert len(results) == 1
    assert results[0]["chunk_id"] is not None


def test_fts_search_malformed_query_returns_empty(conn) -> None:
    """Malformed FTS5 query must not raise — return [] instead."""
    results = fts_search(conn, 'AND OR "unclosed', top_k=5)
    assert isinstance(results, list)


# ---------------------------------------------------------------------------
# projects / sessions / messages / events
# ---------------------------------------------------------------------------

def test_upsert_project(conn) -> None:
    upsert_project(conn, {"project_id": "2023_HDB", "typology_guess": "residential", "file_count": 12})
    row = conn.execute("SELECT * FROM projects WHERE project_id='2023_HDB'").fetchone()
    assert row["file_count"] == 12


def test_session_and_messages(conn) -> None:
    create_session(conn, "sess-001")
    add_message(conn, "sess-001", "user", "Find hospital drawings")
    add_message(conn, "sess-001", "assistant", "Here are the results…", citations=["file1.pdf"])
    msgs = conn.execute(
        "SELECT * FROM messages WHERE session_id='sess-001' ORDER BY message_id"
    ).fetchall()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["citations"] is not None


def test_log_event(conn) -> None:
    log_event(conn, "INDEX_START", detail="indexing kicked off")
    count = conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
    assert count == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def test_get_stats_empty(conn) -> None:
    stats = get_stats(conn)
    assert stats["total_files"] == 0
    assert stats["total_chunks"] == 0


def test_get_stats_counts(conn) -> None:
    upsert_file(conn, _file("/archive/a.pdf", status="INDEXED"))
    upsert_file(conn, _file("/archive/b.pdf", status="INDEXED"))
    upsert_file(conn, _file("/archive/c.pdf", status="FAILED"))
    fid = file_id_from_path("/archive/a.pdf")
    upsert_chunk(conn, _chunk(fid, "p1", "some text"))
    stats = get_stats(conn)
    assert stats["INDEXED"] == 2
    assert stats["FAILED"] == 1
    assert stats["total_files"] == 3
    assert stats["total_chunks"] == 1


# ---------------------------------------------------------------------------
# Fingerprint utilities
# ---------------------------------------------------------------------------

def test_compute_fingerprint_is_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "sample.txt"
    f.write_bytes(b"hello world" * 1000)
    h1 = compute_fingerprint(str(f))
    h2 = compute_fingerprint(str(f))
    assert h1 == h2
    assert len(h1) == 64  # hex sha256


def test_compute_fingerprint_differs_on_change(tmp_path: Path) -> None:
    f = tmp_path / "sample.txt"
    f.write_bytes(b"version 1")
    h1 = compute_fingerprint(str(f))
    f.write_bytes(b"version 2")
    h2 = compute_fingerprint(str(f))
    assert h1 != h2


def test_file_id_from_path_is_stable() -> None:
    fid = file_id_from_path("/archive/project/brief.pdf")
    assert len(fid) == 64
    assert fid == file_id_from_path("/archive/project/brief.pdf")
    assert fid != file_id_from_path("/archive/project/other.pdf")
