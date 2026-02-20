"""
core/db.py — SQLite + FTS5 schema and helpers.

Tables
------
files       : one row per discovered file, with pipeline status
chunks      : text chunks extracted from files
projects    : inferred project metadata
sessions    : UI conversation sessions
messages    : chat messages per session
events      : pipeline audit log
chunks_fts  : FTS5 virtual table (BM25 search lane) — over chunks.text
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Fingerprint utilities
# ---------------------------------------------------------------------------

def compute_fingerprint(abs_path: str, block: int = 65536) -> str:
    """Streaming SHA-256 hash of file content for change detection."""
    h = hashlib.sha256()
    with open(abs_path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def file_id_from_path(posix_path: str) -> str:
    """Stable file identifier: sha256 of the POSIX path string."""
    return hashlib.sha256(posix_path.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS files (
    file_id             TEXT PRIMARY KEY,
    file_path           TEXT NOT NULL UNIQUE,
    file_name           TEXT NOT NULL,
    extension           TEXT NOT NULL DEFAULT '',
    size_bytes          INTEGER,
    mtime_epoch         REAL,
    fingerprint_sha256  TEXT,
    lane                TEXT CHECK(lane IN ('TEXT_EXTRACTABLE', 'METADATA_ONLY', 'SKIPPED')),
    status              TEXT NOT NULL DEFAULT 'DISCOVERED'
                             CHECK(status IN ('DISCOVERED', 'EXTRACTED', 'EMBEDDED',
                                             'INDEXED', 'FAILED', 'SKIPPED')),
    project_id          TEXT,
    project_confidence  REAL,
    typology            TEXT,
    typology_confidence REAL,
    error_code          TEXT,
    error_detail        TEXT,
    updated_at          TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_files_status  ON files(status);
CREATE INDEX IF NOT EXISTS idx_files_project ON files(project_id);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       TEXT PRIMARY KEY,
    file_id        TEXT NOT NULL REFERENCES files(file_id) ON DELETE CASCADE,
    ref_value      TEXT NOT NULL,
    text           TEXT NOT NULL DEFAULT '',
    token_estimate INTEGER,
    content_hash   TEXT,
    embedded       INTEGER NOT NULL DEFAULT 0,
    created_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_chunks_file     ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedded ON chunks(embedded);

CREATE TABLE IF NOT EXISTS projects (
    project_id     TEXT PRIMARY KEY,
    typology_guess TEXT,
    signals        TEXT,
    file_count     INTEGER NOT NULL DEFAULT 0,
    updated_at     TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    last_active TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS messages (
    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    role       TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
    content    TEXT NOT NULL,
    citations  TEXT,
    ts         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS events (
    event_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    file_id    TEXT,
    event_type TEXT,
    detail     TEXT,
    ts         TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS feedback (
    feedback_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT,
    query         TEXT NOT NULL,
    result_id     TEXT,
    rating        INTEGER CHECK(rating IN (1, -1)),
    rank_position INTEGER,
    comment       TEXT,
    ts            TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id  INTEGER PRIMARY KEY AUTOINCREMENT,
    actor     TEXT NOT NULL DEFAULT 'admin',
    action    TEXT NOT NULL,
    detail    TEXT,
    ts        TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    text,
    chunk_id UNINDEXED,
    content='chunks',
    content_rowid='rowid',
    tokenize='porter ascii'
);

CREATE TRIGGER IF NOT EXISTS chunks_fts_insert
AFTER INSERT ON chunks BEGIN
    INSERT INTO chunks_fts(rowid, text, chunk_id)
    VALUES (new.rowid, new.text, new.chunk_id);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_update
AFTER UPDATE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, chunk_id)
    VALUES ('delete', old.rowid, old.text, old.chunk_id);
    INSERT INTO chunks_fts(rowid, text, chunk_id)
    VALUES (new.rowid, new.text, new.chunk_id);
END;

CREATE TRIGGER IF NOT EXISTS chunks_fts_delete
AFTER DELETE ON chunks BEGIN
    INSERT INTO chunks_fts(chunks_fts, rowid, text, chunk_id)
    VALUES ('delete', old.rowid, old.text, old.chunk_id);
END;
"""


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def get_connection(db_path: Path) -> sqlite3.Connection:
    """Return a WAL-mode, FK-enabled connection with row_factory set."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    path_str = str(db_path.resolve())
    conn = sqlite3.connect(path_str, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# files helpers
# ---------------------------------------------------------------------------

def upsert_file(conn: sqlite3.Connection, f: dict[str, Any]) -> None:
    """
    Insert or update a files row.
    Required keys: file_id, file_path, file_name.
    Optional: extension, size_bytes, mtime_epoch, fingerprint_sha256, lane,
              status, project_id, project_confidence, typology, typology_confidence.
    """
    sql = """
        INSERT INTO files
            (file_id, file_path, file_name, extension,
             size_bytes, mtime_epoch, fingerprint_sha256,
             lane, status, project_id, project_confidence,
             typology, typology_confidence,
             error_code, error_detail, updated_at)
        VALUES
            (:file_id, :file_path, :file_name, :extension,
             :size_bytes, :mtime_epoch, :fingerprint_sha256,
             :lane, :status, :project_id, :project_confidence,
             :typology, :typology_confidence,
             :error_code, :error_detail, datetime('now'))
        ON CONFLICT(file_id) DO UPDATE SET
            file_path           = excluded.file_path,
            file_name           = excluded.file_name,
            extension           = excluded.extension,
            size_bytes          = excluded.size_bytes,
            mtime_epoch         = excluded.mtime_epoch,
            fingerprint_sha256  = excluded.fingerprint_sha256,
            lane                = COALESCE(excluded.lane, lane),
            status              = COALESCE(excluded.status, status),
            project_id          = COALESCE(excluded.project_id, project_id),
            project_confidence  = COALESCE(excluded.project_confidence, project_confidence),
            typology            = COALESCE(excluded.typology, typology),
            typology_confidence = COALESCE(excluded.typology_confidence, typology_confidence),
            error_code          = excluded.error_code,
            error_detail        = excluded.error_detail,
            updated_at          = datetime('now')
    """
    conn.execute(sql, {
        "file_id":             f["file_id"],
        "file_path":           f["file_path"],
        "file_name":           f["file_name"],
        "extension":           f.get("extension", ""),
        "size_bytes":          f.get("size_bytes"),
        "mtime_epoch":         f.get("mtime_epoch"),
        "fingerprint_sha256":  f.get("fingerprint_sha256"),
        "lane":                f.get("lane"),
        "status":              f.get("status", "DISCOVERED"),
        "project_id":          f.get("project_id"),
        "project_confidence":  f.get("project_confidence"),
        "typology":            f.get("typology"),
        "typology_confidence": f.get("typology_confidence"),
        "error_code":          f.get("error_code"),
        "error_detail":        f.get("error_detail"),
    })
    conn.commit()


def set_file_status(
    conn: sqlite3.Connection,
    file_id: str,
    status: str,
    error_code: str | None = None,
    error_detail: str | None = None,
) -> None:
    """Update pipeline status (and optional error info) for a file."""
    conn.execute(
        """UPDATE files
           SET status=?, error_code=?, error_detail=?, updated_at=datetime('now')
           WHERE file_id=?""",
        (status, error_code, error_detail, file_id),
    )
    conn.commit()


def get_file_by_path(conn: sqlite3.Connection, posix_path: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM files WHERE file_path = ?", (posix_path,)
    ).fetchone()


def get_files_by_status(conn: sqlite3.Connection, status: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM files WHERE status = ?", (status,)
    ).fetchall()


# ---------------------------------------------------------------------------
# chunks helpers
# ---------------------------------------------------------------------------

def upsert_chunk(conn: sqlite3.Connection, c: dict[str, Any]) -> None:
    """
    Insert or update a chunks row.
    Required: chunk_id, file_id, ref_value.
    Optional: text, token_estimate, content_hash, embedded.
    """
    sql = """
        INSERT INTO chunks
            (chunk_id, file_id, ref_value, text,
             token_estimate, content_hash, embedded, created_at)
        VALUES
            (:chunk_id, :file_id, :ref_value, :text,
             :token_estimate, :content_hash, :embedded, datetime('now'))
        ON CONFLICT(chunk_id) DO UPDATE SET
            text           = excluded.text,
            token_estimate = excluded.token_estimate,
            content_hash   = excluded.content_hash,
            embedded       = excluded.embedded
    """
    conn.execute(sql, {
        "chunk_id":       c["chunk_id"],
        "file_id":        c["file_id"],
        "ref_value":      c["ref_value"],
        "text":           c.get("text", ""),
        "token_estimate": c.get("token_estimate"),
        "content_hash":   c.get("content_hash"),
        "embedded":       c.get("embedded", 0),
    })
    conn.commit()


def set_chunk_embedded(
    conn: sqlite3.Connection,
    chunk_id: str,
    embedded: int = 1,
) -> None:
    conn.execute(
        "UPDATE chunks SET embedded=? WHERE chunk_id=?", (embedded, chunk_id)
    )
    conn.commit()


def get_chunks_by_file(conn: sqlite3.Connection, file_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM chunks WHERE file_id = ?", (file_id,)
    ).fetchall()


def get_pending_chunks(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Return chunks not yet embedded (embedded=0)."""
    return conn.execute(
        "SELECT * FROM chunks WHERE embedded = 0"
    ).fetchall()


# ---------------------------------------------------------------------------
# FTS search (chunks)
# ---------------------------------------------------------------------------

def fts_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 10,
) -> list[sqlite3.Row]:
    """BM25 full-text search over chunks.text. Returns chunk rows with bm25_score."""
    safe_query = query.replace('"', '""')
    sql = """
        SELECT c.*, bm25(chunks_fts) AS bm25_score
        FROM chunks_fts
        JOIN chunks c ON c.rowid = chunks_fts.rowid
        WHERE chunks_fts MATCH ?
        ORDER BY bm25_score
        LIMIT ?
    """
    try:
        return conn.execute(sql, (safe_query, top_k)).fetchall()
    except sqlite3.OperationalError:
        return []


# ---------------------------------------------------------------------------
# projects helpers
# ---------------------------------------------------------------------------

def upsert_project(conn: sqlite3.Connection, p: dict[str, Any]) -> None:
    sql = """
        INSERT INTO projects (project_id, typology_guess, signals, file_count, updated_at)
        VALUES (:project_id, :typology_guess, :signals, :file_count, datetime('now'))
        ON CONFLICT(project_id) DO UPDATE SET
            typology_guess = COALESCE(excluded.typology_guess, typology_guess),
            signals        = COALESCE(excluded.signals, signals),
            file_count     = excluded.file_count,
            updated_at     = datetime('now')
    """
    conn.execute(sql, {
        "project_id":    p["project_id"],
        "typology_guess": p.get("typology_guess"),
        "signals":       p.get("signals"),
        "file_count":    p.get("file_count", 0),
    })
    conn.commit()


# ---------------------------------------------------------------------------
# sessions / messages helpers
# ---------------------------------------------------------------------------

def create_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)", (session_id,)
    )
    conn.commit()


def touch_session(conn: sqlite3.Connection, session_id: str) -> None:
    conn.execute(
        "UPDATE sessions SET last_active=datetime('now') WHERE session_id=?",
        (session_id,),
    )
    conn.commit()


def add_message(
    conn: sqlite3.Connection,
    session_id: str,
    role: str,
    content: str,
    citations: list[str] | None = None,
) -> int:
    cur = conn.execute(
        "INSERT INTO messages (session_id, role, content, citations) VALUES (?, ?, ?, ?)",
        (session_id, role, content, json.dumps(citations) if citations else None),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def get_messages(conn: sqlite3.Connection, session_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM messages WHERE session_id=? ORDER BY message_id", (session_id,)
    ).fetchall()


# ---------------------------------------------------------------------------
# events helper
# ---------------------------------------------------------------------------

def log_event(
    conn: sqlite3.Connection,
    event_type: str,
    detail: str | None = None,
    file_id: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO events (file_id, event_type, detail) VALUES (?, ?, ?)",
        (file_id, event_type, detail),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def get_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    """Return file counts by status and total chunk count."""
    rows = conn.execute(
        "SELECT status, COUNT(*) AS cnt FROM files GROUP BY status"
    ).fetchall()
    stats: dict[str, Any] = {r["status"]: r["cnt"] for r in rows}
    stats["total_files"] = sum(v for k, v in stats.items())
    stats["total_chunks"] = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    return stats
