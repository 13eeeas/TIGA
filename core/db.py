"""
core/db.py — SQLite + FTS5 schema and helpers.

Schema
------
documents          : one row per indexed file
documents_fts      : FTS5 virtual table (BM25 search lane)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path     TEXT    NOT NULL UNIQUE,   -- POSIX string, stored absolute
    file_name     TEXT    NOT NULL,
    extension     TEXT    NOT NULL,
    project       TEXT    NOT NULL DEFAULT 'Unknown',
    typology      TEXT    NOT NULL DEFAULT 'Unknown',
    title         TEXT    NOT NULL DEFAULT '',
    surrogate     TEXT    NOT NULL DEFAULT '',  -- human-readable stand-in for unsearchable files
    text_content  TEXT    NOT NULL DEFAULT '',  -- extracted text (may be empty for surrogates)
    word_count    INTEGER NOT NULL DEFAULT 0,
    file_size     INTEGER NOT NULL DEFAULT 0,   -- bytes
    modified_at   TEXT,                          -- ISO-8601
    indexed_at    TEXT    NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    title,
    surrogate,
    text_content,
    content='documents',
    content_rowid='id',
    tokenize='porter ascii'
);

-- Keep FTS in sync via triggers
CREATE TRIGGER IF NOT EXISTS documents_fts_insert
    AFTER INSERT ON documents BEGIN
    INSERT INTO documents_fts(rowid, title, surrogate, text_content)
    VALUES (new.id, new.title, new.surrogate, new.text_content);
END;

CREATE TRIGGER IF NOT EXISTS documents_fts_update
    AFTER UPDATE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, surrogate, text_content)
    VALUES ('delete', old.id, old.title, old.surrogate, old.text_content);
    INSERT INTO documents_fts(rowid, title, surrogate, text_content)
    VALUES (new.id, new.title, new.surrogate, new.text_content);
END;

CREATE TRIGGER IF NOT EXISTS documents_fts_delete
    AFTER DELETE ON documents BEGIN
    INSERT INTO documents_fts(documents_fts, rowid, title, surrogate, text_content)
    VALUES ('delete', old.id, old.title, old.surrogate, old.text_content);
END;
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection(db_path: Path) -> sqlite3.Connection:
    """Return a WAL-mode connection with row_factory set."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Windows long path safety
    path_str = str(db_path.resolve())
    conn = sqlite3.connect(path_str, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def upsert_document(conn: sqlite3.Connection, doc: dict[str, Any]) -> int:
    """
    Insert or replace a document row.
    Returns the row id.
    `doc` must include: file_path (POSIX str), file_name, extension, project,
    typology, title, surrogate, text_content, word_count, file_size, modified_at.
    """
    sql = """
        INSERT INTO documents
            (file_path, file_name, extension, project, typology,
             title, surrogate, text_content, word_count, file_size,
             modified_at, indexed_at)
        VALUES
            (:file_path, :file_name, :extension, :project, :typology,
             :title, :surrogate, :text_content, :word_count, :file_size,
             :modified_at, datetime('now'))
        ON CONFLICT(file_path) DO UPDATE SET
            file_name    = excluded.file_name,
            extension    = excluded.extension,
            project      = excluded.project,
            typology     = excluded.typology,
            title        = excluded.title,
            surrogate    = excluded.surrogate,
            text_content = excluded.text_content,
            word_count   = excluded.word_count,
            file_size    = excluded.file_size,
            modified_at  = excluded.modified_at,
            indexed_at   = datetime('now')
    """
    cur = conn.execute(sql, doc)
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def fts_search(
    conn: sqlite3.Connection,
    query: str,
    top_k: int = 10,
) -> list[sqlite3.Row]:
    """
    BM25 full-text search over title + surrogate + text_content.
    Returns rows ordered by relevance (bm25 is negative, lower = better).
    """
    # Sanitise query to avoid FTS5 syntax errors
    safe_query = query.replace('"', '""')
    sql = """
        SELECT d.*, bm25(documents_fts) AS bm25_score
        FROM documents_fts
        JOIN documents d ON d.id = documents_fts.rowid
        WHERE documents_fts MATCH ?
        ORDER BY bm25_score
        LIMIT ?
    """
    try:
        return conn.execute(sql, (safe_query, top_k)).fetchall()
    except sqlite3.OperationalError:
        # Malformed FTS query — return empty rather than crash
        return []


def get_by_path(conn: sqlite3.Connection, file_path: str) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM documents WHERE file_path = ?", (file_path,)
    ).fetchone()


def get_all_paths(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute("SELECT file_path FROM documents").fetchall()
    return [r["file_path"] for r in rows]


def get_stats(conn: sqlite3.Connection) -> dict[str, Any]:
    row = conn.execute(
        "SELECT COUNT(*) AS total, SUM(word_count) AS words FROM documents"
    ).fetchone()
    return {"total_documents": row["total"], "total_words": row["words"] or 0}
