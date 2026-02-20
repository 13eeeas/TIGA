"""
core/query.py — Hybrid BM25 + vector search engine.

SearchResult (TypedDict)
  chunk_id, file_id, ref_value, rel_path, file_name,
  project_id, typology, snippet (≤160 chars, context around match),
  bm25_score, vector_score, final_score, citation

Six-step pipeline
-----------------
  Step 1 — BM25 lane (FTS5)
    snippet(chunks_fts, 0, '', '', ' ... ', 25) for context snippets.
    Retrieve top_k*3 candidates; apply filters as WHERE clauses.
    Normalise bm25 scores to [0,1] (most-negative raw → 1.0).

  Step 2 — Vector lane (LanceDB/Ollama)
    embed_text(query) → query_vector(top_k*3, filters).
    score = 1 - distance; normalise to [0,1].
    If Ollama unavailable: skip, log warning, BM25-only (no crash).

  Step 3 — Merge + score
    Union of chunk_ids from both lanes.
    final = alpha * vector_score + (1-alpha) * bm25_score
    Sort DESC final_score; tie-break ASC rel_path.
    Apply offset, return top_k.

  Step 4 — Citation generation
    Single root : rel_path#ref_value
    Multi root  : [root_label] rel_path#ref_value  (root_label = root.name)

  Step 5 — Citation validation
    validate_citation(citation, db_path, root_paths) for every candidate.
    Invalid citations → log ERROR + exclude.
    100% of returned results have valid citations.

  Step 6 — Session memory (if session_id given)
    Save user query to messages table.
    (compose.py loads history and prepends to Ollama prompt.)

alpha = cfg.hybrid_alpha  (default 0.6; 1.0 = pure vector, 0.0 = pure BM25)
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, TypedDict

from config import cfg as _module_cfg, Config
from core.db import (
    get_connection,
    add_message,
    create_session,
    touch_session,
    file_id_from_path,
)
from core.eval import validate_citation

logger = logging.getLogger(__name__)

_SNIPPET_TOKENS  = 25    # approximate tokens returned by FTS5 snippet()
_SNIPPET_MAX     = 160   # hard cap on returned snippet chars
_SESSION_HISTORY = 6     # messages loaded for compose.py context


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class SearchResult(TypedDict):
    chunk_id:     str
    file_id:      str
    ref_value:    str
    rel_path:     str
    file_name:    str
    project_id:   str
    typology:     str
    snippet:      str
    bm25_score:   float
    vector_score: float
    final_score:  float
    citation:     str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fts_escape(query: str) -> str:
    """Remove FTS5 operator characters that would cause syntax errors."""
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    cleaned = " ".join(cleaned.split())
    return cleaned or '""'


def _normalise_bm25(scores: list[float]) -> list[float]:
    """
    FTS5 bm25() is negative; more-negative = better match.
    Map to [0, 1]: best (most negative) → 1.0, worst → 0.0.
    """
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(hi - s) / (hi - lo) for s in scores]


def _normalise(scores: list[float]) -> list[float]:
    """Min-max normalise to [0, 1]; higher = better."""
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _rel_path(file_path: str, roots: list[Path]) -> str:
    """Return path relative to the first matching index_root, else basename."""
    p = Path(file_path)
    for root in roots:
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            continue
    return p.name


def _make_citation(
    file_path: str,
    rel_path: str,
    ref_value: str,
    roots: list[Path],
) -> str:
    """
    Single root  : "rel_path#ref_value"
    Multiple roots: "[root_label] rel_path#ref_value" where
                    root_label = the last component of whichever root owns the file.
    """
    base = f"{rel_path}#{ref_value}"
    if len(roots) <= 1:
        return base
    p = Path(file_path)
    for root in roots:
        try:
            p.relative_to(root)
            return f"[{root.name}] {base}"
        except ValueError:
            continue
    return base


# ---------------------------------------------------------------------------
# BM25 lane
# ---------------------------------------------------------------------------

def _run_bm25(
    query: str,
    limit: int,
    filters: dict[str, str] | None,
    conn: sqlite3.Connection,
) -> list[dict[str, Any]]:
    safe_q = _fts_escape(query)
    params: list[Any] = [safe_q]

    filter_clauses: list[str] = []
    if filters:
        if "project_id" in filters:
            filter_clauses.append("f.project_id = ?")
            params.append(filters["project_id"])
        if "typology" in filters:
            filter_clauses.append("f.typology = ?")
            params.append(filters["typology"])
        if "ext" in filters:
            filter_clauses.append("f.extension = ?")
            params.append(filters["ext"])
        if "root_id" in filters:
            # root_id → prefix match on file_path
            filter_clauses.append("f.file_path LIKE ?")
            params.append(filters["root_id"].rstrip("/") + "/%")

    extra = ("AND " + " AND ".join(filter_clauses)) if filter_clauses else ""

    sql = f"""
        SELECT
            c.chunk_id, c.file_id, c.ref_value, c.text,
            f.file_path,
            COALESCE(f.project_id, 'Unknown') AS project_id,
            COALESCE(f.typology,   'Unknown') AS typology,
            COALESCE(f.extension,  '')        AS extension,
            COALESCE(f.file_name,  '')        AS file_name,
            bm25(chunks_fts)                  AS bm25_raw,
            snippet(chunks_fts, 0, '', '', ' ... ', {_SNIPPET_TOKENS}) AS snippet
        FROM chunks_fts
        JOIN chunks c ON c.rowid = chunks_fts.rowid
        JOIN files  f ON f.file_id = c.file_id
        WHERE chunks_fts MATCH ?
          {extra}
        ORDER BY bm25_raw
        LIMIT ?
    """
    params.append(limit)
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError as e:
        logger.warning("FTS query failed (%s): %s", safe_q, e)
        return []


# ---------------------------------------------------------------------------
# Vector lane
# ---------------------------------------------------------------------------

def _run_vector(
    query: str,
    limit: int,
    filters: dict[str, str] | None,
    cfg_obj: Config,
) -> dict[str, float]:
    """
    Returns {chunk_id: normalised_score}.
    Returns {} on any failure (Ollama down, table empty, etc.).
    """
    try:
        from core.vectors import query_vector
        results = query_vector(query, limit, cfg=cfg_obj, filters=filters)
        if not results:
            return {}
        raw = [max(0.0, 1.0 - r["distance"]) for r in results]
        norm = _normalise(raw)
        return {r["chunk_id"]: ns for r, ns in zip(results, norm)}
    except Exception as e:
        logger.warning("Vector lane unavailable (BM25-only fallback): %s", e)
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(
    query: str,
    top_k: int | None = None,
    offset: int = 0,
    filters: dict[str, str] | None = None,
    session_id: str | None = None,
    cfg_obj: Config | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[SearchResult]:
    """
    Hybrid BM25 + vector search. Returns validated-citation results only.

    Args:
        query:      Natural-language search query.
        top_k:      Max results to return (default: cfg.top_k).
        offset:     Skip first N ranked results (pagination).
        filters:    Optional {project_id, typology, ext, root_id}.
        session_id: If set, saves query to messages table.
        cfg_obj:    Config override (for tests).
        conn:       SQLite connection override (for tests).

    Returns:
        list[SearchResult] — 100% valid citations, sorted by final_score DESC.
    """
    _cfg = cfg_obj or _module_cfg
    _top_k = top_k or _cfg.top_k
    _own_conn = conn is None
    _conn = conn or get_connection(_cfg.get_db_path())

    try:
        return _search_impl(query, _top_k, offset, filters, session_id, _cfg, _conn)
    finally:
        if _own_conn:
            _conn.close()


def _search_impl(
    query: str,
    top_k: int,
    offset: int,
    filters: dict[str, str] | None,
    session_id: str | None,
    _cfg: Config,
    conn: sqlite3.Connection,
) -> list[SearchResult]:

    alpha = _cfg.hybrid_alpha
    roots = _cfg.index_roots
    db_path = str(_cfg.get_db_path())
    root_paths = [str(r) for r in roots]

    # ── Step 1: BM25 ─────────────────────────────────────────────────────────
    bm25_rows = _run_bm25(query, top_k * 3, filters, conn)
    bm25_map: dict[str, dict[str, Any]] = {}
    if bm25_rows:
        raw_scores = [r["bm25_raw"] for r in bm25_rows]
        norm_scores = _normalise_bm25(raw_scores)
        for row, ns in zip(bm25_rows, norm_scores):
            row["bm25_score"] = ns
            bm25_map[row["chunk_id"]] = row

    # ── Step 2: Vector ────────────────────────────────────────────────────────
    vec_scores = _run_vector(query, top_k * 3, filters, _cfg)

    # ── Step 3: Merge ─────────────────────────────────────────────────────────
    all_ids = set(bm25_map) | set(vec_scores)

    # Fetch metadata for vector-only chunks (not in BM25 results)
    missing = all_ids - set(bm25_map)
    if missing:
        placeholders = ",".join("?" * len(missing))
        extra = conn.execute(
            f"SELECT c.chunk_id, c.file_id, c.ref_value, c.text, "
            f"f.file_path, "
            f"COALESCE(f.project_id,'Unknown') AS project_id, "
            f"COALESCE(f.typology,'Unknown')   AS typology, "
            f"COALESCE(f.extension,'')         AS extension, "
            f"COALESCE(f.file_name,'')         AS file_name "
            f"FROM chunks c "
            f"JOIN files f ON f.file_id = c.file_id "
            f"WHERE c.chunk_id IN ({placeholders})",
            list(missing),
        ).fetchall()
        for row in extra:
            bm25_map[row["chunk_id"]] = {
                **dict(row),
                "bm25_raw":   0.0,
                "bm25_score": 0.0,
                "snippet":    row["text"][:_SNIPPET_MAX],
            }

    # Score + sort
    candidates: list[dict[str, Any]] = []
    for cid in all_ids:
        meta = bm25_map.get(cid)
        if not meta:
            continue
        bs = meta.get("bm25_score", 0.0)
        vs = vec_scores.get(cid, 0.0)
        final = alpha * vs + (1.0 - alpha) * bs
        candidates.append({
            "chunk_id":   cid,
            "file_id":    meta["file_id"],
            "ref_value":  meta["ref_value"],
            "file_path":  meta["file_path"],
            "file_name":  meta.get("file_name", Path(meta["file_path"]).name),
            "project_id": meta.get("project_id", "Unknown"),
            "typology":   meta.get("typology", "Unknown"),
            "snippet":    meta.get("snippet", meta.get("text", ""))[:_SNIPPET_MAX],
            "bm25_score": bs,
            "vector_score": vs,
            "final_score":  final,
        })

    candidates.sort(key=lambda r: (-r["final_score"], r["file_path"]))

    # ── Steps 4 & 5: Citation generation + validation ─────────────────────────
    results: list[SearchResult] = []
    for cand in candidates[offset:]:
        rel = _rel_path(cand["file_path"], roots)
        citation = _make_citation(cand["file_path"], rel, cand["ref_value"], roots)

        if not validate_citation(citation, db_path, root_paths):
            logger.error("Invalid citation excluded from results: %s", citation)
            continue

        results.append(SearchResult(
            chunk_id=     cand["chunk_id"],
            file_id=      cand["file_id"],
            ref_value=    cand["ref_value"],
            rel_path=     rel,
            file_name=    cand["file_name"],
            project_id=   cand["project_id"],
            typology=     cand["typology"],
            snippet=      cand["snippet"],
            bm25_score=   round(cand["bm25_score"],   4),
            vector_score= round(cand["vector_score"],  4),
            final_score=  round(cand["final_score"],   4),
            citation=     citation,
        ))

        if len(results) >= top_k:
            break

    # ── Step 6: Session memory ─────────────────────────────────────────────────
    if session_id:
        try:
            create_session(conn, session_id)
            touch_session(conn, session_id)
            add_message(conn, session_id, "user", query)
        except Exception as e:
            logger.warning("Session save failed: %s", e)

    return results


# ---------------------------------------------------------------------------
# Session history loader (called by compose.py)
# ---------------------------------------------------------------------------

def load_session_history(
    session_id: str,
    cfg_obj: Config | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, str]]:
    """
    Return last _SESSION_HISTORY messages as [{role, content}] for Ollama.
    Returns [] on any error.
    """
    _cfg = cfg_obj or _module_cfg
    _own = conn is None
    _conn = conn or get_connection(_cfg.get_db_path())
    try:
        rows = _conn.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id=? ORDER BY message_id DESC LIMIT ?",
            (session_id, _SESSION_HISTORY),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    except Exception as e:
        logger.warning("Failed to load session history: %s", e)
        return []
    finally:
        if _own:
            _conn.close()
