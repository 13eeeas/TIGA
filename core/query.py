"""
core/query.py — Hybrid search engine (BM25 + vector, merged by score).

Strategy:
  1. FTS5 BM25 search (keyword lane)
  2. ChromaDB vector search (semantic lane)
  3. Normalise both score ranges to [0, 1]
  4. Merge with configurable weights
  5. Return top_k deduplicated results
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Any

from config import cfg
from core.db import fts_search, get_connection
from core.vectors import vector_search, get_db

logger = logging.getLogger(__name__)


def _normalise_bm25(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
    """Normalise BM25 scores (negative) to [0, 1]."""
    if not rows:
        return []
    scores = [abs(r["bm25_score"]) for r in rows]
    max_s = max(scores) or 1.0
    return [
        {
            "file_path": row["file_path"],
            "file_name": row["file_name"],
            "project":   row["project"],
            "typology":  row["typology"],
            "title":     row["title"],
            "surrogate": row["surrogate"],
            "bm25_score": round(raw / max_s, 4),
            "vec_score":  0.0,
        }
        for row, raw in zip(rows, scores)
    ]


def _merge(
    fts_results: list[dict[str, Any]],
    vec_results: list[dict[str, Any]],
    fts_weight: float,
    vec_weight: float,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for r in fts_results:
        fp = r["file_path"]
        merged[fp] = {**r, "combined_score": r["bm25_score"] * fts_weight}

    for r in vec_results:
        fp = r["file_path"]
        if fp in merged:
            merged[fp]["vec_score"] = r["score"]
            merged[fp]["combined_score"] += r["score"] * vec_weight
        else:
            merged[fp] = {
                **r,
                "bm25_score": 0.0,
                "vec_score": r["score"],
                "surrogate": r.get("surrogate", ""),
                "combined_score": r["score"] * vec_weight,
            }

    return sorted(merged.values(), key=lambda x: x["combined_score"], reverse=True)


def search(
    query: str,
    top_k: int | None = None,
    conn: sqlite3.Connection | None = None,
    db=None,
) -> list[dict[str, Any]]:
    """
    Hybrid search. Returns top_k results sorted by combined relevance.
    Each result: file_path, file_name, project, typology, title,
    surrogate, bm25_score, vec_score, combined_score.
    """
    k = top_k or cfg.top_k
    _conn = conn or get_connection(cfg.get_db_path())
    _db   = db or get_db()

    fts_rows = fts_search(_conn, query, top_k=k * 2)
    fts_results = _normalise_bm25(fts_rows)
    vec_results = vector_search(query, top_k=k * 2, db=_db)

    merged = _merge(fts_results, vec_results, cfg.fts_weight, cfg.vector_weight)
    return merged[:k]
