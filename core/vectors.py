"""
core/vectors.py — LanceDB vector store + Ollama embeddings.

Replaces ChromaDB (incompatible with Python 3.14).
LanceDB is Apache 2.0, local-first, no Pydantic V1 dependency.

Table: 'tiga_documents'
Row ID: file_path (POSIX string, unique and stable)
"""

from __future__ import annotations

import logging
from typing import Any

import lancedb
import pyarrow as pa
import ollama as ollama_client

from config import cfg

logger = logging.getLogger(__name__)

_TABLE_NAME = "tiga_documents"
_EMBED_DIM  = 768   # nomic-embed-text output dimension


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

def _schema(dim: int = _EMBED_DIM) -> pa.Schema:
    return pa.schema([
        pa.field("file_path",  pa.string()),
        pa.field("file_name",  pa.string()),
        pa.field("extension",  pa.string()),
        pa.field("project",    pa.string()),
        pa.field("typology",   pa.string()),
        pa.field("title",      pa.string()),
        pa.field("vector",     pa.list_(pa.float32(), dim)),
    ])


# ---------------------------------------------------------------------------
# DB + table helpers
# ---------------------------------------------------------------------------

def get_db() -> lancedb.DBConnection:
    return lancedb.connect(str(cfg.get_vector_dir()))


def get_table(db: lancedb.DBConnection | None = None) -> lancedb.table.Table:
    """Return (or create) the tiga_documents table."""
    _db = db or get_db()
    if _TABLE_NAME in _db.table_names():
        return _db.open_table(_TABLE_NAME)
    return _db.create_table(_TABLE_NAME, schema=_schema())


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed texts via Ollama nomic-embed-text.
    Raises on Ollama connection failure.
    """
    embeddings: list[list[float]] = []
    for text in texts:
        resp = ollama_client.embeddings(
            model=cfg.embed_model,
            prompt=text,
        )
        embeddings.append(resp["embedding"])
    return embeddings


# ---------------------------------------------------------------------------
# Upsert
# ---------------------------------------------------------------------------

def upsert_documents(
    docs: list[dict[str, Any]],
    db: lancedb.DBConnection | None = None,
) -> None:
    """
    Upsert documents into LanceDB (insert or overwrite by file_path).
    Each doc must have: file_path, file_name, extension, project,
    typology, title, text_content/surrogate.
    """
    if not docs:
        return

    _db = db or get_db()
    table = get_table(_db)

    rows: list[dict[str, Any]] = []
    texts_to_embed: list[str] = []

    for doc in docs:
        embed_text = (doc.get("text_content") or doc.get("surrogate", ""))[:8000]
        texts_to_embed.append(embed_text)
        rows.append({
            "file_path": doc["file_path"],
            "file_name": doc.get("file_name", ""),
            "extension": doc.get("extension", ""),
            "project":   doc.get("project", "Unknown"),
            "typology":  doc.get("typology", "Unknown"),
            "title":     doc.get("title", ""),
        })

    try:
        embeddings = embed_texts(texts_to_embed)
        for row, vec in zip(rows, embeddings):
            row["vector"] = [float(v) for v in vec]

        # Upsert: overwrite existing rows with same file_path
        (
            table.merge_insert("file_path")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
        logger.info("Upserted %d vectors", len(rows))
    except Exception as e:
        logger.error("Vector upsert failed: %s", e)
        raise


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def vector_search(
    query: str,
    top_k: int | None = None,
    db: lancedb.DBConnection | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search. Returns results with keys:
    file_path, file_name, project, typology, title, score (0-1, higher=better).
    """
    k = top_k or cfg.top_k
    _db = db or get_db()

    try:
        table = get_table(_db)
        if table.count_rows() == 0:
            return []
        query_vec = embed_texts([query])[0]
        results = (
            table.search(query_vec)
            .metric("cosine")
            .limit(k * 2)
            .to_list()
        )
    except Exception as e:
        logger.error("Vector search failed: %s", e)
        return []

    out: list[dict[str, Any]] = []
    for r in results:
        # LanceDB cosine distance: 0 = identical, up to 2 = opposite
        score = max(0.0, 1.0 - r.get("_distance", 1.0))
        out.append({
            "file_path": r["file_path"],
            "file_name": r["file_name"],
            "project":   r["project"],
            "typology":  r["typology"],
            "title":     r["title"],
            "score":     round(score, 4),
        })

    return out
