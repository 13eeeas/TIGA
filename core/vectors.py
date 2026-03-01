"""
core/vectors.py — LanceDB vector store + Ollama embeddings.

Two tables:
  tiga_chunks    — chunk-level vectors (current, chunk-based pipeline)
  tiga_documents — document-level vectors (legacy, kept for query.py compat)

Public API (Chunk B):
  EmbedError                         — raised when Ollama is unreachable/fails
  embed_text(text, cfg)              — embed single text, raises EmbedError
  embed_texts_batched(texts, cfg)    — batch embed with VRAM-friendly pacing
  upsert_chunk(chunk_id, text, metadata, cfg) -> bool
  query_vector(text, top_k, cfg, filters) -> list[dict]

Legacy API (kept for backward compat until query.py / index.py are rewritten):
  embed_texts(texts)        — batch embed, raises on failure
  upsert_documents(docs, db)
  vector_search(query, top_k, db)
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
from typing import Any

import lancedb
import pyarrow as pa

from config import cfg as _module_cfg, Config

logger = logging.getLogger(__name__)

_EMBED_DIM      = 768   # nomic-embed-text output dimension
_TABLE_CHUNKS   = "tiga_chunks"
_TABLE_DOCS     = "tiga_documents"   # legacy


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------

class EmbedError(Exception):
    """Raised when Ollama embedding call fails."""


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

def embed_text(text: str, cfg: Config | None = None) -> list[float]:
    """
    Embed a single text string via Ollama.
    Raises EmbedError on any failure (network, timeout, bad response).
    """
    _cfg = cfg or _module_cfg
    url = _cfg.ollama_base_url.rstrip("/") + "/api/embeddings"
    payload = json.dumps({
        "model":  _cfg.embed_model,
        "prompt": text,
    }).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=_cfg.ollama_timeout) as resp:
            data = json.loads(resp.read())
            return data["embedding"]
    except Exception as exc:
        raise EmbedError(str(exc)) from exc


def embed_texts_batch(
    texts: list[str],
    cfg: Config | None = None,
) -> list[list[float] | None]:
    """
    Batch-embed texts using Ollama /api/embed (array input, single request).
    Much faster than calling embed_text per item.
    Returns None for any item that failed.
    """
    if not texts:
        return []
    _cfg = cfg or _module_cfg
    url = _cfg.ollama_base_url.rstrip("/") + "/api/embed"
    payload = json.dumps({"model": _cfg.embed_model, "input": texts}).encode()
    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=max(_cfg.ollama_timeout, 120)) as resp:
            data = json.loads(resp.read())
            embeddings = data.get("embeddings", [])
            if len(embeddings) != len(texts):
                raise EmbedError(f"Expected {len(texts)} embeddings, got {len(embeddings)}")
            return [list(e) for e in embeddings]
    except EmbedError:
        raise
    except Exception as exc:
        raise EmbedError(str(exc)) from exc


def embed_texts_batched(
    texts: list[str],
    cfg: Config | None = None,
    *,
    batch_size: int | None = None,
) -> list[list[float] | None]:
    """
    Embed a list of texts in batches using the fast batch API.
    Falls back to per-item embed_text on batch failure.
    Sleeps cfg.embed_batch_sleep_s between batches.
    Returns None for any text that failed to embed.

    Args:
        texts:      Texts to embed.
        cfg:        Config object (defaults to module-level singleton).
        batch_size: Override batch size (e.g. from scheduler mode).
                    If None, uses cfg.embed_batch_size directly.
    """
    _cfg = cfg or _module_cfg
    results: list[list[float] | None] = []
    # Respect caller-supplied override (e.g. from scheduler night/day mode).
    # No forced minimum — the scheduler controls this deliberately.
    _batch_size = batch_size if batch_size is not None else _cfg.embed_batch_size
    sleep_s = _cfg.embed_batch_sleep_s

    for i in range(0, len(texts), _batch_size):
        batch = texts[i : i + _batch_size]
        try:
            batch_results = embed_texts_batch(batch, _cfg)
            results.extend(batch_results)
        except EmbedError as e:
            logger.warning("Batch embed failed, falling back to per-item: %s", e)
            for text in batch:
                try:
                    results.append(embed_text(text, _cfg))
                except EmbedError as e2:
                    logger.warning("Embed failed: %s", e2)
                    results.append(None)
        if i + _batch_size < len(texts):
            time.sleep(sleep_s)

    return results


# ---------------------------------------------------------------------------
# LanceDB helpers — chunks table
# ---------------------------------------------------------------------------

def _chunk_schema(dim: int = _EMBED_DIM) -> pa.Schema:
    return pa.schema([
        pa.field("chunk_id",   pa.string()),
        pa.field("file_id",    pa.string()),
        pa.field("ref_value",  pa.string()),
        pa.field("rel_path",   pa.string()),
        pa.field("project_id", pa.string()),
        pa.field("typology",   pa.string()),
        pa.field("ext",        pa.string()),
        pa.field("root_id",    pa.string()),
        pa.field("vector",     pa.list_(pa.float32(), dim)),
    ])


def get_db() -> lancedb.DBConnection:
    return lancedb.connect(str(_module_cfg.get_vector_dir()))


def _get_chunk_table(db: lancedb.DBConnection) -> lancedb.table.Table:
    if _TABLE_CHUNKS in db.table_names():
        return db.open_table(_TABLE_CHUNKS)
    return db.create_table(_TABLE_CHUNKS, schema=_chunk_schema())


# ---------------------------------------------------------------------------
# Chunk upsert
# ---------------------------------------------------------------------------

def upsert_chunk(
    chunk_id: str,
    text: str,
    metadata: dict[str, Any],
    cfg: Config | None = None,
) -> bool:
    """
    Embed text and upsert chunk into LanceDB.

    metadata keys: file_id, rel_path, ref_value, project_id, typology, ext, root_id

    Returns True on success.
    On EmbedError: logs warning, returns False (caller marks embedded=0 in SQLite).
    """
    _cfg = cfg or _module_cfg
    try:
        vec = embed_text(text, _cfg)
    except EmbedError as e:
        logger.warning("Embedding failed for chunk %s: %s", chunk_id, e)
        return False

    row = {
        "chunk_id":   chunk_id,
        "file_id":    metadata.get("file_id", ""),
        "ref_value":  metadata.get("ref_value", ""),
        "rel_path":   metadata.get("rel_path", ""),
        "project_id": metadata.get("project_id", ""),
        "typology":   metadata.get("typology", ""),
        "ext":        metadata.get("ext", ""),
        "root_id":    metadata.get("root_id", ""),
        "vector":     [float(v) for v in vec],
    }

    db = get_db()
    table = _get_chunk_table(db)
    try:
        (
            table.merge_insert("chunk_id")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute([row])
        )
    except Exception as e:
        logger.error("LanceDB upsert failed for chunk %s: %s", chunk_id, e)
        return False

    return True


# ---------------------------------------------------------------------------
# Vector search (chunks)
# ---------------------------------------------------------------------------

def query_vector(
    text: str,
    top_k: int,
    cfg: Config | None = None,
    filters: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search over chunks.
    filters: optional dict with keys project_id, typology, ext.
    Returns: [{chunk_id, distance, metadata}]
    """
    _cfg = cfg or _module_cfg
    try:
        vec = embed_text(text, _cfg)
    except EmbedError as e:
        logger.error("Embedding query failed: %s", e)
        return []

    db = get_db()
    table = _get_chunk_table(db)

    if table.count_rows() == 0:
        return []

    search = table.search(vec).metric("cosine").limit(top_k * 2)

    if filters:
        parts = []
        for key in ("project_id", "typology", "ext"):
            if key in filters:
                val = str(filters[key]).replace("'", "''")
                parts.append(f"{key} = '{val}'")
        if parts:
            search = search.where(" AND ".join(parts))

    try:
        results = search.to_list()
    except Exception as e:
        logger.error("Vector search failed: %s", e)
        return []

    out: list[dict[str, Any]] = []
    for r in results[:top_k]:
        out.append({
            "chunk_id": r["chunk_id"],
            "distance": round(float(r.get("_distance", 1.0)), 4),
            "metadata": {
                "file_id":    r["file_id"],
                "ref_value":  r["ref_value"],
                "rel_path":   r["rel_path"],
                "project_id": r["project_id"],
                "typology":   r["typology"],
                "ext":        r["ext"],
                "root_id":    r["root_id"],
            },
        })
    return out


# ---------------------------------------------------------------------------
# Legacy API — document-level (backward compat for query.py / index.py)
# Kept until those modules are rewritten in a later chunk.
# ---------------------------------------------------------------------------

def _doc_schema(dim: int = _EMBED_DIM) -> pa.Schema:
    return pa.schema([
        pa.field("file_path",  pa.string()),
        pa.field("file_name",  pa.string()),
        pa.field("extension",  pa.string()),
        pa.field("project",    pa.string()),
        pa.field("typology",   pa.string()),
        pa.field("title",      pa.string()),
        pa.field("vector",     pa.list_(pa.float32(), dim)),
    ])


def _get_doc_table(db: lancedb.DBConnection) -> lancedb.table.Table:
    if _TABLE_DOCS in db.table_names():
        return db.open_table(_TABLE_DOCS)
    return db.create_table(_TABLE_DOCS, schema=_doc_schema())


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Batch embed. Raises EmbedError on first failure."""
    return [embed_text(t) for t in texts]


def upsert_documents(
    docs: list[dict[str, Any]],
    db: lancedb.DBConnection | None = None,
) -> None:
    """Legacy document-level upsert (used by index.py until rewrite)."""
    if not docs:
        return
    _db = db or get_db()
    table = _get_doc_table(_db)
    rows: list[dict[str, Any]] = []
    for doc in docs:
        embed_src = (doc.get("text_content") or doc.get("surrogate", ""))[:3000]
        row = {
            "file_path": doc["file_path"],
            "file_name": doc.get("file_name", ""),
            "extension": doc.get("extension", ""),
            "project":   doc.get("project", "Unknown"),
            "typology":  doc.get("typology", "Unknown"),
            "title":     doc.get("title", ""),
        }
        try:
            row["vector"] = [float(v) for v in embed_text(embed_src)]
            rows.append(row)
        except EmbedError as e:
            logger.warning("Skipping vector for %s: %s", doc.get("file_name"), e)
    if not rows:
        return
    try:
        (
            table.merge_insert("file_path")
            .when_matched_update_all()
            .when_not_matched_insert_all()
            .execute(rows)
        )
        logger.info("Upserted %d document vectors", len(rows))
    except Exception as e:
        logger.error("Document vector upsert failed: %s", e)
        raise


def vector_search(
    query: str,
    top_k: int | None = None,
    db: lancedb.DBConnection | None = None,
) -> list[dict[str, Any]]:
    """Legacy document-level semantic search (used by query.py until rewrite)."""
    k = top_k or _module_cfg.top_k
    _db = db or get_db()
    try:
        table = _get_doc_table(_db)
        if table.count_rows() == 0:
            return []
        query_vec = embed_text(query)
        results = (
            table.search(query_vec)
            .metric("cosine")
            .limit(k * 2)
            .to_list()
        )
    except Exception as e:
        logger.error("Document vector search failed: %s", e)
        return []

    out: list[dict[str, Any]] = []
    for r in results:
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
