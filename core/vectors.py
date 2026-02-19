"""
core/vectors.py — ChromaDB vector store + Ollama embeddings.

Collection: 'tiga_documents'
Document IDs are POSIX file path strings (unique and stable).
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from chromadb.config import Settings
import ollama as ollama_client

from config import cfg

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "tiga_documents"


def get_collection() -> chromadb.Collection:
    """Return (or create) the persistent ChromaDB collection."""
    client = chromadb.PersistentClient(
        path=str(cfg.get_vector_dir()),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed texts via Ollama nomic-embed-text, one at a time.
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


def upsert_documents(
    docs: list[dict[str, Any]],
    collection: chromadb.Collection | None = None,
) -> None:
    """
    Upsert documents into ChromaDB.
    Each doc must have: file_path (POSIX str), title, surrogate,
    text_content, project, typology.
    """
    if not docs:
        return

    col = collection or get_collection()

    ids: list[str] = []
    texts_to_embed: list[str] = []
    metadatas: list[dict[str, str]] = []

    for doc in docs:
        ids.append(doc["file_path"])
        embed_text = (doc.get("text_content") or doc.get("surrogate", ""))[:8000]
        texts_to_embed.append(embed_text)
        metadatas.append({
            "file_path": doc["file_path"],
            "file_name": doc.get("file_name", ""),
            "extension": doc.get("extension", ""),
            "project":   doc.get("project", "Unknown"),
            "typology":  doc.get("typology", "Unknown"),
            "title":     doc.get("title", ""),
        })

    try:
        embeddings = embed_texts(texts_to_embed)
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=texts_to_embed,
        )
        logger.info("Upserted %d vectors", len(ids))
    except Exception as e:
        logger.error("Vector upsert failed: %s", e)
        raise


def vector_search(
    query: str,
    top_k: int | None = None,
    collection: chromadb.Collection | None = None,
) -> list[dict[str, Any]]:
    """
    Semantic search. Returns results with keys:
    file_path, file_name, project, typology, title, score (0-1).
    """
    k = top_k or cfg.top_k
    col = collection or get_collection()

    try:
        query_embedding = embed_texts([query])[0]
        count = col.count()
        if count == 0:
            return []
        results = col.query(
            query_embeddings=[query_embedding],
            n_results=min(k * 2, count),
            include=["metadatas", "distances"],
        )
    except Exception as e:
        logger.error("Vector search failed: %s", e)
        return []

    out: list[dict[str, Any]] = []
    if not results["ids"] or not results["ids"][0]:
        return out

    for doc_id, meta, dist in zip(
        results["ids"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        score = 1.0 - (dist / 2.0)
        out.append({
            "file_path": doc_id,
            "file_name": meta.get("file_name", ""),
            "project":   meta.get("project", "Unknown"),
            "typology":  meta.get("typology", "Unknown"),
            "title":     meta.get("title", ""),
            "score":     round(score, 4),
        })

    return out
