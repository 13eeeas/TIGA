"""
core/reranker.py — Cross-encoder reranker for TIGA Hunt.

Why this matters
----------------
Bi-encoder embeddings (nomic-embed-text) produce a query vector and a chunk
vector separately, then compare them by cosine distance.  This means the model
never "sees" the query and chunk together, so subtle relevance signals are lost.

A cross-encoder reads (query, chunk) as a single concatenated sequence and
outputs a direct relevance score.  This is slower per pair but far more
accurate — it's the same technique ChatGPT Projects uses internally to sharpen
its citation precision.

Model
-----
  cross-encoder/ms-marco-MiniLM-L-6-v2
    - 80 MB on disk, runs on CPU
    - Trained on MS MARCO passage ranking (generalises well to domain docs)
    - ~15–20 ms for 20 pairs on a modern CPU
    - Download happens once on first use (cached in ~/.cache/huggingface/)

Usage
-----
  retrieval:
    reranker_enabled: true          # enable in config.yaml
    reranker_model: cross-encoder/ms-marco-MiniLM-L-6-v2
    reranker_top_k: 20              # how many hybrid candidates to rerank

Installation
------------
  pip install sentence-transformers

Fallback
--------
  If sentence-transformers is not installed, reranking is silently skipped
  and the hybrid BM25+vector score is used unchanged (no crash, no warning
  beyond a one-time INFO log).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_reranker_instance: Any = None  # lazy-loaded singleton
_reranker_model_name: str | None = None
_warned_unavailable: bool = False


# ---------------------------------------------------------------------------
# Model loader (lazy, thread-safe enough for single-server use)
# ---------------------------------------------------------------------------

def _load_reranker(model_name: str) -> Any:
    """
    Load the cross-encoder model, caching it as a module-level singleton.
    Returns None if sentence-transformers is not installed.
    """
    global _reranker_instance, _reranker_model_name, _warned_unavailable

    if _reranker_instance is not None and _reranker_model_name == model_name:
        return _reranker_instance

    try:
        from sentence_transformers import CrossEncoder  # type: ignore[import]
        logger.info("reranker: loading model %r (first use — may download ~80 MB)", model_name)
        _reranker_instance = CrossEncoder(model_name, max_length=512)
        _reranker_model_name = model_name
        logger.info("reranker: model loaded")
        return _reranker_instance
    except ImportError:
        if not _warned_unavailable:
            logger.info(
                "reranker: sentence-transformers not installed; reranking disabled. "
                "Install with: pip install sentence-transformers"
            )
            _warned_unavailable = True
        return None
    except Exception as exc:
        logger.warning("reranker: failed to load model %r: %s", model_name, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def rerank_candidates(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_k: int = 20,
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> list[dict[str, Any]]:
    """
    Rerank hybrid BM25+vector candidates using a cross-encoder.

    Replaces `final_score` with the cross-encoder's relevance score and
    re-sorts candidates DESC.  The original `bm25_score` and `vector_score`
    are preserved for transparency.

    Args:
        query:      The user's original query string.
        candidates: List of candidate dicts from the hybrid merge step.
                    Each must have a "snippet" key (used as the passage text).
        top_k:      How many candidates to rerank (rerank the top-N by hybrid
                    score, not all candidates — keeps latency bounded).
        model_name: HuggingFace model identifier.

    Returns:
        The same candidate list, reranked and with `final_score` overwritten
        by the cross-encoder score.  If the model is unavailable, the original
        list is returned unchanged.
    """
    if not candidates:
        return candidates

    model = _load_reranker(model_name)
    if model is None:
        return candidates  # graceful degradation

    # Only rerank the top-N by hybrid score (already sorted DESC by merge step)
    to_rerank = candidates[:top_k]
    rest = candidates[top_k:]

    # Build (query, passage) pairs.  Use the snippet rather than the full chunk
    # text since the snippet is already truncated to ~160 chars — enough context
    # for the cross-encoder and keeps the sequence length well within 512 tokens.
    pairs = [(query, c.get("snippet", "") or "") for c in to_rerank]

    try:
        import time as _time
        t0 = _time.perf_counter()
        scores = model.predict(pairs, show_progress_bar=False)
        elapsed_ms = round((_time.perf_counter() - t0) * 1000, 1)
        logger.debug("reranker: scored %d pairs in %s ms", len(pairs), elapsed_ms)
    except Exception as exc:
        logger.warning("reranker: predict failed (returning hybrid order): %s", exc)
        return candidates

    # Overwrite final_score with cross-encoder score; keep originals
    reranked = []
    for cand, score in zip(to_rerank, scores):
        cand = dict(cand)  # copy — don't mutate the original
        cand["final_score"] = float(score)
        cand["reranker_score"] = float(score)
        reranked.append(cand)

    # Sort reranked candidates DESC by cross-encoder score
    reranked.sort(key=lambda r: (-r["final_score"], r.get("file_path", "")))

    return reranked + rest
