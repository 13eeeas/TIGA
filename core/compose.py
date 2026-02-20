"""
core/compose.py — Answer composer.

ResultView  — lightweight view of a SearchResult for rendering.
ComposeResult — structured return type from compose_answer().

compose_answer(query, results, session_id, cfg_obj, conn)
  1. Build context from top-3 result snippets.
  2. Load session history (last 6 turns) from DB.
  3. Call Ollama /api/chat (stream=false) for answer.
  4. Call Ollama for 3 follow-up prompts (separate call).
  5. Compute confidence from result scores.
  6. Persist: user query + assistant answer + citations to messages.
  7. Log latency to events table.

Fallback (Ollama unavailable or times out):
  answer_summary = "[Ollama unavailable — showing raw excerpts] " + top-3 snippets[:600]
  follow_ups     = []

Hard rule:
  Citations come from results list only.  Never from Ollama output.
  answer_summary is the LLM synthesis.
  results list is the ground truth with citations.

Streaming API (legacy / CLI):
  compose_stream(query, results, history) — yield tokens
  compose(query, results, history)        — return full string
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generator

from config import cfg as _module_cfg, Config
from core.db import (
    add_message,
    create_session,
    get_connection,
    log_event,
    touch_session,
)

logger = logging.getLogger(__name__)

_CONTEXT_RESULTS = 3       # top-N snippets included in Ollama context
_SESSION_HISTORY = 6       # prior turns loaded from DB
_FALLBACK_MAX    = 600     # max chars for raw-excerpt fallback
_FALLBACK_PREFIX = "[Ollama unavailable — showing raw excerpts] "

_SYSTEM_PROMPT = (
    "You are TIGA Hunt, an architecture firm research assistant. "
    "Answer using ONLY the provided context. "
    "If the context does not contain the answer, say so. "
    "Do not invent facts. Cite sources by filename."
)

_FOLLOWUP_PROMPT = (
    "Suggest exactly 3 short follow-up questions the user might ask next, "
    "based on the query and context above. "
    "Output one question per line, no numbering, no bullet points."
)


# ---------------------------------------------------------------------------
# ResultView
# ---------------------------------------------------------------------------

@dataclass
class ResultView:
    title:       str    # filename stem
    rel_path:    str
    citation:    str
    snippet:     str
    project_id:  str
    typology:    str
    ext:         str
    final_score: float

    @classmethod
    def from_search_result(cls, r: dict[str, Any]) -> "ResultView":
        rel = r.get("rel_path", "")
        fn  = r.get("file_name", rel)
        return cls(
            title       = Path(fn).stem if fn else "",
            rel_path    = rel,
            citation    = r.get("citation", ""),
            snippet     = r.get("snippet", ""),
            project_id  = r.get("project_id", "Unknown"),
            typology    = r.get("typology", "Unknown"),
            ext         = Path(rel).suffix.lower() if rel else "",
            final_score = float(r.get("final_score", 0.0)),
        )


# ---------------------------------------------------------------------------
# ComposeResult
# ---------------------------------------------------------------------------

@dataclass
class ComposeResult:
    answer_summary: str
    follow_ups:     list[str]
    confidence:     float
    results:        list[ResultView]
    latency_ms:     float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ollama_chat_http(
    messages: list[dict[str, str]],
    cfg_obj: Config,
) -> str:
    """POST to Ollama /api/chat (stream=false). Returns content string."""
    url = cfg_obj.ollama_base_url.rstrip("/") + "/api/chat"
    payload = json.dumps({
        "model":    cfg_obj.chat_model,
        "messages": messages,
        "stream":   False,
        "options":  {"num_ctx": cfg_obj.num_ctx},
    }).encode()
    req = urllib.request.Request(
        url, data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=cfg_obj.ollama_timeout) as resp:
        data = json.loads(resp.read())
    return data["message"]["content"]


def _build_context(views: list[ResultView]) -> str:
    if not views:
        return "No relevant documents found."
    parts = []
    for i, v in enumerate(views[:_CONTEXT_RESULTS], 1):
        parts.append(f"[{i}] {v.citation}\n{v.snippet}")
    return "\n\n".join(parts)


def _confidence(views: list[ResultView]) -> float:
    """Mean final_score, min-max normalised to [0,1]. Returns 0.0 for empty list."""
    if not views:
        return 0.0
    scores = [v.final_score for v in views]
    lo, hi = min(scores), max(scores)
    mean = sum(scores) / len(scores)
    if hi == lo:
        return mean   # all same; already in [0,1]
    return (mean - lo) / (hi - lo)


def _load_history(conn: sqlite3.Connection, session_id: str) -> list[dict[str, str]]:
    try:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id=? ORDER BY message_id DESC LIMIT ?",
            (session_id, _SESSION_HISTORY),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    except Exception as e:
        logger.warning("Failed to load session history: %s", e)
        return []


def _call_followups(query: str, context: str, cfg_obj: Config) -> list[str]:
    """Ask Ollama for 3 short follow-up questions. Returns [] on any failure."""
    try:
        content = _ollama_chat_http(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": (
                    f"Context:\n{context}\n\nQuery: {query}\n\n{_FOLLOWUP_PROMPT}"
                )},
            ],
            cfg_obj,
        )
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        return lines[:3]
    except Exception as e:
        logger.warning("Follow-up generation failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Primary public API
# ---------------------------------------------------------------------------

def compose_answer(
    query: str,
    results: list[dict[str, Any]],
    session_id: str | None = None,
    cfg_obj: Config | None = None,
    conn: sqlite3.Connection | None = None,
) -> ComposeResult:
    """
    Full answer-generation pipeline.

    Args:
        query:      Natural-language question.
        results:    list[SearchResult] from core.query.search().
        session_id: If set, history is loaded and messages are persisted.
        cfg_obj:    Config override (for tests).
        conn:       SQLite connection override (for tests).

    Returns:
        ComposeResult — answer, follow-ups, confidence, views, latency.

    Hard rule: citations come from results list only, never from Ollama output.
    """
    _cfg = cfg_obj or _module_cfg
    _own_conn = conn is None
    _conn = conn or get_connection(_cfg.get_db_path())
    t0 = time.perf_counter()

    try:
        views = [ResultView.from_search_result(r) for r in results]
        context = _build_context(views)

        # Session setup + history
        history: list[dict[str, str]] = []
        if session_id:
            try:
                create_session(_conn, session_id)
                touch_session(_conn, session_id)
                history = _load_history(_conn, session_id)
            except Exception as e:
                logger.warning("Session setup failed: %s", e)

        # Build Ollama messages
        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ]

        # Step 3 — Call Ollama for answer
        ollama_ok = False
        try:
            answer_summary = _ollama_chat_http(messages, _cfg)
            ollama_ok = True
        except Exception as e:
            logger.warning("Ollama unavailable, using fallback: %s", e)
            raw = " … ".join(v.snippet for v in views[:_CONTEXT_RESULTS])
            answer_summary = _FALLBACK_PREFIX + raw[:_FALLBACK_MAX]

        # Step 4 — Follow-up prompts
        follow_ups = _call_followups(query, context, _cfg) if ollama_ok else []

        # Step 5 — Confidence
        confidence = _confidence(views)

        # Step 6 — Session persistence
        if session_id:
            try:
                add_message(_conn, session_id, "user", query)
                citations = [v.citation for v in views]
                add_message(_conn, session_id, "assistant", answer_summary, citations)
                touch_session(_conn, session_id)
            except Exception as e:
                logger.warning("Session persistence failed: %s", e)

        # Latency log
        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            log_event(_conn, "compose_answer", detail=f"{latency_ms:.0f}ms")
        except Exception as e:
            logger.warning("Latency log failed: %s", e)

        return ComposeResult(
            answer_summary = answer_summary,
            follow_ups     = follow_ups,
            confidence     = confidence,
            results        = views,
            latency_ms     = latency_ms,
        )

    finally:
        if _own_conn:
            _conn.close()


# ---------------------------------------------------------------------------
# Streaming API (legacy — used by tiga.py cmd_query)
# ---------------------------------------------------------------------------

def _build_context_block(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No relevant documents found in the archive."
    lines = ["Retrieved documents (ranked by relevance):\n"]
    for i, r in enumerate(results, 1):
        file_name = r.get("file_name")  or r.get("file_path", "")
        project   = r.get("project_id") or r.get("project", "Unknown")
        typology  = r.get("typology",   "Unknown")
        preview   = r.get("snippet")    or r.get("surrogate", "")
        citation  = r.get("citation")   or r.get("file_path", "")
        lines.append(
            f"[{i}] {file_name}\n"
            f"    Project: {project} | Type: {typology}\n"
            f"    Cite as: {citation}\n"
            f"    Preview: {preview[:200]}\n"
        )
    return "\n".join(lines)


def compose_stream(
    query: str,
    results: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> Generator[str, None, None]:
    """
    Yield answer tokens from Ollama as they arrive (streaming).
    history: list of {role, content} dicts for session memory.
    """
    import ollama as ollama_client

    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-(_SESSION_HISTORY):])
    messages.append({"role": "user", "content": _build_context_block(results)
                     + f"\n\n---\nQuestion: {query}"})

    try:
        stream = ollama_client.chat(
            model=_module_cfg.chat_model,
            messages=messages,
            stream=True,
            options={"num_ctx": _module_cfg.num_ctx},
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
    except Exception as e:
        logger.error("Ollama compose failed: %s", e)
        yield f"\n\n[Error: Could not reach Ollama. Is it running? {e}]"


def compose(
    query: str,
    results: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> str:
    """Non-streaming version. Returns full answer string."""
    return "".join(compose_stream(query, results, history))
