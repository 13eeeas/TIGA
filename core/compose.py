"""
core/compose.py — Answer composer.

Takes query + search results, calls Ollama/mistral to produce:
  - A cited summary answering the query
  - 3 follow-up prompt suggestions
"""

from __future__ import annotations

import logging
from typing import Any, Generator

import ollama as ollama_client

from config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def _build_system_prompt() -> str:
    return (
        "You are TIGA Hunt, a research assistant for an architecture firm. "
        "You help staff find relevant documents in the firm's project archive. "
        "Be concise and precise. Always cite which files you're drawing from. "
        "If the retrieved documents don't answer the question well, say so clearly "
        "rather than fabricating information."
    )


def _build_context_block(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No relevant documents found in the archive."

    lines = ["Retrieved documents (ranked by relevance):\n"]
    for i, r in enumerate(results, 1):
        lines.append(
            f"[{i}] {r['file_name']}\n"
            f"    Project: {r['project']} | Type: {r['typology']}\n"
            f"    Path: {r['file_path']}\n"
            f"    Preview: {r.get('surrogate', '')[:200]}\n"
        )
    return "\n".join(lines)


def _build_user_prompt(query: str, results: list[dict[str, Any]]) -> str:
    context = _build_context_block(results)
    return (
        f"{context}\n\n"
        f"---\n"
        f"Question: {query}\n\n"
        f"Instructions:\n"
        f"1. Answer the question based on the retrieved documents. "
        f"Cite sources using [1], [2], etc.\n"
        f"2. After your answer, add a blank line then list exactly 3 follow-up "
        f"questions the user might want to ask next, formatted as:\n"
        f"Follow-up prompts:\n- ...\n- ...\n- ..."
    )


# ---------------------------------------------------------------------------
# Compose (streaming)
# ---------------------------------------------------------------------------

def compose_stream(
    query: str,
    results: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> Generator[str, None, None]:
    """
    Yield answer tokens from Ollama as they arrive.
    history: list of {role, content} dicts for session memory.
    """
    messages: list[dict[str, str]] = [
        {"role": "system", "content": _build_system_prompt()}
    ]

    # Inject session history (capped)
    if history:
        messages.extend(history[-(cfg.max_session_history):])

    messages.append(
        {"role": "user", "content": _build_user_prompt(query, results)}
    )

    try:
        stream = ollama_client.chat(
            model=cfg.chat_model,
            messages=messages,
            stream=True,
            options={"num_ctx": 4096},
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
