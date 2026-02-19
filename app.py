"""
app.py — TIGA Hunt Streamlit UI.

Simple chat interface with session memory.
Calls the FastAPI server at http://localhost:{server_port}.
Falls back to calling search/compose directly if server not available.
"""

from __future__ import annotations

import time
from pathlib import Path

import httpx
import streamlit as st

from config import cfg

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=cfg.ui_title,
    page_icon="🔍",
    layout="wide",
)

SERVER_URL = f"http://localhost:{cfg.server_port}"


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []   # [{role, content}]

if "results" not in st.session_state:
    st.session_state.results = []    # last search results

if "elapsed_ms" not in st.session_state:
    st.session_state.elapsed_ms = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_server(query: str) -> dict:
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
        if m["role"] in ("user", "assistant")
    ][-cfg.max_session_history:]

    resp = httpx.post(
        f"{SERVER_URL}/query",
        json={"query": query, "top_k": cfg.top_k, "history": history},
        timeout=30.0,
    )
    resp.raise_for_status()
    return resp.json()


def _call_direct(query: str) -> dict:
    """Fallback: call search + compose directly without the server."""
    from core.query import search
    from core.compose import compose

    t0 = time.perf_counter()
    results = search(query)
    history = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ][-cfg.max_session_history:]
    answer = compose(query, results, history=history)
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return {
        "query": query,
        "results": results,
        "answer": answer,
        "elapsed_ms": elapsed_ms,
    }


def _query(query: str) -> dict:
    try:
        return _call_server(query)
    except Exception:
        return _call_direct(query)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("TIGA Hunt")
    st.caption("Archive search — POC")
    st.divider()

    # Health check
    try:
        health = httpx.get(f"{SERVER_URL}/health", timeout=3.0).json()
        if health.get("status") == "ok":
            st.success("Server: online")
        else:
            st.warning("Server: degraded")
        st.caption(f"Model: {health.get('chat_model', '—')}")
    except Exception:
        st.warning("Server offline — running direct mode")

    st.divider()

    # Index status
    try:
        status = httpx.get(f"{SERVER_URL}/status", timeout=3.0).json()
        st.metric("Indexed files", status.get("total_documents", 0))
    except Exception:
        pass

    st.divider()

    # Re-index button
    if st.button("Re-index archive", use_container_width=True):
        try:
            httpx.post(f"{SERVER_URL}/index", json={"force": False}, timeout=5.0)
            st.success("Indexing started in background")
        except Exception:
            st.error("Could not reach server to start indexing.")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.results = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------

st.header(cfg.ui_title, divider="grey")

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask about the archive…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Query + stream answer
    with st.chat_message("assistant"):
        with st.spinner("Searching…"):
            response = _query(prompt)

        answer = response.get("answer", "No answer returned.")
        elapsed = response.get("elapsed_ms", 0)
        results = response.get("results", [])

        st.markdown(answer)

        if elapsed:
            st.caption(f"⏱ {elapsed:.0f} ms")

        # Show source files in expander
        if results:
            with st.expander(f"Sources ({len(results)} files)"):
                for i, r in enumerate(results, 1):
                    file_name = r.get("file_name") or r.get("file_name", "—")
                    project = r.get("project", "—")
                    typology = r.get("typology", "—")
                    score = r.get("combined_score", r.get("score", 0))
                    file_path = r.get("file_path", "")
                    st.markdown(
                        f"**[{i}] {file_name}**  \n"
                        f"`{project}` · {typology} · score: {score:.2f}  \n"
                        f"`{file_path}`"
                    )

    st.session_state.messages.append({"role": "assistant", "content": answer})
    st.session_state.results = results
