"""
server.py — TIGA Hunt FastAPI LAN server.

Endpoints
---------
  POST /api/query                — hybrid search + compose answer
  GET  /api/status               — index stats + Ollama availability
  GET  /api/projects             — distinct project_ids + file counts
  POST /api/session              — create new session, returns {session_id}
  GET  /api/session/{session_id} — message history for session
  GET  /health                   — liveness check {status, ollama}
  POST /api/index                — trigger incremental re-index (background)

CORS: allow all origins (LAN internal use only).
"""

from __future__ import annotations

import json
import logging
import socket
import sqlite3
import uuid
from contextlib import asynccontextmanager
from typing import Any, Generator

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import cfg, ollama_available
from core.compose import ComposeResult, ResultView, compose_answer
from core.db import create_session, get_connection, get_stats
from core.index import run_index
from core.query import search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DB dependency
# ---------------------------------------------------------------------------

def get_db() -> Generator[sqlite3.Connection, None, None]:
    """FastAPI dependency: yields one SQLite connection per request."""
    conn = get_connection(cfg.get_db_path())
    try:
        yield conn
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg.ensure_dirs()
    if not ollama_available(cfg.ollama_base_url):
        logger.warning("Ollama not reachable at %s — vector lane will be disabled", cfg.ollama_base_url)
    port = cfg.server_port
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
    except Exception:
        local_ip = "127.0.0.1"
    logger.info("TIGA Hunt server ready on port %s", port)
    logger.info("LAN access: http://%s:%s", local_ip, port)
    yield
    logger.info("TIGA Hunt server shutting down")


# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="TIGA Hunt", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],    # LAN internal — no public exposure
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    session_id: str | None = None
    filters: dict[str, str] | None = None


class ResultItem(BaseModel):
    title:       str
    rel_path:    str
    citation:    str
    snippet:     str
    project_id:  str
    typology:    str
    ext:         str
    final_score: float


class QueryResponse(BaseModel):
    query:          str
    session_id:     str
    answer_summary: str
    follow_ups:     list[str]
    confidence:     float
    results:        list[ResultItem]
    latency_ms:     float


class SessionResponse(BaseModel):
    session_id: str


class MessageItem(BaseModel):
    role:      str
    content:   str
    citations: list[str] | None
    ts:        str


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages:   list[MessageItem]


class ProjectItem(BaseModel):
    project_id: str
    file_count:  int


class IndexRequest(BaseModel):
    force: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    """Liveness check. Used by Streamlit sidebar to show server status."""
    ok = ollama_available(cfg.ollama_base_url)
    return {"status": "ok" if ok else "degraded", "ollama": ok}


@app.get("/api/status")
async def api_status(conn: sqlite3.Connection = Depends(get_db)) -> dict[str, Any]:
    """Index stats: file counts by status, chunk counts, Ollama flag, last indexed."""
    stats = get_stats(conn)
    embedded = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedded=1"
    ).fetchone()[0]
    last_idx = conn.execute(
        "SELECT MAX(updated_at) FROM files WHERE status='INDEXED'"
    ).fetchone()[0]
    return {
        "files_discovered": stats.get("DISCOVERED", 0),
        "files_extracted":  stats.get("EXTRACTED",  0),
        "files_embedded":   stats.get("EMBEDDED",   0),
        "files_indexed":    stats.get("INDEXED",    0),
        "files_skipped":    stats.get("SKIPPED",    0),
        "chunks_total":     stats.get("total_chunks", 0),
        "embedded_chunks":  embedded,
        "ollama_available": ollama_available(cfg.ollama_base_url),
        "last_indexed_at":  last_idx,
    }


@app.post("/api/query", response_model=QueryResponse)
async def api_query(
    req: QueryRequest,
    conn: sqlite3.Connection = Depends(get_db),
) -> QueryResponse:
    """Hybrid BM25 + vector search, then Ollama-compose answer."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    session_id = req.session_id or str(uuid.uuid4())

    sr = search(req.query, top_k=req.top_k, filters=req.filters, conn=conn)
    cr = compose_answer(req.query, list(sr), session_id=session_id, conn=conn)

    return QueryResponse(
        query          = req.query,
        session_id     = session_id,
        answer_summary = cr.answer_summary,
        follow_ups     = cr.follow_ups,
        confidence     = cr.confidence,
        results        = [
            ResultItem(
                title       = v.title,
                rel_path    = v.rel_path,
                citation    = v.citation,
                snippet     = v.snippet,
                project_id  = v.project_id,
                typology    = v.typology,
                ext         = v.ext,
                final_score = v.final_score,
            )
            for v in cr.results
        ],
        latency_ms = cr.latency_ms,
    )


@app.get("/api/projects")
async def api_projects(
    conn: sqlite3.Connection = Depends(get_db),
) -> list[dict[str, Any]]:
    """Distinct project_ids with file counts, sorted by file_count DESC."""
    rows = conn.execute(
        "SELECT COALESCE(project_id, 'Unknown') AS project_id, "
        "COUNT(*) AS file_count "
        "FROM files GROUP BY project_id ORDER BY file_count DESC"
    ).fetchall()
    return [{"project_id": r["project_id"], "file_count": r["file_count"]} for r in rows]


@app.post("/api/session", response_model=SessionResponse)
async def api_create_session(
    conn: sqlite3.Connection = Depends(get_db),
) -> SessionResponse:
    """Create a new conversation session. Returns {session_id}."""
    session_id = str(uuid.uuid4())
    create_session(conn, session_id)
    return SessionResponse(session_id=session_id)


@app.get("/api/session/{session_id}", response_model=SessionHistoryResponse)
async def api_get_session(
    session_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> SessionHistoryResponse:
    """Return full message history for a session."""
    rows = conn.execute(
        "SELECT role, content, citations, ts FROM messages "
        "WHERE session_id=? ORDER BY message_id",
        (session_id,),
    ).fetchall()
    messages = [
        MessageItem(
            role      = r["role"],
            content   = r["content"],
            citations = json.loads(r["citations"]) if r["citations"] else None,
            ts        = r["ts"],
        )
        for r in rows
    ]
    return SessionHistoryResponse(session_id=session_id, messages=messages)


# ---------------------------------------------------------------------------
# Background index trigger
# ---------------------------------------------------------------------------

_indexing = False


@app.post("/api/index")
async def api_index(
    req: IndexRequest,
    background_tasks: BackgroundTasks,
    conn: sqlite3.Connection = Depends(get_db),
) -> dict[str, str]:
    """Trigger incremental (or forced) re-index in background."""
    global _indexing
    if _indexing:
        return {"status": "already_running"}

    def _run() -> None:
        global _indexing
        _indexing = True
        try:
            _conn = get_connection(cfg.get_db_path())
            try:
                if req.force:
                    from core.index import run_rebuild
                    run_rebuild(_conn, cfg_obj=cfg)
                else:
                    run_index(_conn, cfg_obj=cfg)
            finally:
                _conn.close()
        finally:
            _indexing = False

    background_tasks.add_task(_run)
    return {"status": "started"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=cfg.server_host,
        port=cfg.server_port,
        workers=cfg.server_workers,
        reload=False,
    )
