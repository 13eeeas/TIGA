"""
server.py — TIGA Hunt FastAPI LAN server.

Endpoints:
  POST /query          — hybrid search + compose answer
  GET  /health         — Ollama + DB liveness check
  GET  /status         — index stats
  POST /index          — trigger re-index (background task)
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import cfg
from core.db import get_connection, get_stats
from core.index import run_index
from core.query import search
from core.compose import compose

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg.ensure_dirs()
    logger.info("TIGA Hunt server starting on %s:%s", cfg.server_host, cfg.server_port)
    yield
    logger.info("TIGA Hunt server shutting down")


app = FastAPI(title="TIGA Hunt", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # LAN only — no public exposure
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    history: list[dict[str, str]] = []


class SearchResult(BaseModel):
    file_path: str
    file_name: str
    project: str
    typology: str
    title: str
    combined_score: float


class QueryResponse(BaseModel):
    query: str
    results: list[SearchResult]
    answer: str
    elapsed_ms: float


class IndexRequest(BaseModel):
    force: bool = False


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    import ollama as ollama_client
    ollama_ok = False
    try:
        ollama_client.list()
        ollama_ok = True
    except Exception:
        pass

    db_ok = False
    try:
        conn = get_connection(cfg.db_path)
        conn.execute("SELECT 1")
        conn.close()
        db_ok = True
    except Exception:
        pass

    status = "ok" if (ollama_ok and db_ok) else "degraded"
    return {
        "status": status,
        "ollama": ollama_ok,
        "db": db_ok,
        "chat_model": cfg.chat_model,
        "embed_model": cfg.embed_model,
    }


@app.get("/status")
async def status() -> dict[str, Any]:
    try:
        conn = get_connection(cfg.db_path)
        stats = get_stats(conn)
        conn.close()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return {
        "scan_dirs": [str(d) for d in cfg.scan_dirs],
        **stats,
    }


@app.post("/query", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest) -> QueryResponse:
    import time
    t0 = time.perf_counter()

    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    results = search(req.query, top_k=req.top_k)
    answer = compose(req.query, results, history=req.history)

    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    return QueryResponse(
        query=req.query,
        results=[
            SearchResult(
                file_path=r["file_path"],
                file_name=r["file_name"],
                project=r["project"],
                typology=r["typology"],
                title=r.get("title", ""),
                combined_score=r["combined_score"],
            )
            for r in results
        ],
        answer=answer,
        elapsed_ms=elapsed_ms,
    )


_indexing_in_progress = False


@app.post("/index")
async def index_endpoint(
    req: IndexRequest, background_tasks: BackgroundTasks
) -> dict[str, str]:
    global _indexing_in_progress
    if _indexing_in_progress:
        return {"status": "already running"}

    def _run():
        global _indexing_in_progress
        _indexing_in_progress = True
        try:
            run_index(force=req.force)
        finally:
            _indexing_in_progress = False

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
        reload=False,
    )
