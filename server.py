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

  Pipeline
  POST /api/pipeline/discover|extract|ocr|index|full|rebuild|pause|cancel|re-embed
  POST /api/pipeline/reindex-dir
  GET  /api/pipeline/status

  Directories
  GET  /api/directories
  POST /api/directories/add|toggle/{id}|remove/{id}

  Index
  GET  /api/index/file           — full file journey
  POST /api/index/reindex-file|remove-file|deduplicate|integrity
  GET  /api/index/embedding-health

  Diagnostics
  POST /api/diagnostics/run
  GET  /api/processes
  POST /api/processes/restart|kill

  Workers & Auto-Brain
  GET  /api/autobrain/status
  POST /api/autobrain/toggle|override|limits

  Feedback
  POST /api/feedback
  GET  /api/feedback/summary|queries|file/{id}|zero-results|export

  Config
  GET  /api/config/history
  POST /api/config/rollback

  Audit
  GET  /api/audit|audit/export
  POST /api/audit/log

CORS: allow all origins (LAN internal use only).
"""

from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import socket
import sqlite3
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import cfg, ollama_available
from core.compose import ComposeResult, ResultView, compose_answer
from core.db import create_session, get_connection, get_stats
from core.index import run_index
from core.query import (
    search, execute_structured_query,
    execute_file_locator_query, execute_cross_project_query,
)
from core.router import get_router

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
# Module-level pipeline state (thread-safe via lock)
# ---------------------------------------------------------------------------

_pipeline_lock = threading.Lock()
_pipeline_state: dict[str, Any] = {
    "running":    False,
    "stage":      "",
    "processed":  0,
    "total":      0,
    "eta":        None,
    "throughput": 0.0,
    "errors":     [],
    "output":     [],   # live log lines
    "paused":     False,
    "cancelled":  False,
    "started_at": None,
}

_LEGACY_INDEXING = False   # kept for /api/index backward compat


# ---------------------------------------------------------------------------
# Module-level auto-brain state
# ---------------------------------------------------------------------------

_autobrain_state: dict[str, Any] = {
    "enabled": False,
    "allocations": {
        "discover": 2,
        "extract":  2,
        "ocr":      2,
        "embed":    2,
        "index":    2,
    },
    "limits": {"min": 1, "max": 16},
    "decision_log": [],  # [{ts, message}]
    "overrides": {},     # {stage: count}
    "escalations": [],   # [{worker, issue, ts}]
}


# ---------------------------------------------------------------------------
# Directories registry (persisted to JSON sidecar)
# ---------------------------------------------------------------------------

_DIRS_FILE = cfg.work_dir / "directories.json"


def _load_directories() -> list[dict]:
    if _DIRS_FILE.exists():
        try:
            return json.loads(_DIRS_FILE.read_text(encoding="utf-8"))
        except Exception:
            pass
    # Seed from config.yaml index_roots
    return [
        {
            "id": str(i),
            "path": str(root),
            "enabled": True,
            "status": "active",
        }
        for i, root in enumerate(cfg.index_roots)
    ]


def _save_directories(dirs: list[dict]) -> None:
    _DIRS_FILE.write_text(json.dumps(dirs, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Config history helpers
# ---------------------------------------------------------------------------

_CONFIG_HISTORY_DIR = cfg.work_dir / "config_history"


def _snapshot_config() -> str:
    """Save current config.yaml snapshot. Returns version_id."""
    _CONFIG_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    version_id = f"v_{ts}_{uuid.uuid4().hex[:6]}"
    dst = _CONFIG_HISTORY_DIR / f"{version_id}.yaml"
    if cfg.config_file.exists():
        shutil.copy2(cfg.config_file, dst)
    return version_id


def _list_config_history() -> list[dict]:
    if not _CONFIG_HISTORY_DIR.exists():
        return []
    entries = []
    for f in sorted(_CONFIG_HISTORY_DIR.glob("*.yaml"), reverse=True):
        entries.append({
            "version_id": f.stem,
            "ts":         datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
            "size_bytes": f.stat().st_size,
        })
    return entries


# ---------------------------------------------------------------------------
# Pipeline runner helper
# ---------------------------------------------------------------------------

def _pipeline_run(fn_name: str, kwargs: dict | None = None) -> None:
    """Run a pipeline function in background, updating _pipeline_state."""
    global _pipeline_state
    kwargs = kwargs or {}

    with _pipeline_lock:
        _pipeline_state.update({
            "running":    True,
            "stage":      fn_name,
            "processed":  0,
            "total":      0,
            "eta":        None,
            "throughput": 0.0,
            "errors":     [],
            "output":     [f"[{datetime.now().isoformat()}] Starting {fn_name}…"],
            "paused":     False,
            "cancelled":  False,
            "started_at": time.time(),
        })

    try:
        conn = get_connection(cfg.get_db_path())
        try:
            if fn_name == "discover":
                from core.discover import run_discover
                stats = run_discover(conn, cfg.index_roots, cfg)
                _ps_append(f"Discover complete: {stats}")
            elif fn_name == "extract":
                from core.extract import run_extract
                from pathlib import Path as _P
                discovered = conn.execute(
                    "SELECT file_id, file_path, lane FROM files WHERE status='DISCOVERED'"
                ).fetchall()
                for row in discovered:
                    if _pipeline_state.get("cancelled"):
                        break
                    run_extract(conn, row["file_id"], _P(row["file_path"]),
                                row["lane"] or "METADATA_ONLY", cfg)
                    with _pipeline_lock:
                        _pipeline_state["processed"] += 1
                _ps_append("Extract complete")
            elif fn_name == "ocr":
                _ps_append("OCR not implemented — skipped")
            elif fn_name == "index":
                from core.index import run_index as _run_index
                stats = _run_index(conn, cfg)
                _ps_append(f"Index complete: {stats}")
            elif fn_name == "full":
                from core.index import run_full_pipeline
                stats = run_full_pipeline(conn, cfg)
                _ps_append(f"Full pipeline complete: {stats}")
            elif fn_name == "rebuild":
                from core.index import run_rebuild
                stats = run_rebuild(conn, cfg)
                _ps_append(f"Rebuild complete: {stats}")
            elif fn_name == "re-embed":
                # Reset embedded=0 for all chunks, then re-run embed+fts
                conn.execute("UPDATE chunks SET embedded=0")
                conn.execute(
                    "UPDATE files SET status='EXTRACTED' WHERE status IN ('INDEXED','EMBEDDED')"
                )
                conn.commit()
                from core.index import run_index as _run_index
                stats = _run_index(conn, cfg)
                _ps_append(f"Re-embed complete: {stats}")
            elif fn_name == "reindex-dir":
                path = kwargs.get("path", "")
                # Mark files under this path as EXTRACTED so they get re-embedded
                conn.execute(
                    "UPDATE files SET status='EXTRACTED' "
                    "WHERE status='INDEXED' AND file_path LIKE ?",
                    (f"{path}%",),
                )
                conn.commit()
                from core.index import run_index as _run_index
                stats = _run_index(conn, cfg)
                _ps_append(f"Reindex-dir complete: {stats}")
        finally:
            conn.close()
    except Exception as e:
        with _pipeline_lock:
            _pipeline_state["errors"].append(str(e))
        _ps_append(f"ERROR: {e}")
        logger.error("Pipeline %s failed: %s", fn_name, e)
    finally:
        with _pipeline_lock:
            _pipeline_state["running"] = False
            _pipeline_state["stage"] = ""


def _ps_append(line: str) -> None:
    with _pipeline_lock:
        _pipeline_state["output"].append(f"[{datetime.now().isoformat()}] {line}")


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------

def _audit(action: str, detail: str | None = None) -> None:
    """Write an audit log entry (fire-and-forget, non-blocking)."""
    try:
        conn = get_connection(cfg.get_db_path())
        conn.execute(
            "INSERT INTO audit_log (actor, action, detail) VALUES ('admin', ?, ?)",
            (action, detail),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("Audit log write failed: %s", e)


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

app = FastAPI(title="TIGA Hunt", version="0.3.0", lifespan=lifespan)

# Serve static assets (CSS, JS if ever split out)
_static_dir = Path(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def serve_ui() -> HTMLResponse:
    """Serve the main search UI."""
    html_file = Path(__file__).parent / "static" / "index.html"
    return HTMLResponse(html_file.read_text(encoding="utf-8"))


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
    offset: int = 0
    session_id: str | None = None
    filters: dict[str, str] | None = None


class ResultItem(BaseModel):
    title:       str
    rel_path:    str
    file_path:   str = ""   # absolute path for open-in-folder
    citation:    str
    snippet:     str
    project_id:  str
    typology:    str
    ext:         str
    final_score: float


class OpenFileRequest(BaseModel):
    file_path: str


class QueryResponse(BaseModel):
    query:               str
    session_id:          str
    answer_summary:      str
    follow_up_prompts:   list[str]
    confidence:          float
    confidence_label:    str
    mode:                str          # structured | file_locator | semantic | cross_project
    results:             list[ResultItem]
    files:               list[dict] | None = None   # for file_locator mode
    data:                dict | None = None          # for structured mode
    fallback_suggestion: str | None = None
    index_stats:         dict | None = None
    latency_ms:          float


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


class FeedbackRequest(BaseModel):
    query:         str
    result_id:     str | None = None
    rating:        int | None = None   # 1 or -1
    rank_position: int | None = None
    session_id:    str | None = None
    comment:       str | None = None


class AuditLogRequest(BaseModel):
    action: str
    detail: str | None = None


class DirectoryAddRequest(BaseModel):
    path: str


class PipelineDirRequest(BaseModel):
    path: str


class ProcessActionRequest(BaseModel):
    pid: int


class AutobrainToggleRequest(BaseModel):
    enabled: bool


class AutobrainOverrideRequest(BaseModel):
    stage:   str
    workers: int


class AutobrainLimitsRequest(BaseModel):
    min: int
    max: int


class ConfigRollbackRequest(BaseModel):
    version_id: str


class ReindexFileRequest(BaseModel):
    path: str


class RemoveFileRequest(BaseModel):
    path: str


# ---------------------------------------------------------------------------
# Core endpoints (existing)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    ok = ollama_available(cfg.ollama_base_url)
    return {"status": "ok" if ok else "degraded", "ollama": ok}


@app.get("/api/status")
async def api_status(conn: sqlite3.Connection = Depends(get_db)) -> dict[str, Any]:
    stats = get_stats(conn)
    embedded = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE embedded=1"
    ).fetchone()[0]
    last_idx = conn.execute(
        "SELECT MAX(updated_at) FROM files WHERE status='INDEXED'"
    ).fetchone()[0]

    # Per-extension breakdown
    ext_rows = conn.execute(
        "SELECT extension, status, COUNT(*) AS cnt FROM files GROUP BY extension, status"
    ).fetchall()
    by_ext: dict[str, dict] = {}
    for r in ext_rows:
        ext = r["extension"] or "other"
        if ext not in by_ext:
            by_ext[ext] = {}
        by_ext[ext][r["status"]] = r["cnt"]

    # Per-directory breakdown
    dir_rows = conn.execute(
        "SELECT file_path, status FROM files"
    ).fetchall()
    by_dir: dict[str, dict] = {}
    for root in cfg.index_roots:
        root_str = str(root)
        label = root.name
        by_dir[label] = {"root": root_str, "DISCOVERED": 0, "EXTRACTED": 0, "INDEXED": 0, "FAILED": 0}
    for r in dir_rows:
        for root in cfg.index_roots:
            if r["file_path"].startswith(str(root)):
                label = root.name
                st = r["status"]
                if st in by_dir.get(label, {}):
                    by_dir[label][st] = by_dir[label].get(st, 0) + 1

    # Ollama model status
    embed_ok = ollama_available(cfg.ollama_base_url)
    chat_ok  = embed_ok  # same server

    # Disk usage
    try:
        work_usage = shutil.disk_usage(cfg.work_dir)
        disk_free_gb = round(work_usage.free / 1e9, 2)
        disk_used_gb = round(work_usage.used / 1e9, 2)
    except Exception:
        disk_free_gb = disk_used_gb = 0.0

    # Health summary (Chunk 13)
    try:
        from core.index import check_index_staleness
        staleness = check_index_staleness(conn, cfg_obj=cfg)
        staleness_score = staleness.staleness_score
        files_missing = staleness.files_missing
        if staleness_score < 0.2:
            overall_health = "good"
        elif staleness_score < 0.5:
            overall_health = "degraded"
        else:
            overall_health = "stale"
    except Exception:
        staleness_score = 0.0
        files_missing = 0
        overall_health = "unknown"

    # Per-project health
    project_health: list[dict] = []
    try:
        p_rows = conn.execute(
            "SELECT project_code, COUNT(*) AS n FROM files "
            "WHERE status = 'INDEXED' AND project_code IS NOT NULL "
            "GROUP BY project_code"
        ).fetchall()
        for pr in p_rows[:20]:  # cap at 20
            project_health.append({
                "project_code": pr["project_code"],
                "files_indexed": pr["n"],
            })
    except Exception:
        pass

    return {
        "files_discovered": stats.get("DISCOVERED", 0),
        "files_extracted":  stats.get("EXTRACTED",  0),
        "files_embedded":   stats.get("EMBEDDED",   0),
        "files_indexed":    stats.get("INDEXED",    0),
        "files_failed":     stats.get("FAILED",     0),
        "files_skipped":    stats.get("SKIPPED",    0),
        "chunks_total":     stats.get("total_chunks", 0),
        "embedded_chunks":  embedded,
        "ollama_available": embed_ok,   # backward compat flat field
        "ollama": {
            "embed": {"model": cfg.embed_model, "ok": embed_ok},
            "chat":  {"model": cfg.chat_model,  "ok": chat_ok},
        },
        "last_indexed_at":  last_idx,
        "by_extension":     by_ext,
        "by_directory":     by_dir,
        "disk_free_gb":     disk_free_gb,
        "disk_used_gb":     disk_used_gb,
        # Chunk 13 — health
        "overall_health":   overall_health,
        "staleness_score":  staleness_score,
        "files_missing":    files_missing,
        "projects":         project_health,
    }


def _aggregate_dirs(views: list) -> list:
    """Add folder pseudo-results when 2+ files from the same directory match."""
    from collections import defaultdict

    dir_groups: dict[str, list] = defaultdict(list)
    for v in views:
        # Normalise separators so we group correctly on Windows too
        parts = v.rel_path.replace("\\", "/").split("/")
        if len(parts) > 1:
            parent = "/".join(parts[:-1])
            dir_groups[parent].append(v)

    dir_views = []
    for dir_path, members in dir_groups.items():
        if len(members) < 2:
            continue
        top_score = max(m.final_score for m in members)
        dir_name  = dir_path.split("/")[-1] or dir_path
        names     = ", ".join((m.title or m.rel_path.split("/")[-1]) for m in members[:3])
        if len(members) > 3:
            names += f" +{len(members) - 3} more"
        dir_views.append(ResultView(
            title       = dir_name + "/",
            rel_path    = dir_path,
            file_path   = str(Path(members[0].file_path).parent) if members[0].file_path else "",
            citation    = dir_path,
            snippet     = f"Folder with {len(members)} matching files — {names}",
            project_id  = members[0].project_id,
            typology    = members[0].typology,
            ext         = "dir",
            final_score = top_score * 0.95,   # just below top individual file
        ))
    return dir_views


def _confidence_label(conf: float) -> str:
    if conf >= 0.8:
        return "High"
    if conf >= 0.55:
        return "Medium"
    if conf > 0.0:
        return "Low"
    return "Not Found"


def _get_fallback_suggestion(mode: str, filters: dict, project_code: str | None) -> str | None:
    """Return a manual fallback path hint when confidence is low or no results."""
    root_path = ""
    if project_code:
        try:
            from core.project_card import get_project_card
            card = get_project_card(project_code)
            root_path = (card or {}).get("root_path", "") or ""
        except Exception:
            pass
    root_hint = f"\\\\server\\{project_code}" if not root_path else root_path

    ct = filters.get("content_type", "")
    if ct in ("BIM", "CAD"):
        return f"Try checking the {ct} folder under: {root_hint}"
    if ct in ("Render Scene", "Image"):
        return f"Try checking: {root_hint}\\Images  or  {root_hint}\\Presentations"
    if ct == "Minutes":
        return f"Meeting minutes are usually in: {root_hint}\\Documents\\Minutes"
    if ct == "Transmittal":
        return f"Transmittals are usually in: {root_hint}\\Outgoing"
    if mode == "file_locator":
        return (
            f"No files matched in the index. The index may be stale — "
            f"run 'python tiga.py index' to update. "
            f"Try checking: {root_hint} directly."
        )
    if mode == "semantic":
        return (
            f"Result confidence is low. Consider searching with different terms "
            f"or browsing: {root_hint} directly."
        )
    return None


def _log_query(
    query: str,
    project_code: str | None,
    mode: str,
    confidence: float,
    result_count: int,
    duration_ms: float,
) -> None:
    """Log query to tiga_work/logs/queries.log (and low_confidence.log if needed)."""
    try:
        log_dir = cfg.work_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        entry = json.dumps({
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "project_code": project_code,
            "mode": mode,
            "confidence": round(confidence, 3),
            "result_count": result_count,
            "duration_ms": round(duration_ms, 1),
        }, ensure_ascii=False)
        with open(log_dir / "queries.log", "a", encoding="utf-8") as f:
            f.write(entry + "\n")
        if confidence < 0.6:
            with open(log_dir / "low_confidence.log", "a", encoding="utf-8") as f:
                f.write(entry + "\n")
    except Exception as e:
        logger.debug("Query log write failed: %s", e)


@app.post("/api/query", response_model=QueryResponse)
async def api_query(
    req: QueryRequest,
    conn: sqlite3.Connection = Depends(get_db),
) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    t_start = time.perf_counter()
    session_id = req.session_id or str(uuid.uuid4())

    # ── Route the query ───────────────────────────────────────────────────────
    router = get_router()
    router.load_project_codes(conn)
    project_code_hint = (req.filters or {}).get("project_id") or (req.filters or {}).get("project_code")
    route = router.classify(req.query, project_code=project_code_hint)

    mode = route.mode
    confidence = 0.5
    files_result: list[dict] | None = None
    data_result: dict | None = None
    answer_summary = ""
    follow_up_prompts: list[str] = []
    results_page: list[ResultView] = []
    fallback_suggestion: str | None = None

    # ── Dispatch to appropriate executor ────────────────────────────────────
    if mode == "structured":
        exec_result = execute_structured_query(route, conn=conn)
        answer_summary = exec_result["answer_text"]
        data_result = exec_result.get("data")
        confidence = exec_result.get("confidence", 0.7)
        follow_up_prompts = []

    elif mode == "file_locator":
        exec_result = execute_file_locator_query(route, conn=conn)
        answer_summary = exec_result["answer_text"]
        files_result = exec_result.get("files", [])
        confidence = exec_result.get("confidence", 0.7)
        follow_up_prompts = []

    elif mode == "cross_project":
        exec_result = execute_cross_project_query(route, conn=conn)
        answer_summary = exec_result["answer_text"]
        data_result = exec_result.get("results_per_project")
        confidence = exec_result.get("confidence", 0.7)
        follow_up_prompts = []

    else:
        # semantic — use existing hybrid RAG pipeline with synonym boosting
        pool_k = max((req.top_k + req.offset) * 3, 20)
        search_filters = req.filters or {}
        if route.project_code:
            search_filters = {**search_filters, "project_id": route.project_code}
        # Pass expanded_terms for BM25/vector boosting (Chunk 7)
        _expanded_terms = (
            route.expanded_query.expanded_terms
            if route.expanded_query and route.expanded_query.expanded_terms
            else None
        )
        sr = search(req.query, top_k=pool_k, filters=search_filters or None, conn=conn,
                    expanded_terms=_expanded_terms)
        cr = compose_answer(req.query, list(sr), session_id=session_id, conn=conn)
        answer_summary = cr.answer_summary
        follow_up_prompts = cr.follow_ups
        confidence = cr.confidence

        # Mix in directory pseudo-results, re-sort by score
        dir_results = _aggregate_dirs(cr.results)
        mixed = sorted(cr.results + dir_results, key=lambda x: x.final_score, reverse=True)
        results_page = mixed[req.offset : req.offset + req.top_k]

    # ── Fallback suggestion for low confidence ───────────────────────────────
    if confidence < 0.6 or (mode == "file_locator" and not files_result):
        fallback_suggestion = _get_fallback_suggestion(
            mode, route.filters, route.project_code
        )

    # ── Index stats ───────────────────────────────────────────────────────────
    try:
        idx_stats_row = conn.execute(
            "SELECT COUNT(*) AS cnt, MAX(updated_at) AS last_at FROM files WHERE status='INDEXED'"
        ).fetchone()
        index_stats = {
            "project_code": route.project_code,
            "last_indexed": idx_stats_row["last_at"],
            "file_count": idx_stats_row["cnt"],
        }
    except Exception:
        index_stats = None

    duration_ms = (time.perf_counter() - t_start) * 1000

    # ── Log query ────────────────────────────────────────────────────────────
    result_count = len(results_page) + (len(files_result) if files_result else 0)
    _log_query(req.query, route.project_code, mode, confidence, result_count, duration_ms)

    return QueryResponse(
        query              = req.query,
        session_id         = session_id,
        answer_summary     = answer_summary,
        follow_up_prompts  = follow_up_prompts,
        confidence         = confidence,
        confidence_label   = _confidence_label(confidence),
        mode               = mode,
        results            = [
            ResultItem(
                title       = v.title,
                rel_path    = v.rel_path,
                file_path   = v.file_path,
                citation    = v.citation,
                snippet     = v.snippet,
                project_id  = v.project_id,
                typology    = v.typology,
                ext         = v.ext,
                final_score = v.final_score,
            )
            for v in results_page
        ],
        files              = files_result,
        data               = data_result,
        fallback_suggestion = fallback_suggestion,
        index_stats        = index_stats,
        latency_ms         = round(duration_ms, 1),
    )


@app.get("/api/projects")
async def api_projects(
    conn: sqlite3.Connection = Depends(get_db),
) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT COALESCE(project_id, 'Unknown') AS project_id, "
        "COUNT(*) AS file_count "
        "FROM files GROUP BY project_id ORDER BY file_count DESC"
    ).fetchall()
    return [{"project_id": r["project_id"], "file_count": r["file_count"]} for r in rows]


@app.get("/api/project/{code}")
async def api_get_project(
    code: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> dict[str, Any]:
    """Return full project data card for a given project code."""
    from core.project_card import get_project_card
    card = get_project_card(code, conn=conn)
    if card is None:
        raise HTTPException(status_code=404, detail=f"Project '{code}' not found")
    return card


class CorrectRequest(BaseModel):
    project_code: str
    field: str
    correct_value: Any
    wrong_value: Any = None
    source_path: str | None = None


@app.post("/api/correct")
async def api_correct(req: CorrectRequest) -> dict[str, Any]:
    """Submit a manual field correction for a project."""
    from core.corrections import save_correction
    _audit("Correction submitted", f"{req.project_code}.{req.field} → {req.correct_value!r}")
    save_correction(
        project_code=req.project_code,
        field=req.field,
        correct_value=req.correct_value,
        wrong_value=req.wrong_value,
        source_path=req.source_path,
    )
    return {"status": "saved", "project_code": req.project_code, "field": req.field}


@app.post("/api/session", response_model=SessionResponse)
async def api_create_session(
    conn: sqlite3.Connection = Depends(get_db),
) -> SessionResponse:
    session_id = str(uuid.uuid4())
    create_session(conn, session_id)
    return SessionResponse(session_id=session_id)


@app.get("/api/session/{session_id}", response_model=SessionHistoryResponse)
async def api_get_session(
    session_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> SessionHistoryResponse:
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
# Legacy index trigger (kept for backward compat)
# ---------------------------------------------------------------------------

@app.post("/api/index")
async def api_index(
    req: IndexRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    global _LEGACY_INDEXING
    if _LEGACY_INDEXING or _pipeline_state["running"]:
        return {"status": "already_running"}

    def _run() -> None:
        global _LEGACY_INDEXING
        _LEGACY_INDEXING = True
        try:
            _pipeline_run("rebuild" if req.force else "index")
        finally:
            _LEGACY_INDEXING = False

    background_tasks.add_task(_run)
    return {"status": "started"}


# ---------------------------------------------------------------------------
# Pipeline endpoints
# ---------------------------------------------------------------------------

def _trigger_pipeline(stage: str, kwargs: dict | None = None) -> dict[str, str]:
    if _pipeline_state["running"]:
        return {"status": "already_running", "stage": _pipeline_state["stage"]}
    t = threading.Thread(target=_pipeline_run, args=(stage, kwargs), daemon=True)
    t.start()
    return {"status": "started", "stage": stage}


@app.post("/api/pipeline/discover")
async def pipeline_discover() -> dict:
    _audit("Triggered: Run Discover")
    return _trigger_pipeline("discover")


@app.post("/api/pipeline/extract")
async def pipeline_extract() -> dict:
    _audit("Triggered: Run Extract")
    return _trigger_pipeline("extract")


@app.post("/api/pipeline/ocr")
async def pipeline_ocr() -> dict:
    _audit("Triggered: Run OCR")
    return _trigger_pipeline("ocr")


@app.post("/api/pipeline/index")
async def pipeline_index() -> dict:
    _audit("Triggered: Run Index")
    return _trigger_pipeline("index")


@app.post("/api/pipeline/full")
async def pipeline_full() -> dict:
    _audit("Triggered: Run Full Pipeline")
    return _trigger_pipeline("full")


@app.post("/api/pipeline/rebuild")
async def pipeline_rebuild() -> dict:
    _audit("Triggered: Rebuild")
    return _trigger_pipeline("rebuild")


@app.post("/api/pipeline/pause")
async def pipeline_pause() -> dict:
    with _pipeline_lock:
        _pipeline_state["paused"] = not _pipeline_state["paused"]
        paused = _pipeline_state["paused"]
    _audit("Pipeline: " + ("Paused" if paused else "Resumed"))
    return {"status": "paused" if paused else "resumed"}


@app.post("/api/pipeline/cancel")
async def pipeline_cancel() -> dict:
    with _pipeline_lock:
        _pipeline_state["cancelled"] = True
    _audit("Pipeline: Cancelled")
    return {"status": "cancelled"}


@app.post("/api/pipeline/re-embed")
async def pipeline_re_embed() -> dict:
    _audit("Triggered: Re-embed All")
    return _trigger_pipeline("re-embed")


@app.post("/api/pipeline/reindex-dir")
async def pipeline_reindex_dir(req: PipelineDirRequest) -> dict:
    _audit("Triggered: Reindex Directory", req.path)
    return _trigger_pipeline("reindex-dir", {"path": req.path})


@app.get("/api/pipeline/status")
async def pipeline_status() -> dict:
    with _pipeline_lock:
        state = dict(_pipeline_state)
    # Compute ETA
    if state["running"] and state["started_at"] and state["total"] > 0 and state["processed"] > 0:
        elapsed = time.time() - state["started_at"]
        rate = state["processed"] / elapsed
        remaining = state["total"] - state["processed"]
        state["eta"] = round(remaining / rate) if rate > 0 else None
        state["throughput"] = round(rate, 2)
    return state


# ---------------------------------------------------------------------------
# Directories endpoints
# ---------------------------------------------------------------------------

@app.get("/api/directories")
async def get_directories(conn: sqlite3.Connection = Depends(get_db)) -> list[dict]:
    dirs = _load_directories()
    # Annotate with file counts from DB
    for d in dirs:
        path = d["path"]
        row = conn.execute(
            "SELECT "
            "SUM(CASE WHEN status='DISCOVERED' THEN 1 ELSE 0 END) AS discovered, "
            "SUM(CASE WHEN status='EXTRACTED'  THEN 1 ELSE 0 END) AS extracted, "
            "SUM(CASE WHEN status='INDEXED'    THEN 1 ELSE 0 END) AS indexed, "
            "SUM(CASE WHEN status='FAILED'     THEN 1 ELSE 0 END) AS failed "
            "FROM files WHERE file_path LIKE ?",
            (f"{path}%",),
        ).fetchone()
        d["discovered"] = row["discovered"] or 0
        d["extracted"]  = row["extracted"]  or 0
        d["indexed"]    = row["indexed"]    or 0
        d["failed"]     = row["failed"]     or 0
        d["mounted"]    = Path(path).exists()
    return dirs


@app.post("/api/directories/add")
async def add_directory(req: DirectoryAddRequest) -> dict:
    p = Path(req.path)
    if not p.exists():
        raise HTTPException(status_code=400, detail=f"Path not found or not readable: {req.path}")
    dirs = _load_directories()
    if any(d["path"] == req.path for d in dirs):
        raise HTTPException(status_code=400, detail="Directory already registered")
    new_id = str(uuid.uuid4())[:8]
    dirs.append({"id": new_id, "path": req.path, "enabled": True, "status": "active"})
    _save_directories(dirs)
    _audit("Added index root", req.path)
    return {"status": "added", "id": new_id}


@app.post("/api/directories/toggle/{dir_id}")
async def toggle_directory(dir_id: str) -> dict:
    dirs = _load_directories()
    for d in dirs:
        if d["id"] == dir_id:
            d["enabled"] = not d["enabled"]
            _save_directories(dirs)
            state = "enabled" if d["enabled"] else "disabled"
            _audit(f"Directory {state}", d["path"])
            return {"status": state, "id": dir_id}
    raise HTTPException(status_code=404, detail="Directory not found")


@app.post("/api/directories/remove/{dir_id}")
async def remove_directory(dir_id: str) -> dict:
    dirs = _load_directories()
    before = len(dirs)
    removed_path = next((d["path"] for d in dirs if d["id"] == dir_id), None)
    dirs = [d for d in dirs if d["id"] != dir_id]
    if len(dirs) == before:
        raise HTTPException(status_code=404, detail="Directory not found")
    _save_directories(dirs)
    _audit("Removed index root", removed_path)
    return {"status": "removed", "id": dir_id}


# ---------------------------------------------------------------------------
# Index endpoints
# ---------------------------------------------------------------------------

@app.get("/api/index/file")
async def index_file_search(
    q: str = Query(..., description="Filename or path search term"),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict:
    rows = conn.execute(
        "SELECT * FROM files WHERE file_name LIKE ? OR file_path LIKE ? LIMIT 10",
        (f"%{q}%", f"%{q}%"),
    ).fetchall()
    if not rows:
        return {"results": []}
    results = []
    for r in rows:
        file_id = r["file_id"]
        chunk_count = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE file_id=?", (file_id,)
        ).fetchone()[0]
        fb = conn.execute(
            "SELECT SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END) AS pos, "
            "SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END) AS neg "
            "FROM feedback WHERE result_id=?", (file_id,)
        ).fetchone()
        results.append({
            "file_id":    file_id,
            "file_name":  r["file_name"],
            "file_path":  r["file_path"],
            "extension":  r["extension"],
            "status":     r["status"],
            "lane":       r["lane"],
            "project_id": r["project_id"],
            "typology":   r["typology"],
            "updated_at": r["updated_at"],
            "chunks":     chunk_count,
            "feedback":   {
                "positive": fb["pos"] or 0,
                "negative": fb["neg"] or 0,
            },
        })
    return {"results": results}


@app.post("/api/index/reindex-file")
async def reindex_file(req: ReindexFileRequest) -> dict:
    _audit("Reindex single file", req.path)
    conn = get_connection(cfg.get_db_path())
    try:
        row = conn.execute(
            "SELECT file_id FROM files WHERE file_path LIKE ?", (f"%{req.path}%",)
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="File not found in index")
        file_id = row["file_id"]
        conn.execute("UPDATE files SET status='DISCOVERED' WHERE file_id=?", (file_id,))
        conn.execute("UPDATE chunks SET embedded=0 WHERE file_id=?", (file_id,))
        conn.commit()
    finally:
        conn.close()
    threading.Thread(target=_pipeline_run, args=("full",), daemon=True).start()
    return {"status": "queued", "path": req.path}


@app.post("/api/index/remove-file")
async def remove_file(
    req: RemoveFileRequest,
    conn: sqlite3.Connection = Depends(get_db),
) -> dict:
    _audit("Remove file from index", req.path)
    row = conn.execute(
        "SELECT file_id FROM files WHERE file_path LIKE ?", (f"%{req.path}%",)
    ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="File not found in index")
    file_id = row["file_id"]
    conn.execute("DELETE FROM chunks WHERE file_id=?", (file_id,))
    conn.execute("DELETE FROM files WHERE file_id=?", (file_id,))
    conn.commit()
    return {"status": "removed", "file_id": file_id}


@app.post("/api/index/deduplicate")
async def deduplicate(conn: sqlite3.Connection = Depends(get_db)) -> dict:
    _audit("Triggered: Deduplicate")
    dupes = conn.execute(
        "SELECT fingerprint_sha256, COUNT(*) AS cnt FROM files "
        "WHERE fingerprint_sha256 IS NOT NULL "
        "GROUP BY fingerprint_sha256 HAVING cnt > 1"
    ).fetchall()
    return {"duplicate_groups": len(dupes), "total_dupes": sum(r["cnt"] - 1 for r in dupes)}


@app.post("/api/index/integrity")
async def integrity_check(conn: sqlite3.Connection = Depends(get_db)) -> dict:
    _audit("Triggered: Integrity Check")
    orphan_chunks = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE file_id NOT IN (SELECT file_id FROM files)"
    ).fetchone()[0]
    failed_files = conn.execute(
        "SELECT COUNT(*) FROM files WHERE status='FAILED'"
    ).fetchone()[0]
    return {
        "orphan_chunks": orphan_chunks,
        "failed_files":  failed_files,
        "ok":            orphan_chunks == 0,
    }


@app.get("/api/index/embedding-health")
async def embedding_health() -> dict:
    try:
        import lancedb as _ldb
        db = _ldb.connect(str(cfg.get_vector_dir()))
        if "tiga_chunks" not in db.table_names():
            return {"status": "no_table", "mismatch": False}
        table = db.open_table("tiga_chunks")
        schema = table.schema
        vec_field = next((f for f in schema if f.name == "vector"), None)
        if vec_field is None:
            return {"status": "no_vector_column", "mismatch": False}
        import pyarrow as pa
        index_dim = vec_field.type.list_size
        current_dim = 768  # nomic-embed-text default
        mismatch = index_dim != current_dim
        return {
            "index_model":   cfg.embed_model,
            "current_model": cfg.embed_model,
            "index_dim":     index_dim,
            "current_dim":   current_dim,
            "mismatch":      mismatch,
            "row_count":     table.count_rows(),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "mismatch": False}


# ---------------------------------------------------------------------------
# Diagnostics endpoints
# ---------------------------------------------------------------------------

@app.post("/api/diagnostics/run")
async def run_diagnostics(conn: sqlite3.Connection = Depends(get_db)) -> dict:
    _audit("Triggered: Run Full Diagnostic")
    checks = []

    def _check(name: str, ok: bool, detail: str, fix: str = "") -> dict:
        return {"name": name, "ok": ok, "detail": detail, "fix": fix}

    # Ollama reachable
    oll_ok = ollama_available(cfg.ollama_base_url)
    checks.append(_check("Ollama reachable", oll_ok,
        f"Checked {cfg.ollama_base_url}",
        "Start Ollama: ollama serve" if not oll_ok else ""))

    # Embed model
    checks.append(_check(f"Embed model ({cfg.embed_model})", oll_ok,
        "Model availability assumes Ollama is running",
        f"Run: ollama pull {cfg.embed_model}" if not oll_ok else ""))

    # Chat model
    checks.append(_check(f"Chat model ({cfg.chat_model})", oll_ok,
        "Model availability assumes Ollama is running",
        f"Run: ollama pull {cfg.chat_model}" if not oll_ok else ""))

    # DB readable
    try:
        conn.execute("SELECT 1")
        db_ok = True
        db_detail = str(cfg.get_db_path())
    except Exception as e:
        db_ok = False
        db_detail = str(e)
    checks.append(_check("Index database readable", db_ok, db_detail,
        "Delete DB and re-run index" if not db_ok else ""))

    # Index roots mounted
    for root in cfg.index_roots:
        mounted = Path(root).exists()
        checks.append(_check(f"Root mounted: {root.name}", mounted,
            str(root),
            "Check network drive / NAS connection" if not mounted else ""))

    # Disk space
    try:
        usage = shutil.disk_usage(cfg.work_dir)
        free_gb = usage.free / 1e9
        low = free_gb < 5.0
        checks.append(_check("Disk space",
            not low,
            f"{free_gb:.1f} GB free",
            "Free up disk space" if low else ""))
    except Exception as e:
        checks.append(_check("Disk space", False, str(e), ""))

    # Embedding dimension health
    eh = await embedding_health()
    mismatch = eh.get("mismatch", False)
    checks.append(_check("Embedding dimension match",
        not mismatch,
        f"index: {eh.get('index_dim','?')} current: {eh.get('current_dim','?')}",
        "Click Re-embed All to rebuild vectors" if mismatch else ""))

    # Orphan chunks
    orphans = conn.execute(
        "SELECT COUNT(*) FROM chunks WHERE file_id NOT IN (SELECT file_id FROM files)"
    ).fetchone()[0]
    checks.append(_check("No orphaned chunks", orphans == 0,
        f"{orphans} orphaned chunks found",
        "Run integrity check + deduplicate" if orphans else ""))

    return {"checks": checks, "passed": sum(1 for c in checks if c["ok"]), "total": len(checks)}


@app.get("/api/processes")
async def get_processes() -> list[dict]:
    """Return live pipeline worker info (stubs real process tracking)."""
    procs = []
    if _pipeline_state["running"]:
        procs.append({
            "pid":     os.getpid(),
            "role":    _pipeline_state["stage"] + "-worker",
            "status":  "active",
            "cpu":     0.0,
            "ram_gb":  0.0,
            "runtime": round(time.time() - (_pipeline_state.get("started_at") or time.time())),
        })
    return procs


@app.post("/api/processes/restart")
async def restart_process(req: ProcessActionRequest) -> dict:
    _audit("Restarted process", f"PID {req.pid}")
    return {"status": "ok", "pid": req.pid}


@app.post("/api/processes/kill")
async def kill_process(req: ProcessActionRequest) -> dict:
    _audit("Killed process", f"PID {req.pid}")
    return {"status": "ok", "pid": req.pid}


# ---------------------------------------------------------------------------
# Auto-brain endpoints
# ---------------------------------------------------------------------------

@app.get("/api/autobrain/status")
async def autobrain_status() -> dict:
    return dict(_autobrain_state)


@app.post("/api/autobrain/toggle")
async def autobrain_toggle(req: AutobrainToggleRequest) -> dict:
    _autobrain_state["enabled"] = req.enabled
    state_str = "enabled" if req.enabled else "disabled"
    _audit(f"Auto-brain {state_str}")
    _autobrain_state["decision_log"].append({
        "ts": datetime.now().isoformat(),
        "message": f"Auto-brain {state_str} by admin",
    })
    return {"enabled": req.enabled}


@app.post("/api/autobrain/override")
async def autobrain_override(req: AutobrainOverrideRequest) -> dict:
    old = _autobrain_state["allocations"].get(req.stage, 0)
    _autobrain_state["allocations"][req.stage] = req.workers
    _autobrain_state["overrides"][req.stage] = req.workers
    _audit(f"Set {req.stage} workers: {old} → {req.workers} (manual override)")
    _autobrain_state["decision_log"].append({
        "ts": datetime.now().isoformat(),
        "message": f"Manual override: {req.stage} workers {old}→{req.workers}",
    })
    return {"stage": req.stage, "workers": req.workers}


@app.post("/api/autobrain/limits")
async def autobrain_limits(req: AutobrainLimitsRequest) -> dict:
    _autobrain_state["limits"] = {"min": req.min, "max": req.max}
    _audit(f"Set worker limits: min={req.min} max={req.max}")
    return {"limits": _autobrain_state["limits"]}


# ---------------------------------------------------------------------------
# Feedback endpoints
# ---------------------------------------------------------------------------

@app.post("/api/feedback")
async def post_feedback(
    req: FeedbackRequest,
    conn: sqlite3.Connection = Depends(get_db),
) -> dict:
    conn.execute(
        "INSERT INTO feedback (session_id, query, result_id, rating, rank_position, comment) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (req.session_id, req.query, req.result_id, req.rating, req.rank_position, req.comment),
    )
    conn.commit()
    return {"status": "ok"}


@app.get("/api/feedback/summary")
async def feedback_summary(
    from_date: str | None = Query(None, alias="from"),
    to_date:   str | None = Query(None, alias="to"),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict:
    where = "1=1"
    params: list = []
    if from_date:
        where += " AND ts >= ?"
        params.append(from_date)
    if to_date:
        where += " AND ts <= ?"
        params.append(to_date)

    row = conn.execute(
        f"SELECT "
        f"SUM(CASE WHEN rating=1  THEN 1 ELSE 0 END) AS pos, "
        f"SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END) AS neg "
        f"FROM feedback WHERE {where}",
        params,
    ).fetchone()
    total = (row["pos"] or 0) + (row["neg"] or 0)
    ratio = round((row["pos"] or 0) / total, 3) if total > 0 else None

    trending_down = conn.execute(
        f"SELECT query, COUNT(*) AS cnt FROM feedback "
        f"WHERE rating=-1 AND {where} "
        f"GROUP BY query ORDER BY cnt DESC LIMIT 10",
        params,
    ).fetchall()

    return {
        "total_positive": row["pos"] or 0,
        "total_negative": row["neg"] or 0,
        "positive_ratio": ratio,
        "trending_down":  [{"query": r["query"], "count": r["cnt"]} for r in trending_down],
    }


@app.get("/api/feedback/queries")
async def feedback_queries(conn: sqlite3.Connection = Depends(get_db)) -> list[dict]:
    rows = conn.execute(
        "SELECT query, "
        "COUNT(*) AS total, "
        "SUM(CASE WHEN rating=1  THEN 1 ELSE 0 END) AS pos, "
        "SUM(CASE WHEN rating=-1 THEN 1 ELSE 0 END) AS neg, "
        "COUNT(comment) AS comments "
        "FROM feedback GROUP BY query ORDER BY total DESC LIMIT 100"
    ).fetchall()
    result = []
    for r in rows:
        flagged = r["total"] > 0 and r["neg"] / r["total"] > 0.5
        result.append({
            "query":    r["query"],
            "results":  r["total"],
            "positive": r["pos"],
            "negative": r["neg"],
            "comments": r["comments"],
            "flagged":  flagged,
        })
    return result


@app.get("/api/feedback/file/{file_id}")
async def feedback_for_file(
    file_id: str,
    conn: sqlite3.Connection = Depends(get_db),
) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM feedback WHERE result_id=? ORDER BY ts DESC",
        (file_id,),
    ).fetchall()
    return [dict(r) for r in rows]


@app.get("/api/feedback/zero-results")
async def feedback_zero_results(conn: sqlite3.Connection = Depends(get_db)) -> list[dict]:
    # Queries in feedback that have no positive rating
    rows = conn.execute(
        "SELECT query, COUNT(*) AS attempts, MAX(ts) AS last_seen "
        "FROM feedback WHERE rating IS NULL OR rating = -1 "
        "GROUP BY query "
        "HAVING SUM(CASE WHEN rating=1 THEN 1 ELSE 0 END) = 0 "
        "ORDER BY attempts DESC LIMIT 50"
    ).fetchall()
    return [{"query": r["query"], "attempts": r["attempts"], "last_seen": r["last_seen"]}
            for r in rows]


@app.get("/api/feedback/export")
async def feedback_export(conn: sqlite3.Connection = Depends(get_db)) -> StreamingResponse:
    rows = conn.execute("SELECT * FROM feedback ORDER BY ts DESC").fetchall()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["feedback_id", "session_id", "query", "result_id",
                     "rating", "rank_position", "comment", "ts"])
    for r in rows:
        writer.writerow([r["feedback_id"], r["session_id"], r["query"], r["result_id"],
                         r["rating"], r["rank_position"], r["comment"], r["ts"]])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tiga_feedback.csv"},
    )


# ---------------------------------------------------------------------------
# Config history endpoints
# ---------------------------------------------------------------------------

@app.get("/api/config/history")
async def config_history() -> list[dict]:
    return _list_config_history()


@app.post("/api/config/rollback")
async def config_rollback(req: ConfigRollbackRequest) -> dict:
    src = _CONFIG_HISTORY_DIR / f"{req.version_id}.yaml"
    if not src.exists():
        raise HTTPException(status_code=404, detail="Version not found")
    _snapshot_config()   # save current before overwriting
    import shutil as _shutil
    _shutil.copy2(src, cfg.config_file)
    _audit("Config rollback", f"→ version {req.version_id}")
    return {"status": "rolled_back", "version_id": req.version_id}


# ---------------------------------------------------------------------------
# Audit endpoints
# ---------------------------------------------------------------------------

@app.get("/api/audit")
async def get_audit_log(
    page:  int = Query(1, ge=1),
    limit: int = Query(50, le=500),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict:
    offset = (page - 1) * limit
    rows = conn.execute(
        "SELECT * FROM audit_log ORDER BY audit_id DESC LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    total = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
    return {
        "total": total,
        "page":  page,
        "limit": limit,
        "items": [dict(r) for r in rows],
    }


@app.get("/api/audit/export")
async def export_audit(conn: sqlite3.Connection = Depends(get_db)) -> StreamingResponse:
    rows = conn.execute("SELECT * FROM audit_log ORDER BY audit_id DESC").fetchall()
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["audit_id", "actor", "action", "detail", "ts"])
    for r in rows:
        writer.writerow([r["audit_id"], r["actor"], r["action"], r["detail"], r["ts"]])
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=tiga_audit.csv"},
    )


@app.get("/api/scan")
async def api_scan_folder(
    path: str = Query(..., description="Absolute path to scan"),
    phases: bool = Query(False, description="Scan sub-folders as projects"),
    depth: int = Query(1, description="Sub-folder depth for --phases mode"),
    top: int = Query(20, description="Top-N file types to return"),
) -> dict[str, Any]:
    """
    Windirstat-style quick scan of a directory.
    Returns file type breakdown + size, and (optionally) phase recommendations.
    """
    from tools.scanner import scan_folder, scan_for_phases

    target = Path(path)
    if not target.exists():
        raise HTTPException(status_code=404, detail=f"Path not found: {path}")

    if phases:
        return scan_for_phases(target, depth=depth)
    else:
        scan = scan_folder(target, top_files=top)
        return scan.to_dict()


@app.post("/api/open-file")
async def open_file_in_folder(req: OpenFileRequest) -> dict:
    """
    Open the containing folder of a file in the OS file explorer.

    Security: only opens paths that exist inside a configured index_root.
    This endpoint is intentionally restricted to the LAN server machine
    (the same machine that holds the files).
    """
    import platform
    import subprocess

    raw = req.file_path.strip()
    if not raw:
        raise HTTPException(status_code=400, detail="file_path is required")

    target = Path(raw)

    # Security: path must exist under one of the configured index roots
    allowed = False
    for root in cfg.index_roots:
        try:
            target.resolve().relative_to(root.resolve())
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        raise HTTPException(status_code=403, detail="Path not within any configured index root")

    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        system = platform.system()
        if system == "Windows":
            # Opens Explorer and selects the file
            subprocess.Popen(["explorer", "/select,", str(target)])
        elif system == "Darwin":
            # Opens Finder and reveals the file
            subprocess.Popen(["open", "-R", str(target)])
        else:
            # Linux: open the parent directory
            subprocess.Popen(["xdg-open", str(target.parent)])
        return {"status": "ok", "path": str(target)}
    except Exception as exc:
        logger.warning("open-file failed for %s: %s", target, exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/audit/log")
async def post_audit_log(req: AuditLogRequest) -> dict:
    _audit(req.action, req.detail)
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server:app",
        host=cfg.server_host,
        port=cfg.server_port,
        workers=1,        # single-worker to preserve in-memory pipeline state
        reload=False,
    )
