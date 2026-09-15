"""
core/index_state.py — Single source of truth for index counts + job progress.

Catalog counts come from the `files` table grouped by `project_id` — the same
grain as `/api/projects` and the Atlas wiki list. Configured `index_roots` are
a separate concept (NAS folders), not a proxy for “how many projects are
indexed”. Path-prefix matching against roots is reported as a secondary view
so operators can see unmatched INDEXED files instead of a silent zero.

Honesty: a query failure returns `counts_known=False` and null counts — never
a fake `indexed_projects=0`.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import sqlite3

logger = logging.getLogger(__name__)

ProgressFn = Callable[[dict[str, Any]], None]

IN_PROGRESS_STATUSES = ("DISCOVERED", "EXTRACTED", "EMBEDDED")
CATALOG_STATUSES = (
    "DISCOVERED",
    "EXTRACTED",
    "EMBEDDED",
    "INDEXED",
    "FAILED",
    "SKIPPED",
)

# No progress callback for this long → job is stalled (large PDFs can take a
# couple of minutes; 3 minutes is the operator-visible stall line).
STALL_SECONDS = 180.0

UNKNOWN_PROJECT = "Unknown"


def normalise_index_path(path: str | Path) -> str:
    """Slash- and case-normalise a stored or configured path for comparison."""
    value = str(path).replace("\\", "/").casefold().rstrip("/")
    if value.startswith("//?/unc/"):
        value = "//" + value[len("//?/unc/"):]
    elif value.startswith("//?/"):
        value = value[len("//?/"):]
    return value


def path_matches_root(file_path: str, root: str | Path) -> bool:
    """True if file_path is the root or a descendant (slash/case insensitive)."""
    path = normalise_index_path(file_path)
    normalised_root = normalise_index_path(root)
    if not normalised_root:
        return False
    return path == normalised_root or path.startswith(normalised_root + "/")


def _project_key(raw: str | None) -> str:
    text = (raw or "").strip()
    return text or UNKNOWN_PROJECT


def _sql_norm(column: str = "file_path") -> str:
    # Match Python normalise_index_path for common Windows/UNC stored forms.
    # char(92) is backslash — SQLite does not treat \ as an escape in strings.
    return (
        f"rtrim(replace(replace(replace(lower({column}), char(92), '/'), "
        f"'//?/unc/', '//'), '//?/', ''), '/')"
    )


def _root_bind(root: str | Path) -> tuple[str, str]:
    norm = normalise_index_path(root)
    return norm, f"{norm}/%"


def classify_project_state(
    files_indexed: int,
    files_in_progress: int,
    files_failed: int,
    files_discovered_total: int,
) -> str:
    """Return configured | indexed | in_progress | failed | empty."""
    if files_in_progress > 0:
        return "in_progress"
    if files_indexed > 0:
        return "indexed"
    if files_failed > 0:
        return "failed"
    if files_discovered_total > 0:
        return "configured"
    return "empty"


def _empty_totals() -> dict[str, int]:
    totals = {f"files_{s.lower()}": 0 for s in CATALOG_STATUSES}
    totals["file_count"] = 0
    totals["files_in_progress"] = 0
    return totals


def _last_errors_by_project(conn: sqlite3.Connection) -> dict[str, dict[str, str]]:
    try:
        rows = conn.execute(
            """
            SELECT
                COALESCE(NULLIF(TRIM(project_id), ''), 'Unknown') AS project_id,
                error_code,
                error_detail,
                updated_at
            FROM files
            WHERE status = 'FAILED'
              AND (error_detail IS NOT NULL OR error_code IS NOT NULL)
            ORDER BY updated_at DESC
            """
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("last-error lookup failed: %s", exc)
        return {}
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        key = _project_key(row["project_id"])
        if key in out:
            continue
        out[key] = {
            "error_code": row["error_code"] or "",
            "error_detail": row["error_detail"] or "",
            "updated_at": row["updated_at"] or "",
        }
    return out


def _root_rows(
    conn: sqlite3.Connection,
    configured_roots: Iterable[str | Path],
) -> tuple[list[dict[str, Any]], int]:
    roots = list(configured_roots)
    if not roots:
        try:
            unmatched = conn.execute(
                "SELECT COUNT(*) AS n FROM files WHERE status = 'INDEXED'"
            ).fetchone()["n"]
        except sqlite3.Error:
            unmatched = 0
        return [], int(unmatched or 0)

    clauses = []
    params: list[str] = []
    for root in roots:
        exact, like = _root_bind(root)
        clauses.append(f"({_sql_norm()} = ? OR {_sql_norm()} LIKE ?)")
        params.extend([exact, like])
    any_root = " OR ".join(clauses)

    unmatched_row = conn.execute(
        f"SELECT COUNT(*) AS n FROM files WHERE status = 'INDEXED' AND NOT ({any_root})",
        params,
    ).fetchone()
    unmatched = int(unmatched_row["n"] or 0)

    out: list[dict[str, Any]] = []
    for root in roots:
        exact, like = _root_bind(root)
        row = conn.execute(
            f"""
            SELECT
                COUNT(*) AS file_count,
                SUM(CASE WHEN status = 'INDEXED' THEN 1 ELSE 0 END) AS files_indexed,
                SUM(CASE WHEN status IN ('DISCOVERED','EXTRACTED','EMBEDDED')
                         THEN 1 ELSE 0 END) AS files_in_progress,
                SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) AS files_failed
            FROM files
            WHERE {_sql_norm()} = ? OR {_sql_norm()} LIKE ?
            """,
            (exact, like),
        ).fetchone()
        file_count = int(row["file_count"] or 0)
        files_indexed = int(row["files_indexed"] or 0)
        files_in_progress = int(row["files_in_progress"] or 0)
        files_failed = int(row["files_failed"] or 0)
        name = Path(str(root)).name or str(root)
        out.append({
            "name": name,
            "root": str(root),
            "file_count": file_count,
            "files_indexed": files_indexed,
            "files_in_progress": files_in_progress,
            "files_failed": files_failed,
            "indexed": files_indexed > 0,
            "state": classify_project_state(
                files_indexed, files_in_progress, files_failed, file_count
            ),
        })
    return out, unmatched


def catalog_index_state(
    conn: sqlite3.Connection,
    configured_roots: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Return catalog (project_id) + configured-root views. Never fakes zeros on error."""
    roots = list(configured_roots or [])
    try:
        agg = conn.execute(
            """
            SELECT
                COALESCE(NULLIF(TRIM(project_id), ''), 'Unknown') AS project_id,
                COUNT(*) AS file_count,
                SUM(CASE WHEN status = 'DISCOVERED' THEN 1 ELSE 0 END) AS files_discovered,
                SUM(CASE WHEN status = 'EXTRACTED'  THEN 1 ELSE 0 END) AS files_extracted,
                SUM(CASE WHEN status = 'EMBEDDED'   THEN 1 ELSE 0 END) AS files_embedded,
                SUM(CASE WHEN status = 'INDEXED'    THEN 1 ELSE 0 END) AS files_indexed,
                SUM(CASE WHEN status = 'FAILED'     THEN 1 ELSE 0 END) AS files_failed,
                SUM(CASE WHEN status = 'SKIPPED'    THEN 1 ELSE 0 END) AS files_skipped,
                MAX(updated_at) AS last_updated_at
            FROM files
            GROUP BY 1
            ORDER BY files_indexed DESC, file_count DESC, project_id
            """
        ).fetchall()
    except sqlite3.Error as exc:
        logger.warning("catalog index query failed: %s", exc)
        return unknown_catalog(roots, error=str(exc))

    errors = _last_errors_by_project(conn)
    totals = _empty_totals()
    projects: list[dict[str, Any]] = []

    for row in agg:
        pid = _project_key(row["project_id"])
        files_indexed = int(row["files_indexed"] or 0)
        files_discovered = int(row["files_discovered"] or 0)
        files_extracted = int(row["files_extracted"] or 0)
        files_embedded = int(row["files_embedded"] or 0)
        files_failed = int(row["files_failed"] or 0)
        files_skipped = int(row["files_skipped"] or 0)
        file_count = int(row["file_count"] or 0)
        files_in_progress = files_discovered + files_extracted + files_embedded
        err = errors.get(pid) or {}
        last_error = err.get("error_detail") or err.get("error_code") or None
        last_error_at = err.get("updated_at") or None
        state = classify_project_state(
            files_indexed, files_in_progress, files_failed, file_count
        )
        projects.append({
            "project_id": pid,
            "name": pid,
            "file_count": file_count,
            "files_indexed": files_indexed,
            "files_discovered": files_discovered,
            "files_extracted": files_extracted,
            "files_embedded": files_embedded,
            "files_failed": files_failed,
            "files_skipped": files_skipped,
            "files_in_progress": files_in_progress,
            "indexed": files_indexed > 0,
            "state": state,
            "last_updated_at": row["last_updated_at"],
            "last_error": last_error,
            "last_error_at": last_error_at,
        })
        totals["file_count"] += file_count
        totals["files_discovered"] += files_discovered
        totals["files_extracted"] += files_extracted
        totals["files_embedded"] += files_embedded
        totals["files_indexed"] += files_indexed
        totals["files_failed"] += files_failed
        totals["files_skipped"] += files_skipped
        totals["files_in_progress"] += files_in_progress

    try:
        root_rows, unmatched = _root_rows(conn, roots)
    except sqlite3.Error as exc:
        logger.warning("root match query failed: %s", exc)
        root_rows, unmatched = [], 0

    indexed_projects = sum(1 for p in projects if p["files_indexed"] > 0)
    in_progress_projects = sum(1 for p in projects if p["state"] == "in_progress")
    failed_projects = sum(1 for p in projects if p["state"] == "failed")

    return {
        "ok": True,
        "honesty": "catalog",
        "counts_known": True,
        "counts_error": None,
        "configured_projects": len(roots),
        "configured_roots": len(roots),
        "catalog_projects": len(projects),
        "indexed_projects": indexed_projects,
        "in_progress_projects": in_progress_projects,
        "failed_projects": failed_projects,
        "unmatched_indexed_files": unmatched,
        "totals": totals,
        "projects": projects,
        "roots": root_rows,
    }


def unknown_catalog(
    configured_roots: Iterable[str | Path] | None = None,
    error: str = "catalog unavailable",
) -> dict[str, Any]:
    """Honest unknown payload — never report a fake zero when the catalog failed."""
    roots = [str(r) for r in (configured_roots or [])]
    return {
        "ok": False,
        "honesty": "unknown",
        "counts_known": False,
        "counts_error": error,
        "configured_projects": len(roots),
        "configured_roots": len(roots),
        "catalog_projects": None,
        "indexed_projects": None,
        "in_progress_projects": None,
        "failed_projects": None,
        "unmatched_indexed_files": None,
        "totals": None,
        "projects": [],
        "roots": [
            {
                "name": Path(r).name or r,
                "root": r,
                "file_count": None,
                "files_indexed": None,
                "files_in_progress": None,
                "files_failed": None,
                "indexed": None,
                "state": "unknown",
            }
            for r in roots
        ],
    }


def status_counts_under_root(
    conn: sqlite3.Connection,
    root: str | Path,
) -> dict[str, int]:
    """Slash/case-insensitive per-status counts for one configured directory."""
    exact, like = _root_bind(root)
    row = conn.execute(
        f"""
        SELECT
            SUM(CASE WHEN status = 'DISCOVERED' THEN 1 ELSE 0 END) AS discovered,
            SUM(CASE WHEN status = 'EXTRACTED'  THEN 1 ELSE 0 END) AS extracted,
            SUM(CASE WHEN status = 'EMBEDDED'   THEN 1 ELSE 0 END) AS embedded,
            SUM(CASE WHEN status = 'INDEXED'    THEN 1 ELSE 0 END) AS indexed,
            SUM(CASE WHEN status = 'FAILED'     THEN 1 ELSE 0 END) AS failed
        FROM files
        WHERE {_sql_norm()} = ? OR {_sql_norm()} LIKE ?
        """,
        (exact, like),
    ).fetchone()
    return {
        "discovered": int(row["discovered"] or 0),
        "extracted": int(row["extracted"] or 0),
        "embedded": int(row["embedded"] or 0),
        "indexed": int(row["indexed"] or 0),
        "failed": int(row["failed"] or 0),
    }


def annotate_job(
    job: dict[str, Any],
    *,
    now: float | None = None,
    stall_seconds: float = STALL_SECONDS,
) -> dict[str, Any]:
    """Add rate / ETA / stalled flags to an in-memory pipeline job dict."""
    now = time.time() if now is None else now
    running = bool(job.get("running"))
    processed = int(job.get("processed") or 0)
    total = int(job.get("total") or 0)
    started_at = job.get("started_at")
    last_progress_at = job.get("last_progress_at") or started_at
    errors = job.get("errors") or []

    job["phase"] = job.get("phase") or job.get("stage") or ""
    if job.get("last_error") is None and errors:
        job["last_error"] = str(errors[-1])
        job.setdefault("last_error_at", None)

    rate = None
    eta = None
    if running and started_at and processed > 0:
        elapsed = max(now - float(started_at), 1e-6)
        rate = processed / elapsed
        job["throughput"] = round(rate, 2)
        remaining = max(total - processed, 0)
        if rate > 0 and total > 0:
            eta = round(remaining / rate)
    job["eta"] = eta
    if rate is None:
        job.setdefault("throughput", 0.0)

    idle = None
    stalled = False
    if running and last_progress_at:
        idle = max(now - float(last_progress_at), 0.0)
        stalled = idle >= stall_seconds
    job["idle_seconds"] = None if idle is None else round(idle)
    job["stalled"] = stalled
    job["last_progress_at"] = last_progress_at
    return job


def honour_control(
    should_cancel: Callable[[], bool] | None,
    is_paused: Callable[[], bool] | None,
    *,
    sleep_s: float = 0.4,
) -> str | None:
    """Pause until resumed; return 'cancelled' if the operator cancelled."""
    if should_cancel and should_cancel():
        return "cancelled"
    while is_paused and is_paused():
        if should_cancel and should_cancel():
            return "cancelled"
        time.sleep(sleep_s)
    return None
