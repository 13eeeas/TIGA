"""Index progress and ETA estimates derived from database transition events."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Iterable


_ACTIVE_STATUSES = ("DISCOVERED", "EXTRACTED", "EMBEDDED")


def _normalise(path: str) -> str:
    normal = path.replace("/", "\\")
    if normal.startswith("\\\\?\\"):
        normal = normal[4:]
    return normal.rstrip("\\").casefold()


def _format_seconds(seconds: float | None) -> str | None:
    if seconds is None:
        return None
    if seconds <= 0:
        return "0m"
    minutes = max(1, round(seconds / 60))
    if minutes < 60:
        return f"{minutes}m"
    hours, mins = divmod(minutes, 60)
    if hours < 48:
        return f"{hours}h {mins}m" if mins else f"{hours}h"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}h" if hours else f"{days}d"


def get_index_progress(
    conn: sqlite3.Connection,
    index_roots: Iterable[Path],
    *,
    window_minutes: int = 60,
) -> dict[str, Any]:
    """Return overall and per-root queues, recent rates, and conservative ETAs.

    Rates use transition events observed during the full lookback window. This
    intentionally avoids over-promising from a short burst of fast files.
    """
    roots = []
    for root in index_roots:
        prefix = _normalise(str(root))
        aliases = [prefix]
        shared_at = prefix.find("shared\\")
        if shared_at >= 0:
            aliases.append(prefix[shared_at:])
        roots.append((root.name, str(root), aliases))
    projects: dict[str, dict[str, Any]] = {
        label: {
            "project": label,
            "root": root,
            "counts": {status: 0 for status in (
                "DISCOVERED", "EXTRACTED", "EMBEDDED", "INDEXED", "FAILED", "SKIPPED"
            )},
            "events": {"EXTRACTED": 0, "INDEXED": 0},
        }
        for label, root, _ in roots
    }

    def project_for(path: str) -> dict[str, Any] | None:
        normal = _normalise(path)
        for label, _root, aliases in roots:
            if any(
                normal == prefix
                or normal.startswith(prefix + "\\")
                or ("\\" + prefix + "\\") in ("\\" + normal + "\\")
                for prefix in aliases
            ):
                return projects[label]
        return None

    for row in conn.execute("SELECT file_path, status FROM files"):
        project = project_for(row["file_path"])
        if project is not None:
            status = row["status"]
            project["counts"][status] = project["counts"].get(status, 0) + 1

    modifier = f"-{max(1, int(window_minutes))} minutes"
    event_rows = conn.execute(
        "SELECT f.file_path, e.event_type FROM events e "
        "JOIN files f ON f.file_id=e.file_id "
        "WHERE e.event_type IN ('EXTRACTED','INDEXED') AND e.ts >= datetime('now', ?)",
        (modifier,),
    ).fetchall()
    for row in event_rows:
        project = project_for(row["file_path"])
        if project is not None:
            project["events"][row["event_type"]] += 1

    window_hours = max(1, int(window_minutes)) / 60

    def finalise(item: dict[str, Any]) -> dict[str, Any]:
        counts = item.pop("counts")
        events = item.pop("events")
        extract_rate = round(events["EXTRACTED"] / window_hours, 2)
        index_rate = round(events["INDEXED"] / window_hours, 2)
        extraction_left = counts.get("DISCOVERED", 0)
        indexing_left = sum(counts.get(status, 0) for status in _ACTIVE_STATUSES)
        extraction_seconds = (
            extraction_left / extract_rate * 3600 if extraction_left and extract_rate else 0.0
        )
        indexing_seconds = (
            indexing_left / index_rate * 3600 if indexing_left and index_rate else 0.0
        )
        eta_seconds: float | None
        if (extraction_left and not extract_rate) or (indexing_left and not index_rate):
            eta_seconds = None
        else:
            eta_seconds = extraction_seconds + indexing_seconds
        if indexing_left == 0:
            state = "complete"
            eta_seconds = None
        elif not extract_rate and not index_rate:
            state = "stalled"
        else:
            state = "running"
        item.update({
            **{status.lower(): counts.get(status, 0) for status in counts},
            "files_total": sum(counts.values()),
            "files_remaining": indexing_left,
            "extract_rate_files_per_hour": extract_rate,
            "index_rate_files_per_hour": index_rate,
            "eta_seconds": round(eta_seconds) if eta_seconds is not None else None,
            "eta": _format_seconds(eta_seconds),
            "state": state,
        })
        return item

    aggregate = {
        "project": "All projects",
        "root": None,
        "counts": {status: sum(p["counts"].get(status, 0) for p in projects.values()) for status in (
            "DISCOVERED", "EXTRACTED", "EMBEDDED", "INDEXED", "FAILED", "SKIPPED"
        )},
        "events": {status: sum(p["events"][status] for p in projects.values()) for status in ("EXTRACTED", "INDEXED")},
    }
    result_projects = [finalise(item) for item in projects.values()]
    return {
        "window_minutes": window_minutes,
        "overall": finalise(aggregate),
        "projects": result_projects,
    }
