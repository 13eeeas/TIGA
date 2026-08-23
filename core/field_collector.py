"""
core/field_collector.py — Office field-test data collector (no API cost).

Captures real Hunt usage in the office LAN so you can export a bundle,
bring it back to dev, and refine synonyms / domain expand / eval fixtures.

Storage (under tiga_work/):
  field_data/search_events.jsonl   — rich per-query retrieval events
  field_data/labels.yaml           — gold expected_paths for refinement
  field_exports/                   — zip bundles ready to copy off-site
  field_imports/                   — imported bundles from office runs

CLI:
  python tiga.py collect status
  python tiga.py collect export [--since 2026-08-01]
  python tiga.py collect import path/to/field_export_*.zip
  python tiga.py collect label --query "..." --expected path/to/file.pdf
"""

from __future__ import annotations

import json
import logging
import platform
import shutil
import socket
import sqlite3
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from config import Config, cfg as _module_cfg

logger = logging.getLogger(__name__)

_EVENTS_FILE = "search_events.jsonl"
_LABELS_FILE = "labels.yaml"
_MANIFEST_NAME = "manifest.json"


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def field_data_dir(cfg: Config | None = None) -> Path:
    c = cfg or _module_cfg
    return c.work_dir / "field_data"


def field_exports_dir(cfg: Config | None = None) -> Path:
    c = cfg or _module_cfg
    return c.work_dir / "field_exports"


def field_imports_dir(cfg: Config | None = None) -> Path:
    c = cfg or _module_cfg
    return c.work_dir / "field_imports"


def ensure_field_dirs(cfg: Config | None = None) -> None:
    for d in (field_data_dir(cfg), field_exports_dir(cfg), field_imports_dir(cfg)):
        d.mkdir(parents=True, exist_ok=True)
    labels = field_data_dir(cfg) / _LABELS_FILE
    if not labels.exists():
        labels.write_text(
            "# Gold labels from office testing — used to refine Hunt\n"
            "# Run: python tiga.py collect label --query \"...\" --expected path/suffix\n\n"
            "labels: []\n",
            encoding="utf-8",
        )


def is_enabled(cfg: Config | None = None) -> bool:
    c = cfg or _module_cfg
    return bool(getattr(c, "field_collect_enabled", True))


def _host_id() -> str:
    try:
        return socket.gethostname()
    except Exception:
        return platform.node() or "unknown"


def _events_path(cfg: Config | None = None) -> Path:
    return field_data_dir(cfg) / _EVENTS_FILE


def record_search_event(
    *,
    query: str,
    mode: str,
    project_code: str | None,
    confidence: float,
    duration_ms: float,
    session_id: str | None,
    results: list[dict[str, Any]],
    answer_preview: str | None = None,
    feedback_rating: int | None = None,
    cfg_obj: Config | None = None,
) -> None:
    """Append one rich search event (Hunt layer — safe to share, no API keys)."""
    cfg = cfg_obj or _module_cfg
    if not is_enabled(cfg):
        return
    ensure_field_dirs(cfg)
    max_r = int(getattr(cfg, "field_collect_max_results", 10))
    preview_len = int(getattr(cfg, "field_collect_answer_preview_chars", 240))

    row = {
        "ts": _utc_now(),
        "host": _host_id(),
        "query": query.strip(),
        "mode": mode,
        "project_code": project_code,
        "confidence": round(confidence, 3),
        "duration_ms": round(duration_ms, 1),
        "session_id": session_id,
        "result_count": len(results),
        "results": [
            {
                "rank": i + 1,
                "rel_path": r.get("rel_path", ""),
                "file_path": r.get("file_path", ""),
                "citation": r.get("citation", ""),
                "project_id": r.get("project_id", ""),
                "final_score": r.get("final_score"),
                "snippet": (r.get("snippet") or "")[:160],
            }
            for i, r in enumerate(results[:max_r])
        ],
    }
    if answer_preview and getattr(cfg, "field_collect_include_answer", False):
        row["answer_preview"] = answer_preview[:preview_len]
    if feedback_rating is not None:
        row["feedback_rating"] = feedback_rating

    path = _events_path(cfg)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_labels(cfg: Config | None = None) -> list[dict[str, Any]]:
    ensure_field_dirs(cfg)
    path = field_data_dir(cfg) / _LABELS_FILE
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        items = data.get("labels") if isinstance(data, dict) else data
        return list(items) if isinstance(items, list) else []
    except Exception as exc:
        logger.warning("Could not load labels: %s", exc)
        return []


def save_labels(labels: list[dict[str, Any]], cfg: Config | None = None) -> None:
    ensure_field_dirs(cfg)
    path = field_data_dir(cfg) / _LABELS_FILE
    path.write_text(
        yaml.dump({"labels": labels}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def add_label(
    query: str,
    expected_paths: list[str],
    *,
    notes: str | None = None,
    source: str = "manual",
    rating: int | None = None,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    """Add or update a gold label for refinement."""
    cfg = cfg_obj or _module_cfg
    ensure_field_dirs(cfg)
    labels = load_labels(cfg)
    q_norm = query.strip().lower()
    entry = {
        "query": query.strip(),
        "expected_paths": [p.strip() for p in expected_paths if p.strip()],
        "notes": notes or "",
        "source": source,
        "ts": _utc_now(),
    }
    if rating is not None:
        entry["rating"] = rating

    updated = False
    for i, existing in enumerate(labels):
        if existing.get("query", "").strip().lower() == q_norm:
            labels[i] = {**existing, **entry}
            updated = True
            break
    if not updated:
        labels.append(entry)
    save_labels(labels, cfg)
    return entry


def _read_jsonl(path: Path, since: str | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            if since and (row.get("ts") or "") < since:
                continue
            rows.append(row)
        except json.JSONDecodeError:
            continue
    return rows


def _read_log_jsonl(path: Path, since: str | None = None) -> list[dict[str, Any]]:
    return _read_jsonl(path, since)


def _feedback_rows(conn: sqlite3.Connection, since: str | None = None) -> list[dict[str, Any]]:
    sql = "SELECT * FROM feedback ORDER BY ts DESC"
    params: list[Any] = []
    if since:
        sql = "SELECT * FROM feedback WHERE ts >= ? ORDER BY ts DESC"
        params = [since]
    try:
        return [dict(r) for r in conn.execute(sql, params).fetchall()]
    except Exception:
        return []


def _index_snapshot(conn: sqlite3.Connection) -> dict[str, Any]:
    try:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM files WHERE status='INDEXED'"
        ).fetchone()
        projects = conn.execute(
            "SELECT COALESCE(project_id,'Unknown') AS pid, COUNT(*) AS n "
            "FROM files WHERE status='INDEXED' GROUP BY pid ORDER BY n DESC LIMIT 20"
        ).fetchall()
        return {
            "files_indexed": row["n"] if row else 0,
            "top_projects": [{"project_id": r["pid"], "files": r["n"]} for r in projects],
        }
    except Exception:
        return {"files_indexed": 0, "top_projects": []}


def collect_status(
    conn: sqlite3.Connection | None = None,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    cfg = cfg_obj or _module_cfg
    ensure_field_dirs(cfg)
    events = _read_jsonl(_events_path(cfg))
    labels = load_labels(cfg)
    feedback_n = 0
    if conn:
        feedback_n = len(_feedback_rows(conn))
    exports = sorted(field_exports_dir(cfg).glob("field_export_*.zip"), reverse=True)
    return {
        "enabled": is_enabled(cfg),
        "host": _host_id(),
        "search_events": len(events),
        "labels": len(labels),
        "feedback_rows": feedback_n,
        "exports": [p.name for p in exports[:10]],
        "data_dir": str(field_data_dir(cfg)),
        "exports_dir": str(field_exports_dir(cfg)),
    }


def _build_refinement_fixture(
    events: list[dict[str, Any]],
    feedback: list[dict[str, Any]],
    labels: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Build eval-style entries for dev refinement.
    Priority: explicit labels → negative feedback → low confidence events.
    """
    seen: set[str] = set()
    out: list[dict[str, Any]] = []

    def _add(entry: dict[str, Any]) -> None:
        key = entry.get("query", "").strip().lower()
        if not key or key in seen:
            return
        seen.add(key)
        out.append(entry)

    for lab in labels:
        if lab.get("expected_paths"):
            _add({
                "query": lab["query"],
                "expected_paths": lab["expected_paths"],
                "source": lab.get("source", "label"),
                "notes": lab.get("notes", ""),
            })

    for fb in feedback:
        if fb.get("rating") != -1:
            continue
        q = fb.get("query", "")
        top_paths = []
        if fb.get("result_id"):
            top_paths = [str(fb["result_id"])]
        _add({
            "query": q,
            "expected_paths": [],
            "returned_at_rank": fb.get("rank_position"),
            "wrong_result_id": fb.get("result_id"),
            "comment": fb.get("comment"),
            "source": "feedback_negative",
            "needs_label": True,
        })

    for ev in events:
        conf = float(ev.get("confidence") or 0)
        if conf >= 0.6 and ev.get("feedback_rating") != -1:
            continue
        q = ev.get("query", "")
        returned = [r.get("rel_path") or r.get("file_path") for r in ev.get("results") or []]
        _add({
            "query": q,
            "expected_paths": [],
            "returned_paths": returned[:5],
            "confidence": conf,
            "mode": ev.get("mode"),
            "source": "low_confidence_event",
            "needs_label": True,
        })

    return out


def export_bundle(
    conn: sqlite3.Connection | None = None,
    *,
    since: str | None = None,
    cfg_obj: Config | None = None,
    out_name: str | None = None,
) -> Path:
    """
    Create a zip bundle for copying off the office LAN machine.
    Returns path to the zip file.
    """
    cfg = cfg_obj or _module_cfg
    ensure_field_dirs(cfg)
    slug = out_name or f"field_export_{_ts_slug()}"
    if not slug.endswith(".zip"):
        slug = f"{slug}.zip"
    zip_path = field_exports_dir(cfg) / slug

    events = _read_jsonl(_events_path(cfg), since=since)
    labels = load_labels(cfg)
    feedback = _feedback_rows(conn, since) if conn else []
    index_snap = _index_snapshot(conn) if conn else {}

    # Redacted config snapshot (no secrets)
    config_snap = {
        "index_roots": [str(r) for r in cfg.index_roots],
        "reranker_enabled": cfg.reranker_enabled,
        "top_k": cfg.top_k,
        "field_collect_enabled": is_enabled(cfg),
    }

    manifest = {
        "export_version": 1,
        "exported_at": _utc_now(),
        "host": _host_id(),
        "since_filter": since,
        "counts": {
            "search_events": len(events),
            "labels": len(labels),
            "feedback": len(feedback),
        },
        "index_snapshot": index_snap,
        "config_snapshot": config_snap,
        "refinement_instructions": (
            "Copy this zip to dev. Run: python tiga.py collect import <zip> "
            "Then review field_data/refinement_candidates.yaml and office_eval_merged.yaml"
        ),
    }

    refinement = _build_refinement_fixture(events, feedback, labels)
    eval_entries = [
        e for e in refinement if e.get("expected_paths") and not e.get("needs_label")
    ]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(_MANIFEST_NAME, json.dumps(manifest, indent=2))
        zf.writestr("search_events.jsonl", "\n".join(json.dumps(e) for e in events) + ("\n" if events else ""))
        zf.writestr("labels.yaml", yaml.dump({"labels": labels}, allow_unicode=True))
        zf.writestr("feedback.json", json.dumps(feedback, indent=2, default=str))
        zf.writestr(
            "refinement_candidates.yaml",
            yaml.dump({"candidates": refinement}, allow_unicode=True, sort_keys=False),
        )
        zf.writestr(
            "office_eval.yaml",
            yaml.dump(eval_entries, allow_unicode=True, sort_keys=False),
        )

        log_dir = cfg.get_log_dir()
        for name in ("queries.log", "low_confidence.log"):
            lp = log_dir / name
            if lp.exists():
                zf.write(lp, arcname=f"logs/{name}")

        # Latest validate report if present
        reports = sorted(cfg.get_report_dir().glob("validate_*.json"), reverse=True)
        if reports:
            zf.write(reports[0], arcname=f"reports/{reports[0].name}")

    logger.info("Field export written: %s", zip_path)
    return zip_path


def import_bundle(
    zip_path: Path,
    *,
    merge_events: bool = True,
    merge_labels: bool = True,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    """
    Import an office export zip into field_imports/ and merge into field_data/.
    """
    cfg = cfg_obj or _module_cfg
    ensure_field_dirs(cfg)
    zip_path = Path(zip_path).resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"Export not found: {zip_path}")

    dest = field_imports_dir(cfg) / zip_path.stem
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    manifest_path = dest / _MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    merged = {"events_added": 0, "labels_added": 0, "labels_updated": 0}

    if merge_events:
        incoming = _read_jsonl(dest / "search_events.jsonl")
        existing_keys = {
            (e.get("ts"), e.get("query"))
            for e in _read_jsonl(_events_path(cfg))
        }
        with _events_path(cfg).open("a", encoding="utf-8") as f:
            for row in incoming:
                key = (row.get("ts"), row.get("query"))
                if key in existing_keys:
                    continue
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                existing_keys.add(key)
                merged["events_added"] += 1

    if merge_labels:
        incoming_labels: list[dict[str, Any]] = []
        labels_file = dest / "labels.yaml"
        if labels_file.exists():
            data = yaml.safe_load(labels_file.read_text(encoding="utf-8")) or {}
            incoming_labels = list(data.get("labels") or [])
        current = load_labels(cfg)
        by_q = {x.get("query", "").strip().lower(): x for x in current}
        for lab in incoming_labels:
            qk = lab.get("query", "").strip().lower()
            if not qk:
                continue
            if qk in by_q:
                by_q[qk] = {**by_q[qk], **lab, "imported_from": zip_path.stem}
                merged["labels_updated"] += 1
            else:
                lab["imported_from"] = zip_path.stem
                by_q[qk] = lab
                merged["labels_added"] += 1
        save_labels(list(by_q.values()), cfg)

    # Copy refinement artifacts for easy review
    fixtures_dir = cfg.work_dir / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    for name in ("refinement_candidates.yaml", "office_eval.yaml"):
        src = dest / name
        if src.exists():
            merged_name = fixtures_dir / f"{zip_path.stem}_{name}"
            shutil.copy2(src, merged_name)

    office_merged = fixtures_dir / "office_eval_merged.yaml"
    eval_src = dest / "office_eval.yaml"
    if eval_src.exists():
        new_entries = yaml.safe_load(eval_src.read_text(encoding="utf-8")) or []
        if not isinstance(new_entries, list):
            new_entries = []
        existing: list[dict[str, Any]] = []
        if office_merged.exists():
            existing = yaml.safe_load(office_merged.read_text(encoding="utf-8")) or []
            if not isinstance(existing, list):
                existing = []
        seen = {e.get("query", "").lower() for e in existing}
        for e in new_entries:
            if e.get("query", "").lower() not in seen:
                existing.append(e)
                seen.add(e.get("query", "").lower())
        office_merged.write_text(
            yaml.dump(existing, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )

    return {
        "imported_to": str(dest),
        "manifest": manifest,
        "merged": merged,
        "office_eval_merged": str(office_merged) if eval_src.exists() else None,
    }
