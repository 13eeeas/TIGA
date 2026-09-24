"""
core/atlas_wiki.py — Cited project wiki inside TIGA Hunt.

North star: Notion-like browse + Wikipedia-like contribution + Grokopedia-like
auto draft — every durable claim needs a citation path.

Layers
------
  auto   : drafted when a project is indexed, stored on the overlay, refreshed
            only when new files are indexed. A missing draft is built once on open.
  wiki   : human pins / facts / hidden paths in tiga_work/atlas/*.overlay.json
  gate   : Published only when curated identity fields + ≥1 pin + cited facts exist

Public API
----------
  list_wiki_projects(conn) -> list
  get_wiki_page(code, conn, cfg) -> dict   # auto + wiki merge
  wiki_pin / wiki_hide / wiki_fact / wiki_unhide / wiki_overview
  wiki_compose(code, question, conn, cfg) -> dict  # Einstein-lite stub/API
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import cfg as _module_cfg, Config

logger = logging.getLogger(__name__)
_overlay_lock = threading.Lock()

# Index-time classification. These match files.doc_type / content_type / extension.
_DRAWING_DOCS = {
    "general arrangement", "floor plan", "section", "elevation", "detail",
    "site plan", "roof plan",
}
_CORR_DOCS = {"rfi", "transmittal", "minutes"}
_KEY_DOCS = {"specification", "schedule"}
_DRAWING_EXT = {".dwg", ".dxf", ".rvt", ".rfa", ".ifc", ".3dm", ".skp", ".nwd", ".nwc"}
_CORR_EXT = {".msg", ".eml"}
_PER_BUCKET = 4

PIN_ROLES = (
    "overview_deck",
    "design_report",
    "authoritative_ga",
    "authority_pack",
    "model_master",
    "other",
)

# Identity fields for the <5s wiki blurb. All four must be filled (not
# "Needs curation") before a page can show Published / a health score.
CURATED_FIELDS = ("typology", "client", "stage", "location")
OVERVIEW_FIELDS = ("name", "typology", "client", "stage", "location")
_EMPTY_TOKENS = {
    "",
    "needs curation",
    "unknown",
    "n/a",
    "na",
    "none",
    "—",
    "-",
    "tbd",
}
_AUTO_SUMMARY_MARKERS = (
    "auto-draft grokopedia",
    "hunt proposed candidates",
    "confirm pins and cite facts",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(code: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", code.lower()).strip("-") or "project"


def field_is_filled(value: Any) -> bool:
    """True when a curated identity field is a real value, not a placeholder."""
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in _EMPTY_TOKENS


def is_human_summary(summary: str | None) -> bool:
    """True when summary is staff-written, not the Grokopedia auto placeholder."""
    text = (summary or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return not any(marker in lowered for marker in _AUTO_SUMMARY_MARKERS)


def missing_curated_fields(project: dict[str, Any] | None) -> list[str]:
    proj = project or {}
    return [key for key in CURATED_FIELDS if not field_is_filled(proj.get(key))]


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def compose_blurb(project: dict[str, Any] | None, summary: str | None = None) -> str:
    """One short wiki sentence: what / client or typology / stage / location."""
    if is_human_summary(summary):
        return str(summary).strip()

    proj = project or {}
    name = str(proj.get("name") or proj.get("code") or "This project").strip()
    typology = str(proj.get("typology") or "").strip() if field_is_filled(proj.get("typology")) else ""
    client = str(proj.get("client") or "").strip() if field_is_filled(proj.get("client")) else ""
    stage = str(proj.get("stage") or "").strip() if field_is_filled(proj.get("stage")) else ""
    location = str(proj.get("location") or "").strip() if field_is_filled(proj.get("location")) else ""

    if not any((typology, client, stage, location)):
        return (
            f"{name} — Needs curation. Add what this job is, client or typology, "
            "stage, and location if known."
        )

    if typology and client:
        lead = f"{name} is {_article(typology)} {typology} project for {client}"
    elif typology:
        lead = f"{name} is {_article(typology)} {typology} project"
    elif client:
        lead = f"{name} is a project for {client}"
    else:
        lead = name

    extras: list[str] = []
    if stage:
        extras.append(f"currently at {stage}")
    if location:
        extras.append(f"in {location}")
    if extras:
        return f"{lead}, {', '.join(extras)}."
    return f"{lead}."


def _project_from_card_and_overlay(
    code: str,
    card: dict[str, Any] | None,
    overlay: dict[str, Any],
) -> dict[str, Any]:
    project = {
        "code": code,
        "name": (card or {}).get("name") or code,
        "aliases": (card or {}).get("alt_names") or [],
        "typology": (card or {}).get("typology_primary") or "",
        "stage": (card or {}).get("stage") or "",
        "location": (card or {}).get("location") or "",
        "client": (card or {}).get("client") or "",
        "status": "needs-curation",
    }
    for key, value in (overlay.get("project") or {}).items():
        if value not in (None, "", []):
            project[key] = value
    return project


def _atlas_dir(cfg_obj: Config | None = None) -> Path:
    cfg = cfg_obj or _module_cfg
    d = cfg.work_dir / "atlas"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _overlay_path(code: str, cfg_obj: Config | None = None) -> Path:
    """One file per project code. Slugs that collide get a short hash suffix."""
    folder = _atlas_dir(cfg_obj)
    slug = _slug(code)
    direct = folder / f"{slug}.overlay.json"
    for path in sorted(folder.glob(f"{slug}*.overlay.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if str(data.get("project_code") or "") == code:
            return path
    if direct.exists():
        try:
            owner = json.loads(direct.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            owner = {}
        other = str(owner.get("project_code") or "")
        if other and other != code:
            digest = hashlib.sha1(code.encode("utf-8")).hexdigest()[:8]
            return folder / f"{slug}-{digest}.overlay.json"
    return direct


def project_codes_for_scope(
    *,
    typology: str | None = None,
    location: str | None = None,
    cfg_obj: Config | None = None,
) -> list[str]:
    """Project codes whose wiki overlay matches a typology and/or location.

    Hunt uses this so a typology filter follows the project page, not the
    empty files.typology column.
    """
    want_type = (typology or "").strip().casefold()
    want_loc = (location or "").strip().casefold()
    if not want_type and not want_loc:
        return []
    found: list[str] = []
    for path in sorted(_atlas_dir(cfg_obj).glob("*.overlay.json")):
        try:
            overlay = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        project = overlay.get("project") or {}
        code = str(overlay.get("project_code") or project.get("code") or "").strip()
        if not code:
            continue
        if want_type:
            have = str(project.get("typology") or "").strip().casefold()
            if want_type not in have:
                continue
        if want_loc and want_loc not in str(project.get("location") or "").casefold():
            continue
        found.append(code)
    return found


def tagged_typologies(cfg_obj: Config | None = None) -> list[str]:
    """Distinct typology labels already saved on project overlays."""
    seen: set[str] = set()
    labels: list[str] = []
    for path in sorted(_atlas_dir(cfg_obj).glob("*.overlay.json")):
        try:
            overlay = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        label = str((overlay.get("project") or {}).get("typology") or "").strip()
        key = label.casefold()
        if not label or key in seen or key in {"unknown", "needs curation", "n/a", "na", "none", "tbd"}:
            continue
        seen.add(key)
        labels.append(label)
    return labels


def load_overlay(code: str, cfg_obj: Config | None = None) -> dict[str, Any]:
    path = _overlay_path(code, cfg_obj)
    empty = {
        "schema_version": 1,
        "product": "tiga-atlas-overlay",
        "project_code": code,
        "updated_at": _utc(),
        "pins": [],
        "facts": [],
        "hidden_paths": [],
        "summary": None,
        "project": {},
    }
    if not path.exists():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Atlas overlay unreadable for %s: %s", code, exc)
        return empty
    if not isinstance(data, dict):
        logger.warning("Atlas overlay for %s is not an object", code)
        return empty
    return data


def save_overlay(code: str, overlay: dict[str, Any], cfg_obj: Config | None = None) -> Path:
    overlay["project_code"] = code
    overlay["updated_at"] = _utc()
    path = _overlay_path(code, cfg_obj)
    payload = json.dumps(overlay, indent=2)
    tmp = path.with_suffix(".json.tmp")
    with _overlay_lock:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(tmp, path)
    return path


def list_wiki_projects(
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,
) -> list[dict[str, Any]]:
    from core.project_card import get_project_card

    rows = conn.execute(
        "SELECT COALESCE(project_id, 'Unknown') AS project_id, "
        "COUNT(*) AS file_count, "
        "SUM(CASE WHEN status = 'INDEXED' THEN 1 ELSE 0 END) AS files_indexed, "
        "SUM(CASE WHEN status IN ('DISCOVERED','EXTRACTED','EMBEDDED') "
        "         THEN 1 ELSE 0 END) AS files_in_progress, "
        "SUM(CASE WHEN status = 'FAILED' THEN 1 ELSE 0 END) AS files_failed "
        "FROM files GROUP BY project_id ORDER BY file_count DESC"
    ).fetchall()
    out = []
    for r in rows:
        code = r["project_id"]
        ov = load_overlay(code, cfg_obj)
        card = get_project_card(code, conn=conn)
        project = _project_from_card_and_overlay(code, card, ov)
        facts_by_key = {f["key"]: f for f in _auto_facts_from_card(card) if f.get("key")}
        for fact in ov.get("facts") or []:
            if fact.get("key"):
                facts_by_key[fact["key"]] = fact
        page = {
            "project": project,
            "summary": ov.get("summary"),
            "pins": ov.get("pins") or [],
            "facts": list(facts_by_key.values()),
        }
        health = compute_health(page)
        blurb = compose_blurb(project, ov.get("summary"))
        pin_n = len(ov.get("pins") or [])
        fact_n = len(ov.get("facts") or [])
        files_indexed = int(r["files_indexed"] or 0)
        files_in_progress = int(r["files_in_progress"] or 0)
        files_failed = int(r["files_failed"] or 0)
        if files_in_progress:
            index_state = "in_progress"
        elif files_indexed:
            index_state = "indexed"
        elif files_failed:
            index_state = "failed"
        else:
            index_state = "configured"
        out.append(
            {
                "project_id": code,
                "name": project.get("name") or code,
                "blurb": blurb,
                "file_count": r["file_count"],
                "files_indexed": files_indexed,
                "files_in_progress": files_in_progress,
                "files_failed": files_failed,
                "index_state": index_state,
                "wiki_pins": pin_n,
                "wiki_facts": fact_n,
                "missing_fields": health.get("missing_fields") or [],
                "published": health["published"],
                "status": "published" if health["published"] else "needs-curation",
            }
        )
    return out


def _ext(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text and not text.startswith("."):
        text = "." + text
    return text


# Same shelves on every job. A shelf is omitted when the folder is not there.
_SHELVES: tuple[tuple[str, str], ...] = (
    ("brief", "Brief"),
    ("client", "Client"),
    ("design", "Design"),
    ("drawings", "Drawings"),
    ("consultants", "Consultants"),
    ("authorities", "Authorities"),
    ("contractors", "Contractors"),
    ("meetings", "Meetings"),
    ("contract", "Contract"),
    ("site", "Site"),
    ("presentations", "Presentations"),
)
_SHELF_FOLDERS = {
    "brief": "brief",
    "client": "client",
    "consultant": "consultants",
    "consultants": "consultants",
    "authorities": "authorities",
    "authority": "authorities",
    "submission": "authorities",
    "contractor": "contractors",
    "contractors": "contractors",
    "meeting": "meetings",
    "meetings": "meetings",
    "contract documentation": "contract",
    "issued": "drawings",
    "archi study": "design",
    "design": "design",
    "id": "design",
    "site photos": "site",
    "construction photos": "site",
    "site information": "site",
    "presentations": "presentations",
    "presentation": "presentations",
    "ppt": "presentations",
}
_CONTAINERS = {"documents", "document", "images", "image", "cad"}
_SKIP_FOLDERS = {
    "working", "temp", "temporary", "superseded", "archived", "backup", "obsolete",
    "bim", "library",
}
_SHELF_TITLES = {key: title for key, title in _SHELVES}
_STREAM_LIMIT = 8


def _strip_index(part: str) -> str:
    text = part.strip()
    while True:
        nxt = re.sub(r"^\d+\s+", "", text)
        if nxt == text:
            return text.strip()
        text = nxt


def _folder_label(part: str) -> str:
    return _strip_index(part).casefold()


def _folder_display(part: str) -> str:
    return _strip_index(part)


def _place(path: str) -> tuple[str, str] | None:
    """Shelf and stream from the folder names. None when the path is not part of the job snapshot."""
    parts = [p for p in path.replace("\\", "/").split("/") if p]
    if len(parts) < 2:
        return None
    folders = parts[:-1]
    shelf: str | None = None
    stream = ""
    saw_cad = False
    for index, part in enumerate(folders):
        label = _folder_label(part)
        if label in _SKIP_FOLDERS:
            return None
        if label == "cad":
            saw_cad = True
        if label in _CONTAINERS:
            continue
        if shelf is not None:
            continue
        mapped = _SHELF_FOLDERS.get(label)
        if not mapped:
            continue
        shelf = mapped
        nxt = folders[index + 1] if index + 1 < len(folders) else ""
        nxt_label = _folder_label(nxt) if nxt else ""
        if nxt and nxt_label not in _CONTAINERS and nxt_label not in _SHELF_FOLDERS and nxt_label not in _SKIP_FOLDERS:
            stream = _folder_display(nxt)
        else:
            stream = _SHELF_TITLES[mapped]
    if shelf is None and saw_cad:
        shelf = "drawings"
        stream = "Drawings"
    if shelf is None or not stream:
        return None
    return shelf, stream


def _candidate(row: sqlite3.Row, section: str, title: str, stream: str) -> dict[str, Any]:
    path = str(row["file_path"] or "")
    when = str(row["file_date"] or "").strip()[:10]
    return {
        "title": str(row["file_name"] or Path(path).name),
        "path": path,
        "snippet": when,
        "score": None,
        "via": f"record:{section}",
        "section": section,
        "section_title": title,
        "stream": stream,
        "suggested_role": "other",
        "kind": "candidate",
        "pinned": False,
    }


def _auto_candidates(code: str, cfg_obj: Config, conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """One timeline per folder that exists under the job's shelves."""
    del cfg_obj
    rows = conn.execute(
        """
        SELECT file_path, file_name, extension, doc_type, file_date, mtime_epoch
        FROM files
        WHERE COALESCE(project_id, 'Unknown') = ?
          AND COALESCE(is_superseded, 0) = 0
        """,
        (code,),
    ).fetchall()
    grouped: dict[tuple[str, str], list[sqlite3.Row]] = {}
    for row in rows:
        path = str(row["file_path"] or "")
        name = str(row["file_name"] or "").strip()
        if not path or not name:
            continue
        ext = _ext(row["extension"])
        if ext in {".lnk", ".zip", ".psd", ".tmp"}:
            continue
        placed = _place(path)
        if placed is None:
            continue
        shelf, _stream = placed
        if ext in {".jpg", ".jpeg", ".png"} and shelf != "site":
            continue
        if ext not in {".pdf", ".doc", ".docx", ".dwg", ".ppt", ".pptx", ".xls", ".xlsx", ".jpg", ".jpeg", ".png"}:
            continue
        grouped.setdefault(placed, []).append(row)

    items: list[dict[str, Any]] = []
    for (shelf, stream), bucket in grouped.items():
        bucket.sort(
            key=lambda row: (str(row["file_date"] or ""), float(row["mtime_epoch"] or 0)),
            reverse=True,
        )
        title = _SHELF_TITLES[shelf]
        for row in bucket[:_STREAM_LIMIT]:
            items.append(_candidate(row, shelf, title, stream))
    return items


def refresh_auto_drafts(
    conn: sqlite3.Connection,
    project_ids: set[str] | list[str],
    cfg_obj: Config | None = None,
) -> int:
    """Rewrite the auto half of the wiki for projects that just changed.

    Pins, facts, hides, and the staff overview stay on the overlay.
    """
    updated = 0
    for code in sorted({c for c in project_ids if c and c != "Unknown"}):
        overlay = load_overlay(code, cfg_obj)
        overlay["auto_candidates"] = _auto_candidates(code, cfg_obj or _module_cfg, conn)
        overlay["auto_built_at"] = _utc()
        save_overlay(code, overlay, cfg_obj)
        updated += 1
    return updated


def _auto_facts_from_card(card: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not card:
        return []
    facts = []
    mapping = [
        ("typology", "Typology", card.get("typology_primary")),
        ("location", "Location", card.get("location")),
        ("client", "Client", card.get("client")),
        ("stage", "Stage", card.get("stage")),
        ("gfa", "GFA (sqm)", card.get("gfa_sqm")),
    ]
    for key, label, value in mapping:
        if value in (None, "", []):
            continue
        facts.append(
            {
                "key": key,
                "label": label,
                "value": str(value),
                "cite_paths": [],  # auto facts start uncited → wiki must confirm
                "status": "draft",
                "source": "auto",
            }
        )
    return facts


def compute_health(page: dict[str, Any]) -> dict[str, Any]:
    """Published only with curated identity + ≥1 pin + cited facts.

    Empty / Needs-curation cards must never report Published or 100/100.
    File counts are not part of this gate.
    """
    facts = page.get("facts") or []
    pins = page.get("pins") or []
    project = page.get("project") or {}
    missing = missing_curated_fields(project)
    # Machine drafts stay on the page as rumours. Only staff facts without a
    # citation block Published.
    uncited = [
        f.get("key")
        for f in facts
        if not (f.get("cite_paths") or []) and f.get("source") != "auto"
    ]
    cited_n = sum(1 for f in facts if f.get("cite_paths"))
    fact_n = len(facts)
    ratio = (cited_n / fact_n) if fact_n else 0.0
    fields_ok = not missing
    pin_ok = len(pins) >= 1
    facts_ok = cited_n >= 1 and len(uncited) == 0
    blurb_ok = is_human_summary(page.get("summary")) or fields_ok
    checks = [
        {
            "id": "curated_fields",
            "ok": fields_ok,
            "detail": (
                "Identity fields filled"
                if fields_ok
                else "Needs curation: " + ", ".join(missing)
            ),
        },
        {
            "id": "has_pin",
            "ok": pin_ok,
            "detail": f"{len(pins)} pin(s) — need ≥1 for Published",
        },
        {
            "id": "cited_facts",
            "ok": facts_ok,
            "detail": (
                f"{cited_n}/{fact_n} facts cited — need ≥1 cited fact and none uncited"
            ),
        },
        {
            "id": "readable_blurb",
            "ok": blurb_ok,
            "detail": "Wiki blurb readable in a glance",
        },
    ]
    ok_n = sum(1 for c in checks if c["ok"])
    published = fields_ok and pin_ok and facts_ok
    score = int(round(100 * ok_n / len(checks))) if checks else 0
    grade = None
    if published:
        # A staff-written blurb is the tighter article. Identity-only blurbs are B.
        grade = "A" if is_human_summary(page.get("summary")) else "B"
    return {
        "score": score,
        "grade": grade,
        "checks": checks,
        "pin_count": len(pins),
        "uncited_facts": len(uncited),
        "cited_facts": cited_n,
        "fact_cited_ratio": round(ratio, 2),
        "published": published,
        "show_health_score": published,
        "missing_fields": missing,
        "state": "Published" if published else "Needs curation",
    }


def _focus_hit(conn: sqlite3.Connection, code: str, like: str) -> dict[str, str] | None:
    row = conn.execute(
        """
        SELECT file_name, file_path, file_date
        FROM files
        WHERE COALESCE(project_id, 'Unknown') = ?
          AND COALESCE(is_superseded, 0) = 0
          AND file_name LIKE ?
          AND lower(COALESCE(extension, '')) IN ('.pdf', 'pdf', '.docx', 'docx', '.xlsx', 'xlsx', '.dwg', 'dwg')
        ORDER BY COALESCE(file_date, '') DESC, file_name
        LIMIT 1
        """,
        (code, like),
    ).fetchone()
    if row is None:
        return None
    return {
        "title": str(row["file_name"] or ""),
        "path": str(row["file_path"] or ""),
        "date": str(row["file_date"] or "")[:10],
    }


def _project_focus(code: str, conn: sqlite3.Connection) -> dict[str, Any]:
    """A few objects a person can think with. Claims stop at the filename."""
    specs = [
        ("reception-f", "Reception", "%SK-022F%Reception%"),
        ("reception-e", "Reception", "%SK-022E%Reception%"),
        ("glass", "Facade", "%Glass Specs Reconciliation%"),
        ("ettv", "Facade", "%ETTV%20260901%"),
        ("waterproof", "Waterproofing", "%Waterproofing system%"),
        ("clearance", "Approvals", "%clearance certificate%SUCCESS%"),
        ("ihg", "Hotel operator", "%email from IHG%"),
        ("meeting", "Coordination", "%Stakeholders Weekly Meeting%"),
        ("bollard-vo", "Reception", "%VO-024%Bollard%"),
    ]
    found = {key: _focus_hit(conn, code, like) for key, _area, like in specs}
    found = {key: hit for key, hit in found.items() if hit}

    def ev(*keys: str) -> list[dict[str, str]]:
        return [found[key] for key in keys if key in found]

    events: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    if "clearance" in found:
        events.append({
            "date": found["clearance"]["date"] or "2026-09-10",
            "area": "Approvals",
            "title": "A clearance certificate was recorded as successful",
            "why": "",
            "evidence": ev("clearance"),
        })
    if "ihg" in found:
        events.append({
            "date": found["ihg"]["date"],
            "area": "Hotel operator",
            "title": "An email from IHG was filed",
            "why": "What the operator said is not resolved. The message has not been read.",
            "evidence": ev("ihg"),
        })
    if "meeting" in found:
        events.append({
            "date": found["meeting"]["date"],
            "area": "Coordination",
            "title": "Stakeholder weekly meeting correspondence was filed",
            "why": "The meeting outcome is not resolved. This is the correspondence, not minutes that have been read.",
            "evidence": ev("meeting"),
        })
    if "reception-f" in found:
        events.append({
            "date": found["reception-f"]["date"] or "2026-09-02",
            "area": "Reception",
            "title": "L1 reception plan Rev F was issued",
            "why": "The filename says drop-off bollard revision. Why the bollards changed is not resolved.",
            "evidence": ev("reception-f", "reception-e"),
        })
        decisions.append({
            "area": "Reception",
            "title": "Reception sheet is at Rev F",
            "status": "Issued",
            "date": found["reception-f"]["date"] or "2026-09-02",
            "why": "Why Rev F was adopted is not resolved from indexed evidence.",
            "supersedes": "Rev E of the same sheet" if "reception-e" in found else "",
            "evidence": ev("reception-f", "reception-e"),
        })
    if "glass" in found:
        events.append({
            "date": found["glass"]["date"] or "2026-09-01",
            "area": "Facade",
            "title": "A glass-specification reconciliation file was filed",
            "why": "Whether the specification is agreed is not resolved. The file has not been read.",
            "evidence": ev("glass", "ettv"),
        })
        issues.append({
            "area": "Facade",
            "title": "Glass specification reconciliation",
            "status": "Filed, not read",
            "date": found["glass"]["date"] or "2026-09-01",
            "evidence": ev("glass", "ettv"),
        })
    if "waterproof" in found:
        issues.append({
            "area": "Waterproofing",
            "title": "Waterproofing system Rev B is in the index",
            "status": "Filed, not read",
            "date": found["waterproof"]["date"],
            "evidence": ev("waterproof"),
        })
    if "bollard-vo" in found:
        issues.append({
            "area": "Reception",
            "title": "A bollard variation, VO-024, was filed after Rev F",
            "status": "Filed, not read",
            "date": found["bollard-vo"]["date"],
            "evidence": ev("bollard-vo", "reception-f"),
        })

    events.sort(key=lambda item: item.get("date") or "", reverse=True)
    working = []
    for key, label in (
        ("reception-f", "Current reception sheet"),
        ("clearance", "Latest clearance record"),
        ("meeting", "Latest stakeholder correspondence"),
        ("glass", "Glass specification file"),
        ("ihg", "Latest IHG email"),
        ("waterproof", "Waterproofing revision"),
    ):
        if key in found:
            working.append({**found[key], "role": label, "area": next(a for k, a, _ in specs if k == key)})
    return {"events": events, "issues": issues, "decisions": decisions, "working": working}


def get_wiki_page(
    code: str,
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    cfg = cfg_obj or _module_cfg
    from core.project_card import get_project_card

    card = get_project_card(code, conn=conn)
    overlay = load_overlay(code, cfg)
    if not overlay.get("auto_built_at"):
        refresh_auto_drafts(conn, [code], cfg)
        overlay = load_overlay(code, cfg)
    candidates = list(overlay.get("auto_candidates") or [])
    project = _project_from_card_and_overlay(code, card, overlay)

    human_summary = overlay.get("summary")
    if not is_human_summary(human_summary):
        human_summary = None
    blurb = compose_blurb(project, human_summary)

    pins = list(overlay.get("pins") or [])
    pin_paths = {p.get("path") for p in pins if p.get("path")}
    hidden = set(overlay.get("hidden_paths") or [])

    grouped: dict[str, dict[str, Any]] = {}
    for item in candidates:
        if item["path"] in hidden:
            continue
        if item["path"] in pin_paths:
            item = {**item, "pinned": True, "kind": "pinned"}
            for p in pins:
                if p.get("path") == item["path"]:
                    item["role"] = p.get("role")
        key = str(item.get("section") or "")
        if not key:
            continue
        bucket = grouped.setdefault(
            key,
            {
                "title": item.get("section_title") or _SHELF_TITLES.get(key, key),
                "summary": "",
                "items": [],
            },
        )
        bucket["items"].append(item)
    sections = {key: grouped[key] for key, _title in _SHELVES if grouped.get(key, {}).get("items")}

    # Auto facts then overlay facts win by key
    facts_by_key = {f["key"]: f for f in _auto_facts_from_card(card) if f.get("key")}
    for f in overlay.get("facts") or []:
        if f.get("key"):
            facts_by_key[f["key"]] = f
    facts = list(facts_by_key.values())

    file_row = conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE COALESCE(project_id, 'Unknown') = ?",
        (code,),
    ).fetchone()
    file_count = int(file_row["n"] if file_row is not None else 0)

    focus = _project_focus(code, conn)

    page = {
        "schema_version": 1,
        "product": "tiga-atlas-wiki",
        "project": project,
        "focus": focus,
        "summary": human_summary or "",
        "blurb": blurb,
        "concept_summary": str((card or {}).get("concept_summary") or "").strip(),
        "pins": pins,
        "facts": facts,
        "sections": sections,
        "open_questions": overlay.get("open_questions")
        or [
            "Fill the overview blurb (what / client or typology / stage / location)",
            "Pin the overview deck / GA / authority pack",
            "Add cite_paths to every auto fact you keep",
            "Hide junk candidates (Copy of…, caches)",
        ],
        "index": {
            "file_count": file_count,
            "candidate_count": len(candidates),
            "pin_count": len(pins),
            "hidden_count": len(hidden),
            "overlay": _overlay_path(code, cfg).name,
            "auto_built_at": overlay.get("auto_built_at"),
        },
        "sources": {
            "mode": "hunt+wiki",
            "built_at": _utc(),
            "card_found": card is not None,
            "candidate_count": len(candidates),
        },
    }
    page["health"] = compute_health(page)
    if page["health"]["published"]:
        page["project"]["status"] = "published"
    else:
        page["project"]["status"] = "needs-curation"
    return page


def wiki_pin(
    code: str,
    *,
    role: str,
    path: str,
    title: str = "",
    note: str = "",
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    if role not in PIN_ROLES:
        role = "other"
    ov = load_overlay(code, cfg_obj)
    pins = ov.setdefault("pins", [])
    pin = {
        "role": role,
        "path": path,
        "title": title or Path(path).name,
        "note": note or "Pinned by wiki contributor",
        "source": "wiki",
        "pinned_at": _utc(),
    }
    for i, existing in enumerate(pins):
        if existing.get("path") == path:
            pins[i] = pin
            break
    else:
        pins.append(pin)
    save_overlay(code, ov, cfg_obj)
    return pin


def wiki_hide(code: str, path: str, cfg_obj: Config | None = None) -> None:
    ov = load_overlay(code, cfg_obj)
    hidden = ov.setdefault("hidden_paths", [])
    if path not in hidden:
        hidden.append(path)
    save_overlay(code, ov, cfg_obj)


def wiki_unhide(code: str, path: str, cfg_obj: Config | None = None) -> None:
    ov = load_overlay(code, cfg_obj)
    ov["hidden_paths"] = [p for p in (ov.get("hidden_paths") or []) if p != path]
    save_overlay(code, ov, cfg_obj)


def wiki_fact(
    code: str,
    *,
    key: str,
    label: str,
    value: str,
    cite_paths: list[str] | None = None,
    status: str = "draft",
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    if status not in ("draft", "verified", "disputed"):
        status = "draft"
    fact = {
        "key": key,
        "label": label or key,
        "value": value,
        "cite_paths": cite_paths or [],
        "status": status,
        "source": "wiki",
    }
    ov = load_overlay(code, cfg_obj)
    facts = ov.setdefault("facts", [])
    for i, existing in enumerate(facts):
        if existing.get("key") == key:
            facts[i] = fact
            break
    else:
        facts.append(fact)
    save_overlay(code, ov, cfg_obj)
    return fact


def wiki_overview(
    code: str,
    *,
    summary: str | None = None,
    project_fields: dict[str, Any] | None = None,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    """Save the human wiki blurb + identity fields into the overlay."""
    ov = load_overlay(code, cfg_obj)
    if summary is not None:
        text = summary.strip()
        ov["summary"] = text or None
    proj = ov.setdefault("project", {})
    for key, value in (project_fields or {}).items():
        if key not in OVERVIEW_FIELDS:
            continue
        if isinstance(value, str):
            value = value.strip()
        if value in (None, ""):
            proj.pop(key, None)
            proj.pop(f"{key}_source", None)
        else:
            proj[key] = value
            if key in {"typology", "client", "stage", "location"}:
                proj[f"{key}_source"] = "human"
    if summary is not None:
        ov["summary_source"] = "human" if ov.get("summary") else None
    save_overlay(code, ov, cfg_obj)
    return {
        "summary": ov.get("summary"),
        "project": dict(proj),
        "overlay": _overlay_path(code, cfg_obj).name,
    }


def evidence_pack(page: dict[str, Any], max_n: int = 12) -> list[dict[str, Any]]:
    pack: list[dict[str, Any]] = []
    for pin in page.get("pins") or []:
        pack.append({"kind": "pin", **{k: pin.get(k) for k in ("role", "title", "path", "note")}})
    for fact in page.get("facts") or []:
        if not fact.get("cite_paths"):
            continue
        pack.append(
            {
                "kind": "fact",
                "key": fact.get("key"),
                "label": fact.get("label"),
                "value": fact.get("value"),
                "cite_paths": fact.get("cite_paths"),
                "status": fact.get("status"),
            }
        )
    return pack[:max_n]


def wiki_compose(page: dict[str, Any], question: str) -> dict[str, Any]:
    """Einstein-lite: pins + cited facts only. API if keyed; else local stub."""
    if not (page.get("pins") or []):
        return {
            "ok": False,
            "provider": "none",
            "answer": "Ask is locked until someone pins at least one authoritative document on this project page.",
            "pack_size": 0,
        }
    pack = evidence_pack(page)
    api_key = os.environ.get("TIGA_LLM_API_KEY") or os.environ.get("TIGA_OPENAI_API_KEY")
    if not api_key:
        lines = [
            "(local Ask stub — set TIGA_LLM_API_KEY for enterprise compose)",
            f"Q: {question}",
            "",
            "Cited evidence only:",
        ]
        for item in pack:
            if item["kind"] == "fact":
                lines.append(f"- FACT {item.get('label')}: {item.get('value')} ← {item.get('cite_paths')}")
            elif item["kind"] == "pin":
                lines.append(f"- PIN {item.get('role')}: {item.get('path')}")
            else:
                lines.append(f"- CANDIDATE {item.get('title')}: {item.get('path')}")
        lines.append("")
        lines.append("Answer from the evidence above only. Do not invent.")
        return {"ok": True, "provider": "local-stub", "answer": "\n".join(lines), "pack_size": len(pack)}

    base = os.environ.get("TIGA_LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("TIGA_LLM_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are TIGA Einstein. Answer ONLY from the evidence pack. "
                    "Cite paths. If unknown, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "evidence_pack": pack},
                    ensure_ascii=False,
                ),
            },
        ],
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        answer = data["choices"][0]["message"]["content"]
        return {"ok": True, "provider": "api", "model": model, "answer": answer, "pack_size": len(pack)}
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
        return {"ok": False, "provider": "api", "answer": f"Ask failed: {exc}", "pack_size": len(pack)}
