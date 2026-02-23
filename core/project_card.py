"""
core/project_card.py — Project Data Card management.

The project_cards table is the canonical source of truth for project identity,
scale, and stakeholder information. Fields can come from three sources:
  - "manual"    : entered via CLI / corrections.py — NEVER overwritten
  - "woha_web"  : scraped from WOHA website       — overwritten only by manual
  - "inferred"  : derived from path/filename scan  — overwritten by any source

Every field is tracked in data_sources:
  {field_name: {value, source, confidence}}

Public API
----------
  upsert_project_card(data, conn=None)    → None
  get_project_card(project_code, conn=None) → dict | None
  list_project_cards(filters=None, conn=None) → list[dict]
  search_project_cards(query, conn=None)  → list[dict]
  get_missing_fields(project_code, conn=None) → list[str]
  auto_populate_from_path(project_code, root_path, conn=None) → dict
  scrape_woha_project(url)               → dict
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from config import cfg as _module_cfg, Config

logger = logging.getLogger(__name__)

# Source priority: manual > woha_web > inferred
_SOURCE_RANK = {"manual": 3, "woha_web": 2, "inferred": 1}

# Fields that must be populated for a "complete" project card
_REQUIRED_FIELDS = [
    "name", "typology_primary", "location", "client", "stage",
    "gfa_sqm", "storeys_above",
]

# JSON-serialised fields
_JSON_FIELDS = {
    "alt_names", "program_components", "consultants",
    "authorities", "milestone_dates", "awards", "data_sources",
}

_TYPOLOGY_KEYWORDS: dict[str, list[str]] = {
    "residential": ["residential", "resi", "apartment", "condo", "housing", "hdb"],
    "hospitality": ["hotel", "hospitality", "resort", "serviced", "keys"],
    "commercial":  ["office", "commercial", "retail", "mall", "shophouse"],
    "civic":       ["civic", "community", "cultural", "library", "museum"],
    "mixed-use":   ["mixed", "mixed-use"],
    "education":   ["school", "campus", "university", "polytechnic", "sit"],
    "healthcare":  ["hospital", "clinic", "healthcare", "medical"],
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _json_decode(v: str | None, default: Any = None) -> Any:
    if v is None:
        return default
    try:
        return json.loads(v)
    except (json.JSONDecodeError, TypeError):
        return default


def _json_encode(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, str):
        return v  # already serialised (legacy)
    return json.dumps(v, ensure_ascii=False)


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    for field in _JSON_FIELDS:
        if field in d:
            d[field] = _json_decode(d[field], None)
    return d


def _get_conn(cfg_obj: Config | None = None, conn: sqlite3.Connection | None = None):
    if conn is not None:
        return conn, False
    _cfg = cfg_obj or _module_cfg
    from core.db import get_connection
    return get_connection(_cfg.get_db_path()), True


# ---------------------------------------------------------------------------
# upsert_project_card
# ---------------------------------------------------------------------------

def upsert_project_card(
    data: dict[str, Any],
    conn: sqlite3.Connection | None = None,
    cfg_obj: Config | None = None,
) -> None:
    """
    Insert or update a project card.

    Rules:
    - Manual fields (data_sources[field].source == "manual") are never overwritten.
    - New data of equal or higher source rank overwrites existing.
    - data_sources dict is merged field-by-field.
    """
    _conn, _own = _get_conn(cfg_obj, conn)
    try:
        project_code = data.get("project_code")
        if not project_code:
            raise ValueError("project_code is required")

        # Load existing card
        existing = _conn.execute(
            "SELECT * FROM project_cards WHERE project_code = ?",
            (project_code,),
        ).fetchone()

        if existing is None:
            _insert_card(_conn, data)
        else:
            _merge_card(_conn, existing, data)
    finally:
        if _own:
            _conn.close()


def _insert_card(conn: sqlite3.Connection, data: dict[str, Any]) -> None:
    """Insert a new project card row."""
    source = data.get("_source", "inferred")
    confidence = data.get("_confidence", 0.6)

    # Build data_sources from provided fields
    ds: dict[str, dict] = {}
    for field in _all_card_fields():
        if field in data and data[field] is not None and field not in (
            "project_code", "created_at", "updated_at", "data_sources"
        ):
            ds[field] = {"value": data[field], "source": source, "confidence": confidence}

    # Merge any explicit data_sources from caller
    caller_ds = data.get("data_sources") or {}
    if isinstance(caller_ds, str):
        caller_ds = _json_decode(caller_ds, {})
    ds.update(caller_ds)

    row = _build_row(data, ds)
    columns = ", ".join(row.keys())
    placeholders = ", ".join("?" * len(row))
    conn.execute(
        f"INSERT INTO project_cards ({columns}) VALUES ({placeholders})",
        list(row.values()),
    )
    conn.commit()
    logger.info("Inserted project card: %s", data.get("project_code"))


def _merge_card(
    conn: sqlite3.Connection,
    existing: sqlite3.Row,
    data: dict[str, Any],
) -> None:
    """Merge new data into existing card, respecting source priorities."""
    existing_dict = _row_to_dict(existing)
    existing_ds: dict[str, dict] = existing_dict.get("data_sources") or {}

    new_source = data.get("_source", "inferred")
    new_confidence = data.get("_confidence", 0.6)
    caller_ds = data.get("data_sources") or {}
    if isinstance(caller_ds, str):
        caller_ds = _json_decode(caller_ds, {})

    merged: dict[str, Any] = dict(existing_dict)
    merged_ds: dict[str, dict] = dict(existing_ds)

    for field in _all_card_fields():
        if field in ("project_code", "created_at", "updated_at", "data_sources"):
            continue
        new_val = data.get(field)
        if new_val is None:
            continue

        # Determine effective source for this field
        field_source = caller_ds.get(field, {}).get("source", new_source)
        field_confidence = caller_ds.get(field, {}).get("confidence", new_confidence)

        existing_source = existing_ds.get(field, {}).get("source", "inferred")
        existing_rank = _SOURCE_RANK.get(existing_source, 0)
        new_rank = _SOURCE_RANK.get(field_source, 0)

        if new_rank >= existing_rank:
            merged[field] = new_val
            merged_ds[field] = {
                "value": new_val,
                "source": field_source,
                "confidence": field_confidence,
            }

    merged["data_sources"] = merged_ds
    merged["updated_at"] = datetime.now().isoformat()

    row = _build_row(merged, merged_ds)
    row.pop("project_code", None)
    row.pop("created_at", None)
    sets = ", ".join(f"{k} = ?" for k in row)
    values = list(row.values()) + [merged["project_code"]]
    conn.execute(
        f"UPDATE project_cards SET {sets} WHERE project_code = ?",
        values,
    )
    conn.commit()
    logger.info("Merged project card: %s", merged.get("project_code"))


def _all_card_fields() -> list[str]:
    return [
        "project_code", "name", "alt_names", "typology_primary", "typology_secondary",
        "location", "client", "stage", "gfa_sqm", "site_area_sqm",
        "plot_ratio_proposed", "plot_ratio_allowable", "storeys_above", "storeys_below",
        "units", "keys", "beds", "program_components", "contract_value",
        "architect", "pm_job_captain", "consultants", "contractor", "authorities",
        "milestone_dates", "root_path", "woha_url", "concept_summary", "awards",
        "data_sources", "created_at", "updated_at",
    ]


def _build_row(data: dict[str, Any], ds: dict[str, dict]) -> dict[str, Any]:
    """Build an INSERT/UPDATE row dict, JSON-encoding array/dict fields."""
    row: dict[str, Any] = {}
    for field in _all_card_fields():
        if field == "data_sources":
            continue
        val = data.get(field)
        if field in _JSON_FIELDS:
            row[field] = _json_encode(val)
        else:
            row[field] = val
    row["data_sources"] = _json_encode(ds)
    if not row.get("created_at"):
        row["created_at"] = datetime.now().isoformat()
    row["updated_at"] = datetime.now().isoformat()
    return row


# ---------------------------------------------------------------------------
# get_project_card
# ---------------------------------------------------------------------------

def get_project_card(
    project_code: str,
    conn: sqlite3.Connection | None = None,
    cfg_obj: Config | None = None,
) -> dict[str, Any] | None:
    _conn, _own = _get_conn(cfg_obj, conn)
    try:
        row = _conn.execute(
            "SELECT * FROM project_cards WHERE project_code = ?",
            (project_code,),
        ).fetchone()
        return _row_to_dict(row) if row else None
    finally:
        if _own:
            _conn.close()


# ---------------------------------------------------------------------------
# list_project_cards
# ---------------------------------------------------------------------------

def list_project_cards(
    filters: dict[str, Any] | None = None,
    conn: sqlite3.Connection | None = None,
    cfg_obj: Config | None = None,
) -> list[dict[str, Any]]:
    """
    List project cards with optional filtering.
    Supported filters: typology, stage, gfa_min, gfa_max, location
    """
    _conn, _own = _get_conn(cfg_obj, conn)
    try:
        clauses: list[str] = []
        params: list[Any] = []

        if filters:
            if "typology" in filters:
                clauses.append("typology_primary = ?")
                params.append(filters["typology"])
            if "stage" in filters:
                clauses.append("stage = ?")
                params.append(filters["stage"])
            if "gfa_min" in filters:
                clauses.append("gfa_sqm >= ?")
                params.append(float(filters["gfa_min"]))
            if "gfa_max" in filters:
                clauses.append("gfa_sqm <= ?")
                params.append(float(filters["gfa_max"]))
            if "location" in filters:
                clauses.append("location LIKE ?")
                params.append(f"%{filters['location']}%")

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = _conn.execute(
            f"SELECT * FROM project_cards {where} ORDER BY project_code",
            params,
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        if _own:
            _conn.close()


# ---------------------------------------------------------------------------
# search_project_cards
# ---------------------------------------------------------------------------

def search_project_cards(
    query: str,
    conn: sqlite3.Connection | None = None,
    cfg_obj: Config | None = None,
) -> list[dict[str, Any]]:
    """Full-text search across name, alt_names, client, location."""
    _conn, _own = _get_conn(cfg_obj, conn)
    try:
        like = f"%{query}%"
        rows = _conn.execute(
            "SELECT * FROM project_cards "
            "WHERE name LIKE ? OR alt_names LIKE ? OR client LIKE ? OR location LIKE ? "
            "ORDER BY project_code",
            (like, like, like, like),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        if _own:
            _conn.close()


# ---------------------------------------------------------------------------
# get_missing_fields
# ---------------------------------------------------------------------------

def get_missing_fields(
    project_code: str,
    conn: sqlite3.Connection | None = None,
    cfg_obj: Config | None = None,
) -> list[str]:
    """Return list of required fields that are None or have low confidence."""
    card = get_project_card(project_code, conn=conn, cfg_obj=cfg_obj)
    if card is None:
        return _REQUIRED_FIELDS[:]

    ds = card.get("data_sources") or {}
    missing = []
    for field in _REQUIRED_FIELDS:
        val = card.get(field)
        if val is None:
            missing.append(field)
            continue
        # Flag unconfirmed inferred fields
        field_source = ds.get(field, {}).get("source", "inferred")
        if field_source == "inferred":
            confidence = ds.get(field, {}).get("confidence", 0.0)
            if confidence < 0.5:
                missing.append(f"{field} (low confidence)")
    return missing


# ---------------------------------------------------------------------------
# auto_populate_from_path
# ---------------------------------------------------------------------------

def auto_populate_from_path(
    project_code: str,
    root_path: str,
    conn: sqlite3.Connection | None = None,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    """
    Infer project card fields from folder name and structure.
    All results marked source="inferred", confidence=0.6.
    """
    root = Path(root_path)
    folder_name = root.name
    text = folder_name.lower()
    path_text = " ".join(root.parts).lower()

    data: dict[str, Any] = {
        "project_code": project_code,
        "root_path": str(root),
        "_source": "inferred",
        "_confidence": 0.6,
    }

    # project_code from leading number
    m = re.match(r'^(\d{3,4})[-_ ]', folder_name)
    if m and project_code == "Unknown":
        data["project_code"] = m.group(1)

    # Typology from keywords
    for typology, keywords in _TYPOLOGY_KEYWORDS.items():
        if any(kw in path_text for kw in keywords):
            data["typology_primary"] = typology
            break

    # Milestone dates from subfolder names
    milestone_dates: dict[str, str] = {}
    if root.exists():
        try:
            for subdir in root.iterdir():
                if not subdir.is_dir():
                    continue
                dn = subdir.name.lower()
                # IFC date
                if any(kw in dn for kw in ["ifc", "issued for construction", "for construction"]):
                    date_str = _extract_date_from_name(subdir.name)
                    if date_str:
                        milestone_dates["IFC"] = date_str
                # Tender date
                if "tender" in dn:
                    date_str = _extract_date_from_name(subdir.name)
                    if date_str:
                        milestone_dates["tender"] = date_str
        except (PermissionError, OSError):
            pass

    if milestone_dates:
        data["milestone_dates"] = milestone_dates

    # Upsert to DB
    upsert_project_card(data, conn=conn, cfg_obj=cfg_obj)
    return data


def _extract_date_from_name(name: str) -> str | None:
    """Extract ISO date from a folder/filename. Returns YYYY-MM-DD or None."""
    patterns = [
        (r'(\d{4})[-_](\d{2})[-_](\d{2})', lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"),
        (r'(\d{8})', lambda m: f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:]}"),
        (r'(\d{2})[-_](\d{2})[-_](\d{4})', lambda m: f"{m.group(3)}-{m.group(2)}-{m.group(1)}"),
    ]
    for pattern, formatter in patterns:
        m = re.search(pattern, name)
        if m:
            try:
                date_str = formatter(m)
                datetime.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# scrape_woha_project
# ---------------------------------------------------------------------------

def scrape_woha_project(url: str) -> dict[str, Any]:
    """
    Scrape a WOHA project page and return a data dict ready for upsert_project_card.
    All fields marked source="woha_web", confidence=0.85.
    Requires: requests, beautifulsoup4 (pip install requests beautifulsoup4)
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except ImportError:
        raise ImportError(
            "scrape_woha_project requires: pip install requests beautifulsoup4"
        )

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
    except Exception as e:
        logger.error("Failed to fetch WOHA URL %s: %s", url, e)
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    data: dict[str, Any] = {
        "woha_url": url,
        "_source": "woha_web",
        "_confidence": 0.85,
    }
    ds: dict[str, dict] = {}

    def _add(field: str, value: Any) -> None:
        if value:
            data[field] = value
            ds[field] = {"value": value, "source": "woha_web", "confidence": 0.85}

    # Project name — try various selectors
    name_sel = (
        soup.find("h1") or
        soup.find(class_=re.compile(r"project.?title|title|heading", re.I)) or
        soup.find("title")
    )
    if name_sel:
        raw = name_sel.get_text(strip=True)
        # Strip "WOHA – " prefix if present
        raw = re.sub(r'^WOHA\s*[–-]\s*', '', raw).strip()
        _add("name", raw)

    # Location
    for el in soup.find_all(string=re.compile(r'\bSingapore\b|\bChina\b|\bVietnam\b|\bMalaysia\b|\bIndonesia\b|\bThailand\b|\bIndia\b|\bAustralia\b', re.I)):
        parent = el.find_parent()
        if parent and parent.name in ("p", "span", "div", "li"):
            text = el.strip()
            if len(text) < 80:
                _add("location", text)
                break

    # Typology from description text
    body_text = soup.get_text(" ", strip=True).lower()
    for typology, keywords in _TYPOLOGY_KEYWORDS.items():
        if any(kw in body_text for kw in keywords):
            _add("typology_primary", typology)
            break

    # GFA — look for "XXX sqm" / "XXX m²" patterns
    gfa_match = re.search(r'([\d,]+)\s*(?:sqm|m²|m2|sq\.?\s*m)', body_text)
    if gfa_match:
        try:
            gfa = float(gfa_match.group(1).replace(",", ""))
            _add("gfa_sqm", gfa)
        except ValueError:
            pass

    # Concept / description — first substantial paragraph
    for p in soup.find_all("p"):
        text = p.get_text(strip=True)
        if len(text) > 100:
            _add("concept_summary", text[:2000])
            break

    # Completion year
    year_match = re.search(r'\b(20\d{2}|19\d{2})\b', body_text)
    if year_match:
        year = year_match.group(1)
        existing = data.get("milestone_dates") or {}
        if "completion" not in existing:
            existing["completion"] = year
            _add("milestone_dates", existing)

    # Awards — look for award text patterns
    awards = []
    for el in soup.find_all(string=re.compile(r'award|prize|winner|recognition', re.I)):
        parent = el.find_parent()
        if parent:
            text = parent.get_text(strip=True)
            if 20 < len(text) < 200:
                awards.append(text)
    if awards:
        _add("awards", awards[:10])

    data["data_sources"] = ds
    return data
