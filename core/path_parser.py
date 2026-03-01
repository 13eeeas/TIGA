"""
core/path_parser.py — Deterministic metadata extraction from file paths.

All parsing is string-matching only — fast, no LLM required.
Drives the files table's structured metadata columns.

Public API
----------
  parse_file_path(full_path, project_root) -> dict
  load_semantics(yaml_path)               -> dict  (auto-cached, auto-reloads)
  match_stage(folder_segment, semantics)  -> str | None
  update_is_latest(conn, project_code)    -> int    (rows updated)
"""

from __future__ import annotations

import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Content type map (extension → content_type)
# ---------------------------------------------------------------------------

_EXT_CONTENT_TYPE: dict[str, str] = {
    ".rvt": "BIM", ".ifc": "BIM", ".nwd": "BIM", ".nwc": "BIM", ".rfa": "BIM",
    ".dwg": "CAD", ".dxf": "CAD", ".dgn": "CAD",
    ".psd": "Photoshop", ".psb": "Photoshop",
    ".ai": "Illustrator", ".eps": "Illustrator",
    ".indd": "InDesign", ".indt": "InDesign",
    ".pptx": "Presentation", ".ppt": "Presentation", ".key": "Presentation",
    ".xlsx": "Spreadsheet", ".xls": "Spreadsheet", ".csv": "Spreadsheet",
    ".pdf": "PDF",
    ".jpg": "Image", ".jpeg": "Image", ".png": "Image",
    ".tiff": "Image", ".tif": "Image", ".bmp": "Image", ".webp": "Image",
    ".lsm": "Render Scene", ".tma": "Render Scene", ".skp": "Render Scene",
    ".max": "Render Scene", ".c4d": "Render Scene", ".blend": "Render Scene",
    ".docx": "Word", ".doc": "Word",
    ".mp4": "Video", ".mov": "Video", ".avi": "Video",
    ".txt": "Text", ".md": "Text",
    ".msg": "Email", ".eml": "Email",
}

# ---------------------------------------------------------------------------
# Discipline prefix map (filename prefix → discipline)
# ---------------------------------------------------------------------------

_DISC_PREFIX: list[tuple[re.Pattern, str]] = [
    (re.compile(r'^ID[-_]', re.I), "Interior"),
    (re.compile(r'^A[-_]',  re.I), "Architecture"),
    (re.compile(r'^S[-_]',  re.I), "Structural"),
    (re.compile(r'^M[-_]',  re.I), "Mechanical"),
    (re.compile(r'^E[-_]',  re.I), "Electrical"),
    (re.compile(r'^C[-_]',  re.I), "Civil"),
    (re.compile(r'^L[-_]',  re.I), "Landscape"),
]

# ---------------------------------------------------------------------------
# Doc type patterns (applied to filename stem, case-insensitive)
# ---------------------------------------------------------------------------

_DOC_TYPE_PATTERNS: list[tuple[re.Pattern, str, str | None]] = [
    # (pattern, doc_type, override_content_type or None)
    (re.compile(r'\bGA\b|General\s*Arrangement',      re.I), "General Arrangement", None),
    (re.compile(r'\bFP\b|Floor\s*Plan',                re.I), "Floor Plan",          None),
    (re.compile(r'\bSEC\b|Section(?:al)?(?!\s*\d)',    re.I), "Section",             None),
    (re.compile(r'\bEL\b|Elev(?:ation)?',              re.I), "Elevation",           None),
    (re.compile(r'\bDET\b|Detail',                     re.I), "Detail",              None),
    (re.compile(r'\bSCH\b|Schedule',                   re.I), "Schedule",            None),
    (re.compile(r'\bSPEC\b|Specification',             re.I), "Specification",       None),
    (re.compile(r'\bMIN\b|Minutes',                    re.I), "Minutes",             "Minutes"),
    (re.compile(r'\bTRX\b|Transmittal',                re.I), "Transmittal",         "Transmittal"),
    (re.compile(r'\bRFI\b',                            re.I), "RFI",                 "RFI"),
    (re.compile(r'Site[\s_-]*Plan',                    re.I), "Site Plan",           None),
    (re.compile(r'\bROOF\b',                           re.I), "Roof Plan",           None),
]

# ---------------------------------------------------------------------------
# Revision patterns (applied to full filename)
# ---------------------------------------------------------------------------

_REV_PATTERNS: list[re.Pattern] = [
    re.compile(r'[_\-\s]Rev[_\-]?(\d+)',  re.I),
    re.compile(r'[_\-\s][Rr](\d+)(?=[_\-\s.]|$)'),
    re.compile(r'[_\-\s][Vv](\d+)(?=[_\-\s.]|$)'),
    re.compile(r'[_\-\s][Vv]\.?(\d+)(?=[_\-\s.]|$)'),
    re.compile(r'[_\-](\d+)$'),    # trailing number after separator
]

# ---------------------------------------------------------------------------
# Date patterns (applied to filename stem and parent folder)
# ---------------------------------------------------------------------------

_DATE_PATTERNS: list[tuple[re.Pattern, str]] = [
    # YYYYMMDD
    (re.compile(r'(?<!\d)(20\d{2})(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)'),
     "{0}-{1}-{2}"),
    # YYYY-MM-DD or YYYY_MM_DD
    (re.compile(r'(?<!\d)(20\d{2})[-_](0[1-9]|1[0-2])[-_](0[1-9]|[12]\d|3[01])(?!\d)'),
     "{0}-{1}-{2}"),
    # DD-MM-YYYY or DD_MM_YYYY
    (re.compile(r'(?<!\d)(0[1-9]|[12]\d|3[01])[-_](0[1-9]|1[0-2])[-_](20\d{2})(?!\d)'),
     "{2}-{1}-{0}"),
]

# Folder prefix normaliser:
#   "272 CAD" -> "CAD"
#   "07 Submission" -> "Submission"
#   "270 04 Client" -> "Client"
_FOLDER_PREFIX_RE = re.compile(r'^\s*(?:\d{1,4}[\s._-]+){1,3}')

# ---------------------------------------------------------------------------
# Issued / superseded keywords (compiled word-boundary patterns)
# ---------------------------------------------------------------------------

def _make_keyword_pattern(keywords: list[str]) -> re.Pattern:
    """Build a single OR pattern with word boundaries for all keywords."""
    parts = [r'(?<![a-z])' + re.escape(kw) + r'(?![a-z])' for kw in keywords]
    return re.compile('|'.join(parts), re.IGNORECASE)


_ISSUED_KEYWORDS_RAW = [
    "issued", "outgoing", "to client", "transmittal", "transmittals",
    "for approval", "for review", "for construction",
    "approved", "released", "client issue", "final issue", "sent to",
    "drawing register",
]

_RECEIVED_KEYWORDS_RAW = [
    "received", "from client", "incoming", "from consultant",
]

_RECEIVED_RE = _make_keyword_pattern(_RECEIVED_KEYWORDS_RAW)

# Construction photo folder patterns (content_type override)
_CONSTRUCTION_PHOTO_RE = re.compile(
    r'construction\s+photos?|site\s+visit|completion\s+site\s+visit|'
    r'site\s+photography|construction\s+photography',
    re.IGNORECASE,
)

# Filename keywords that require OCR regardless of folder type
_OCR_FILENAME_RE = re.compile(
    r'\b(program|programme|schedule|milestone|gantt|timeline|programme|'
    r'area\s+schedule|room\s+schedule)\b',
    re.IGNORECASE,
)

# Folder date pattern: 8-digit YYYYMMDD (2000-2040) embedded anywhere in a folder name
_FOLDER_DATE_RE = re.compile(r'(?<!\d)(20[0-3]\d)(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])(?!\d)')

_SUPERSEDED_KEYWORDS_RAW = [
    "superseded", "obsolete", "void", "replaced",
    "old versions", "deprecated",
    # Single words that are common folder names (need exact word match)
    r"\bold\b", r"\barchive\b", r"\barchived\b", r"\bbackup\b",
    r"\bprev\b", r"\bprevious\b",
]

_ISSUED_RE = _make_keyword_pattern(_ISSUED_KEYWORDS_RAW)

# Superseded uses a separate pattern that includes \b word boundaries
_SUPERSEDED_RE = re.compile(
    r'(?i)\b(?:superseded|obsolete|void|replaced|old versions|deprecated|'
    r'old|archive|archived|backup|prev|previous)\b'
)


# ---------------------------------------------------------------------------
# Semantics loader (file-modification-aware cache)
# ---------------------------------------------------------------------------

_semantics_cache: dict[str, Any] = {}
_semantics_mtime: dict[str, float] = {}


def load_semantics(yaml_path: str | Path) -> dict:
    """
    Load folder_semantics.yml.
    Automatically reloads if the file has been modified since last load.
    Returns empty dict (with default structure) if file not found.
    """
    path = str(yaml_path)
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return _default_semantics()

    if path in _semantics_cache and _semantics_mtime.get(path) == mtime:
        return _semantics_cache[path]

    try:
        import yaml  # type: ignore
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _semantics_cache[path] = data
        _semantics_mtime[path] = mtime
        logger.debug("Loaded folder_semantics: %s", path)
        return data
    except Exception as e:
        logger.warning("Failed to load folder_semantics from %s: %s", path, e)
        return _default_semantics()


def _default_semantics() -> dict:
    """Minimal built-in defaults when the YAML file is absent."""
    return {
        "stage_synonyms": {
            "IFC": ["IFC", "Issued for Construction"],
            "tender": ["Tender"],
            "submission": ["Submission", "BCA", "URA"],
            "superseded": ["Superseded", "Old", "Archive"],
            "design": ["Design", "Schematic", "SD", "DD"],
            "renders": ["Renders", "Viz", "CGI"],
            "presentations": ["Presentation", "Presentations"],
            "bim": ["BIM", "Revit"],
            "cad": ["CAD", "Drawings"],
        },
        "issued_synonyms": ["Issued", "Outgoing", "To Client"],
        "discipline_folders": {
            "Architecture": ["Arch", "Architecture"],
            "Structural": ["Struct", "Structural"],
            "Mechanical": ["Mech", "Mechanical"],
            "Electrical": ["Elec", "Electrical"],
        },
    }


# ---------------------------------------------------------------------------
# match_stage
# ---------------------------------------------------------------------------

def match_stage(folder_segment: str, semantics: dict) -> str | None:
    """
    Check a folder name against all stage_synonyms.
    Returns the canonical stage key if matched, None otherwise.

    Short synonyms (≤ 4 chars) require word-boundary matching to avoid
    false positives (e.g. "ITT" matching inside "transmittals").
    Long synonyms use substring matching.
    """
    stage_map: dict[str, list[str]] = semantics.get("stage_synonyms", {})
    seg_lower = folder_segment.lower()

    best: tuple[int, str] | None = None
    for stage, synonyms in stage_map.items():
        for syn in synonyms:
            syn_l = syn.lower()
            length = len(syn_l)
            matched = False

            if length <= 4:
                # Require word-boundary for short abbreviations
                if re.search(r'(?<![a-zA-Z])' + re.escape(syn_l) + r'(?![a-zA-Z])', seg_lower):
                    matched = True
            else:
                # Substring match for longer synonyms
                if syn_l in seg_lower:
                    matched = True

            if matched and (best is None or length > best[0]):
                best = (length, stage)

    return best[1] if best else None


# ---------------------------------------------------------------------------
# parse_file_path  (main public function)
# ---------------------------------------------------------------------------

def _disc_name_matches(seg_l: str, names: list[str]) -> bool:
    """
    Check if a folder segment matches any discipline name.
    - Long names (> 3 chars): substring match
    - Short abbreviations (≤ 3 chars): word-boundary match to avoid false positives
      (e.g. "LS" would false-match inside "transmittals")
    """
    for name in names:
        n = name.lower()
        if len(n) <= 3:
            # Require word boundary for short abbreviations
            if re.search(r'(?<![a-zA-Z])' + re.escape(n) + r'(?![a-zA-Z])', seg_l):
                return True
            # Also: full segment is exactly the name
            if seg_l == n:
                return True
        else:
            if n in seg_l or seg_l in n:
                return True
    return False


def _normalise_folder_segment(seg: str) -> str:
    """Remove common numeric/prefix tokens used in archive folder conventions."""
    s = seg.strip()
    s = _FOLDER_PREFIX_RE.sub("", s)
    return " ".join(s.split())


def parse_file_path(
    full_path: str,
    project_root: str,
    semantics: dict | None = None,
) -> dict[str, Any]:
    """
    Parse all structured metadata from a file path.

    Args:
        full_path:    Absolute path to the file.
        project_root: Root path for this project (for relative path calculation).
        semantics:    Pre-loaded semantics dict; loads from cfg.work_dir if None.

    Returns:
        dict with keys matching the files table's new columns:
          project_code, folder_stage, discipline, doc_type, revision,
          file_date, content_type, is_issued, is_superseded
    """
    if semantics is None:
        try:
            from config import cfg as _cfg
            sem_path = _cfg.work_dir / "folder_semantics.yml"
            semantics = load_semantics(sem_path)
        except Exception:
            semantics = _default_semantics()

    path = Path(full_path)
    root = Path(project_root)

    # Relative path parts (folder segments between root and file)
    try:
        rel = path.relative_to(root)
        parts = list(rel.parts)
    except ValueError:
        parts = list(path.parts)

    folder_parts = parts[:-1]  # all but the filename
    filename = path.name
    stem = path.stem
    ext = path.suffix.lower()

    result: dict[str, Any] = {
        "project_code": None,
        "folder_stage": "unknown",
        "discipline": "unknown",
        "doc_type": "unknown",
        "revision": None,
        "file_date": None,
        "folder_date": None,
        "content_type": "unknown",
        "is_issued": 0,
        "is_superseded": 0,
        "is_received": 0,
        "canonical_category": None,
        # is_latest is computed in bulk via update_is_latest()
    }

    # ── Content type ──────────────────────────────────────────────────────────
    result["content_type"] = _EXT_CONTENT_TYPE.get(ext, "unknown")

    # ── Folder stage (scan all parts, use deepest/most specific match) ────────
    stage_matches: list[tuple[int, str]] = []  # (depth, stage)
    folder_date_candidates: list[str] = []

    normalised_parts = [_normalise_folder_segment(seg) for seg in folder_parts]

    for i, seg in enumerate(folder_parts):
        seg_norm = normalised_parts[i]

        # Try both raw and normalised forms for robust convention matching
        stage = match_stage(seg, semantics) or match_stage(seg_norm, semantics)
        if stage:
            stage_matches.append((i, stage))
        # Issued / superseded / received flags
        if _ISSUED_RE.search(seg) or _ISSUED_RE.search(seg_norm):
            result["is_issued"] = 1
        if _SUPERSEDED_RE.search(seg) or _SUPERSEDED_RE.search(seg_norm):
            result["is_superseded"] = 1
        if _RECEIVED_RE.search(seg) or _RECEIVED_RE.search(seg_norm):
            result["is_received"] = 1
        # Construction photo folder detection
        if _CONSTRUCTION_PHOTO_RE.search(seg) or _CONSTRUCTION_PHOTO_RE.search(seg_norm):
            if ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp", ".webp", ".heic"):
                result["content_type"] = "Construction Photo"
            elif ext in (".mp4", ".mov", ".avi", ".mkv"):
                result["content_type"] = "Construction Video"
        # Folder date extraction: YYYYMMDD embedded in folder name
        m = _FOLDER_DATE_RE.search(seg)
        if m:
            try:
                from datetime import datetime as _dt
                iso = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                _dt.strptime(iso, "%Y-%m-%d")   # validate
                folder_date_candidates.append(iso)
            except ValueError:
                pass

    # Use the deepest stage match (most specific)
    if stage_matches:
        result["folder_stage"] = stage_matches[-1][1]
        # Superseded stage implies is_superseded
        if result["folder_stage"] == "superseded":
            result["is_superseded"] = 1

    # Use the deepest (most specific) folder date
    if folder_date_candidates:
        result["folder_date"] = folder_date_candidates[-1]

    # ── Discipline — from filename prefix first, then folder names ────────────
    for pattern, discipline in _DISC_PREFIX:
        if pattern.match(filename):
            result["discipline"] = discipline
            break

    if result["discipline"] == "unknown":
        disc_folders: dict[str, list[str]] = semantics.get("discipline_folders", {})
        for seg, seg_norm in zip(folder_parts, normalised_parts):
            seg_l = seg.lower()
            seg_nl = seg_norm.lower()
            for discipline, names in disc_folders.items():
                if _disc_name_matches(seg_l, names) or _disc_name_matches(seg_nl, names):
                    result["discipline"] = discipline
                    break
            if result["discipline"] != "unknown":
                break

    # ── Doc type ─────────────────────────────────────────────────────────────
    for pattern, doc_type, override_ct in _DOC_TYPE_PATTERNS:
        if pattern.search(stem) or pattern.search(filename):
            result["doc_type"] = doc_type
            if override_ct:
                result["content_type"] = override_ct
            break

    # ── Revision ─────────────────────────────────────────────────────────────
    for pat in _REV_PATTERNS:
        m = pat.search(stem)
        if m:
            try:
                result["revision"] = int(m.group(1))
            except (ValueError, IndexError):
                pass
            break

    # ── File date ─────────────────────────────────────────────────────────────
    # 1. Try filename
    date_str = _extract_date(stem)
    # 2. Try immediate parent folder name
    if not date_str and folder_parts:
        date_str = _extract_date(folder_parts[-1])
    # 3. Fall back to folder_date if nothing else found
    if not date_str and result["folder_date"]:
        date_str = result["folder_date"]
    result["file_date"] = date_str

    # ── OCR flag from filename (regardless of folder type) ───────────────────
    # Images with "program/schedule/milestone" in name need OCR even outside OCR folders
    if ext in (".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp") and _OCR_FILENAME_RE.search(stem):
        result["needs_ocr"] = True   # picked up by image_indexing phase

    # ── Canonical category assignment ─────────────────────────────────────────
    result["canonical_category"] = _assign_canonical_category(
        folder_parts, normalised_parts, result["folder_stage"], result["content_type"], semantics
    )

    return result


def _assign_canonical_category(
    folder_parts: list[str],
    normalised_parts: list[str],
    folder_stage: str,
    content_type: str,
    semantics: dict,
) -> str | None:
    """
    Assign a canonical_category from the canonical_categories section of folder_semantics.yml.
    Resolution order:
      1. Direct folder name match against canonical_categories[cat].folder_names
      2. Fall back to folder_stage → canonical_category mapping
    Returns None if no match found.
    """
    canonical = semantics.get("canonical_categories", {})
    if not canonical:
        # Built-in fallback: map folder_stage to canonical names
        return _STAGE_TO_CANONICAL.get(folder_stage)

    # Check all folder segments (deepest match wins)
    best_cat: str | None = None
    for seg, seg_norm in zip(folder_parts, normalised_parts):
        seg_l = seg.lower()
        seg_nl = seg_norm.lower()
        for cat_name, cat_data in canonical.items():
            if not isinstance(cat_data, dict):
                continue
            for fname in cat_data.get("folder_names", []):
                fn_l = fname.lower()
                if len(fn_l) > 3:
                    if fn_l in seg_l or seg_l in fn_l or fn_l in seg_nl or seg_nl in fn_l:
                        best_cat = cat_name
                else:
                    if (
                        re.search(r'(?<![a-zA-Z])' + re.escape(fn_l) + r'(?![a-zA-Z])', seg_l)
                        or re.search(r'(?<![a-zA-Z])' + re.escape(fn_l) + r'(?![a-zA-Z])', seg_nl)
                    ):
                        best_cat = cat_name

    if best_cat:
        return best_cat

    # Fall back to stage → canonical
    return _STAGE_TO_CANONICAL.get(folder_stage)


# Stage → canonical_category fallback map
_STAGE_TO_CANONICAL: dict[str, str] = {
    "renders":          "renders_3d",
    "construction_photos": "construction_photos",
    "bim":              "bim",
    "cad":              "cad",
    "IFC":              "issued",
    "tender":           "issued",
    "submission":       "submission",
    "design":           "working",
    "presentations":    "submission_materials",
    "meetings":         "meetings",
    "correspondence":   "correspondence",
    "superseded":       "working",
    "programme":        "programme",
    "qms":              "qms",
    "publication":      "publication",
    "consultants_folder": "consultants",
    "photos":           "construction_photos",
    "reports":          "documents",
    "references":       "research",
}


def _extract_date(text: str) -> str | None:
    """Extract an ISO date string from arbitrary text."""
    for pattern, fmt in _DATE_PATTERNS:
        m = pattern.search(text)
        if m:
            groups = m.groups()
            try:
                from datetime import datetime as _dt
                date_str = fmt.format(*groups)
                _dt.strptime(date_str, "%Y-%m-%d")
                return date_str
            except ValueError:
                continue
    return None


# ---------------------------------------------------------------------------
# update_is_latest  (batch SQL update)
# ---------------------------------------------------------------------------

def update_is_latest(conn: sqlite3.Connection, project_code: str | None = None) -> int:
    """
    For each (project_code, discipline, doc_type) group, mark the file with
    the highest revision (or most recent file_date) as is_latest=1,
    all others is_latest=0.

    If project_code is given, only updates that project. Otherwise updates all.
    Returns number of files marked is_latest=1.
    """
    # Reset all
    if project_code:
        conn.execute(
            "UPDATE files SET is_latest = 0 WHERE project_code = ?",
            (project_code,),
        )
    else:
        conn.execute("UPDATE files SET is_latest = 0")

    # Find groups
    where = "WHERE project_code = ?" if project_code else "WHERE project_code IS NOT NULL"
    params = [project_code] if project_code else []

    rows = conn.execute(
        f"""
        SELECT file_id, project_code, discipline, doc_type,
               COALESCE(revision, -1) AS rev,
               COALESCE(file_date, '') AS fdate,
               file_name
        FROM files
        {where}
        AND discipline != 'unknown'
        AND doc_type != 'unknown'
        ORDER BY project_code, discipline, doc_type,
                 COALESCE(revision, -1) DESC,
                 COALESCE(file_date, '') DESC
        """,
        params,
    ).fetchall()

    seen: set[tuple[str, str, str]] = set()
    latest_ids: list[str] = []

    for row in rows:
        key = (str(row["project_code"]), str(row["discipline"]), str(row["doc_type"]))
        if key not in seen:
            seen.add(key)
            latest_ids.append(row["file_id"])

    if latest_ids:
        conn.executemany(
            "UPDATE files SET is_latest = 1 WHERE file_id = ?",
            [(fid,) for fid in latest_ids],
        )

    conn.commit()
    logger.info("update_is_latest: marked %d files as latest", len(latest_ids))
    return len(latest_ids)
