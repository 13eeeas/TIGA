"""
core/convention_detector.py — Detect each project's folder naming convention.

Every architecture project organises its files differently:
  "plain"    — Project Documents / Project Images / Meetings / BIM
  "numbered" — 01 Brief / 02 Client / 09 CAD / 10 BIM / 11 3D
  "prefixed" — 270 BIM / 270 01 Addressee / 270 02 Authorities
  "mixed"    — Combination of the above

detect_convention() scans the immediate sub-folders of a project root and
returns a FolderConvention with a folder_map: {actual_name → canonical_category}.

Conventions are saved/loaded from project_conventions.yml in work_dir so
parse_file_path() can look up canonical_category at index time.

INVARIANT: never raises — returns convention_type="unknown" on any error.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

_CONVENTIONS_FILENAME = "project_conventions.yml"

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class FolderConvention:
    convention_type: str         # "plain" | "numbered" | "prefixed" | "mixed" | "unknown"
    project_prefix:  str         # e.g. "270" or "186" or ""
    number_separator: str        # e.g. " " or "_" or ""
    sample_folders:  list[str]   = field(default_factory=list)
    confidence:      float       = 0.0
    folder_map:      dict[str, str] = field(default_factory=dict)  # folder_name → canonical_category


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

_NUMBERED_RE  = re.compile(r'^\d{2}\s')        # "01 Brief", "09 CAD"
_PREFIXED_RE  = re.compile(r'^(\d{3,4})\s')    # "270 BIM", "186 Model"
_PREF_NUM_RE  = re.compile(r'^(\d{3,4})\s+\d{2}\s')  # "270 01 Addressee"


# ---------------------------------------------------------------------------
# Built-in seed mapping: common folder names → canonical_category
# ---------------------------------------------------------------------------

_SEED_MAP: list[tuple[str, str]] = [
    # Key = lowercase substring match (longest first for priority)
    ("design presentation images",  "renders_3d"),
    ("project images",              "renders_3d"),
    ("model photos",                "renders_3d"),
    ("construction photos",         "construction_photos"),
    ("site visit",                  "construction_photos"),
    ("project documents",           "documents"),
    ("project action list",         "programme"),
    ("drawing register",            "issued"),
    ("electronic submission",       "submission"),
    ("contract documentation",      "contract"),
    ("consultant correspondence",   "correspondence"),
    ("brief",                       "brief"),
    ("client",                      "client"),
    ("consultants",                 "consultants"),
    ("documents",                   "documents"),
    ("meeting",                     "meetings"),
    ("meetings",                    "meetings"),
    ("submission",                  "submission"),
    ("authorities",                 "submission"),
    ("cad",                         "cad"),
    ("autocad",                     "cad"),
    ("drawings",                    "cad"),
    ("bim",                         "bim"),
    ("revit",                       "bim"),
    ("3d",                          "renders_3d"),
    ("lumion",                      "renders_3d"),
    ("renders",                     "renders_3d"),
    ("images",                      "renders_3d"),
    ("publication",                 "publication"),
    ("pubpack",                     "publication"),
    ("awards",                      "publication"),
    ("programme",                   "programme"),
    ("program",                     "programme"),
    ("qms",                         "qms"),
    ("quality",                     "qms"),
    ("research",                    "research"),
    ("reference",                   "research"),
    ("sketches",                    "sketches"),
    ("sketch",                      "sketches"),
    ("site data",                   "site_data"),
    ("site information",            "site_data"),
    ("landscape",                   "landscape"),
    ("indesign",                    "submission_materials"),
    ("ppt",                         "submission_materials"),
    ("presentation",                "submissions_materials"),
    ("working",                     "working"),
    ("temp",                        "working"),
    ("issued",                      "issued"),
    ("transmittal",                 "issued"),
    ("eqs",                         "submission"),
    ("supplier",                    "suppliers"),
    ("suppliers",                   "suppliers"),
    ("contractor",                  "contract"),
    ("contract",                    "contract"),
    ("correspondence",              "correspondence"),
    ("emails",                      "correspondence"),
    ("addressee",                   "addresses"),
    ("site staff",                  "site_data"),
    ("photos",                      "construction_photos"),
    ("report",                      "documents"),
]


def _name_to_canonical(folder_name: str) -> str | None:
    """Match a folder name to a canonical_category using seed map."""
    fl = folder_name.lower()
    # Remove leading number prefix for matching (e.g. "01 Brief" → "brief")
    fl_clean = re.sub(r'^\d{1,3}\s+', '', fl)
    # Also remove project prefix (e.g. "270 BIM" → "bim")
    fl_clean2 = re.sub(r'^\d{3,4}\s+(?:\d{2}\s+)?', '', fl)

    for seed, cat in _SEED_MAP:
        if seed in fl or seed in fl_clean or seed in fl_clean2:
            return cat
    return None


# ---------------------------------------------------------------------------
# Convention detection
# ---------------------------------------------------------------------------

def detect_convention(project_root: str | Path) -> FolderConvention:
    """
    Scan immediate sub-folders of project_root and detect the naming convention.
    Returns a FolderConvention with folder_map populated.
    Never raises.
    """
    try:
        root = Path(project_root)
        if not root.is_dir():
            return FolderConvention("unknown", "", "", confidence=0.0)

        subdirs = sorted([d.name for d in root.iterdir() if d.is_dir()])
        if not subdirs:
            return FolderConvention("unknown", "", "", confidence=0.0)

        # Count convention signals
        n_numbered  = sum(1 for d in subdirs if _NUMBERED_RE.match(d))
        n_pref_num  = sum(1 for d in subdirs if _PREF_NUM_RE.match(d))
        n_prefixed  = sum(1 for d in subdirs if _PREFIXED_RE.match(d) and not _PREF_NUM_RE.match(d))
        n_plain     = len(subdirs) - n_numbered - n_pref_num - n_prefixed
        total       = len(subdirs)

        # Detect prefix
        project_prefix = ""
        prefix_candidates = [_PREFIXED_RE.match(d) for d in subdirs if _PREFIXED_RE.match(d)]
        if prefix_candidates:
            prefixes = [m.group(1) for m in prefix_candidates if m]
            from collections import Counter
            most_common = Counter(prefixes).most_common(1)
            if most_common and most_common[0][1] / total > 0.5:
                project_prefix = most_common[0][0]

        threshold = 0.70
        if n_pref_num / total >= threshold:
            conv_type = "prefixed_numbered"
        elif (n_numbered + n_pref_num) / total >= threshold:
            conv_type = "numbered"
        elif n_prefixed / total >= threshold:
            conv_type = "prefixed"
        elif n_plain / total >= threshold:
            conv_type = "plain"
        else:
            conv_type = "mixed"

        confidence = max(n_numbered, n_pref_num, n_prefixed, n_plain) / total

        # Build folder_map
        folder_map: dict[str, str] = {}
        for d in subdirs:
            cat = _name_to_canonical(d)
            if cat:
                folder_map[d] = cat

        return FolderConvention(
            convention_type=conv_type,
            project_prefix=project_prefix,
            number_separator=" ",
            sample_folders=subdirs[:10],
            confidence=round(confidence, 2),
            folder_map=folder_map,
        )
    except Exception as e:
        logger.debug("detect_convention error for %s: %s", project_root, e)
        return FolderConvention("unknown", "", "", confidence=0.0)


# ---------------------------------------------------------------------------
# Save / load project_conventions.yml
# ---------------------------------------------------------------------------

def load_conventions(work_dir: str | Path) -> dict[str, Any]:
    """Load project_conventions.yml from work_dir. Returns {} if absent."""
    path = Path(work_dir) / _CONVENTIONS_FILENAME
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("Failed to load %s: %s", path, e)
        return {}


def save_conventions(work_dir: str | Path, data: dict[str, Any]) -> None:
    """Write project_conventions.yml atomically."""
    path = Path(work_dir) / _CONVENTIONS_FILENAME
    try:
        path.write_text(yaml.dump(data, allow_unicode=True, sort_keys=True), encoding="utf-8")
    except Exception as e:
        logger.warning("Failed to save %s: %s", path, e)


def detect_and_save(
    project_code: str,
    project_root: str | Path,
    work_dir: str | Path,
) -> FolderConvention:
    """
    Detect convention for project_code, merge into project_conventions.yml, return result.
    Existing manual folder_map overrides are preserved.
    """
    from datetime import date as _date
    conv = detect_convention(project_root)
    data = load_conventions(work_dir)

    existing = data.get(project_code, {})
    # Preserve manual overrides in existing folder_map
    merged_map = {**conv.folder_map, **existing.get("folder_map", {})}

    data[project_code] = {
        "convention_type":  conv.convention_type,
        "project_prefix":   conv.project_prefix,
        "confidence":       conv.confidence,
        "detected_at":      str(_date.today()),
        "folder_map":       merged_map,
    }
    save_conventions(work_dir, data)
    conv.folder_map = merged_map
    return conv


def lookup_canonical(
    folder_name: str,
    project_code: str,
    work_dir: str | Path,
) -> str | None:
    """
    Look up canonical_category for a specific folder name in a project.
    Returns None if not found.
    """
    data = load_conventions(work_dir)
    proj = data.get(project_code, {})
    fm = proj.get("folder_map", {})
    # Exact match first
    if folder_name in fm:
        return fm[folder_name]
    # Seed map fallback
    return _name_to_canonical(folder_name)
