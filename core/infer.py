"""
core/infer.py — Project ID, typology, and title inference.

infer_project() is the primary deliverable:
  Returns (project_id, confidence, signals) — never raises.

Manual override:
  Drop tiga_project.yaml in any folder to assert project identity.
  Beats all inference. Used for POC test directories.
  Format:
    project_id: 2023_MARINA_TOWERS
    typology: residential

Four-step pipeline (only reached if no override found):
  1. Regex patterns from cfg.project_inference.patterns
  2. Keyword boosts from cfg.project_inference.keyword_boosts
  3. Sibling frequency heuristic
  4. Confidence threshold → 'Unknown' if below

INVARIANT: never raises. All logic wrapped in try/except.
  On any exception: return ('Unknown', 0.0, ['error'])
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import yaml

from config import cfg as _module_cfg, Config

logger = logging.getLogger(__name__)

_OVERRIDE_FILENAME = "tiga_project.yaml"
_MAX_WALK_DEPTH = 12   # safety limit for ancestor walk


# ---------------------------------------------------------------------------
# tiga_project.yaml override
# ---------------------------------------------------------------------------

def _find_project_yaml(
    path: Path,
    cfg_obj: Config,
) -> dict[str, Any] | None:
    """
    Walk up from path.parent looking for tiga_project.yaml.
    Stop when we reach (or pass) an index_root boundary or hit _MAX_WALK_DEPTH.
    Returns parsed dict or None.
    """
    roots = {r.resolve() for r in cfg_obj.index_roots}
    check = path.parent.resolve()

    for _ in range(_MAX_WALK_DEPTH):
        yaml_file = check / _OVERRIDE_FILENAME
        if yaml_file.exists():
            try:
                with yaml_file.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and "project_id" in data:
                    return data
            except Exception:
                pass   # corrupt YAML — keep walking

        if check in roots:
            break
        parent = check.parent
        if parent == check:   # filesystem root
            break
        check = parent

    return None


# ---------------------------------------------------------------------------
# Token builder
# ---------------------------------------------------------------------------

def _build_tokens(path: Path) -> list[str]:
    """
    Nearest 3 ancestor folder names + file stem split on [-_. ].
    All returned lowercase.
    """
    ancestors = [p.name.lower() for p in list(path.parents)[:3] if p.name]
    stem_tokens = [t.lower() for t in re.split(r"[-_.\s]+", path.stem) if t]
    return ancestors + stem_tokens


# ---------------------------------------------------------------------------
# Sibling frequency heuristic
# ---------------------------------------------------------------------------

def _sibling_score(project_folder: str, root_dir: Path) -> float:
    """
    Check whether >50% of peer directories in root_dir share the same
    top token as project_folder. Returns 0.15 if yes, else 0.0.
    """
    top_token = re.split(r"[-_.\s]+", project_folder)[0].lower()
    if not top_token:
        return 0.0
    try:
        siblings = [d for d in root_dir.iterdir() if d.is_dir()]
        if not siblings:
            return 0.0
        matching = sum(
            1 for s in siblings
            if re.split(r"[-_.\s]+", s.name)[0].lower() == top_token
        )
        return 0.15 if matching / len(siblings) > 0.5 else 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# infer_project
# ---------------------------------------------------------------------------

def infer_project(
    path: Path,
    cfg_obj: Config | None = None,
) -> tuple[str, float, list[str]]:
    """
    Infer the project ID for a file.

    Returns:
        (project_id, confidence, signals)
        confidence: 0.0–1.0
        signals: list of strings describing what contributed

    Never raises — on any exception returns ('Unknown', 0.0, ['error']).
    """
    try:
        _cfg = cfg_obj or _module_cfg

        # --- Manual override ---
        override = _find_project_yaml(path, _cfg)
        if override:
            return (override["project_id"], 1.0, ["manual_override"])

        # --- Identify project folder (first subdir under any index_root) ---
        resolved = path.resolve()
        project_folder: str | None = None
        root_dir: Path | None = None

        for root in _cfg.index_roots:
            try:
                rel = resolved.relative_to(root.resolve())
                root_dir = root.resolve()
                project_folder = rel.parts[0] if rel.parts else (root.name or "Unknown")
                break
            except ValueError:
                continue

        if project_folder is None:
            project_folder = path.parent.name or "Unknown"

        # --- Build tokens ---
        tokens = _build_tokens(path)

        confidence = 0.0
        signals: list[str] = []

        # --- Step 1: Regex patterns ---
        if _cfg.project_inference_enable:
            for pat in _cfg.project_patterns:
                try:
                    if re.search(pat["regex"], project_folder, re.IGNORECASE):
                        w = float(pat.get("weight", 0.5))
                        confidence += w
                        signals.append(f"pattern:{pat.get('name', 'unnamed')}")
                except (re.error, KeyError, TypeError):
                    pass

        # --- Step 2: Keyword boosts ---
        for boost in _cfg.project_keyword_boosts:
            try:
                kw = boost["keyword"].lower()
                if any(kw in t for t in tokens):
                    w = float(boost.get("weight", 0.1))
                    confidence += w
                    signals.append(f"keyword:{boost['keyword']}")
            except (KeyError, TypeError):
                pass

        # --- Step 3: Sibling frequency heuristic ---
        if root_dir and project_folder:
            sib_score = _sibling_score(project_folder, root_dir)
            if sib_score > 0:
                confidence += sib_score
                signals.append("sibling_heuristic")

        # --- Step 4: Threshold ---
        confidence = min(confidence, 1.0)
        threshold = _cfg.project_confidence_threshold

        if confidence < threshold:
            return ("Unknown", confidence, signals)

        return (project_folder, confidence, signals)

    except Exception as e:
        logger.exception("infer_project failed for %s: %s", path, e)
        return ("Unknown", 0.0, ["error"])


# ---------------------------------------------------------------------------
# infer_typology
# ---------------------------------------------------------------------------

def infer_typology(
    path: Path,
    cfg_obj: Config | None = None,
) -> str:
    """
    Infer architectural typology (healthcare, education, etc.).
    Checks tiga_project.yaml override first, then keyword map.
    Returns 'Unknown' on failure — never raises.
    """
    try:
        _cfg = cfg_obj or _module_cfg

        # Manual override
        override = _find_project_yaml(path, _cfg)
        if override and "typology" in override:
            return override["typology"]

        keyword_map = _cfg.typology_keyword_map
        if not keyword_map:
            return "Unknown"

        path_text = " ".join(path.parts).lower()
        best_type = "Unknown"
        best_score = 0

        for typology, keywords in keyword_map.items():
            score = sum(1 for kw in keywords if kw.lower() in path_text)
            if score > best_score:
                best_score = score
                best_type = typology

        return best_type if best_score > 0 else "Unknown"

    except Exception:
        return "Unknown"


# ---------------------------------------------------------------------------
# infer_title
# ---------------------------------------------------------------------------

def infer_title(path: Path) -> str:
    """
    Generate a readable title from the filename stem.
    Strips leading numbering prefixes; normalises separators.
    Never raises.
    """
    try:
        stem = path.stem
        stem = re.sub(r"^[\w]{0,6}[-_.\s]+", "", stem).strip()
        stem = re.sub(r"[-_]+", " ", stem).strip()
        return stem if stem else path.stem
    except Exception:
        return path.stem


# ---------------------------------------------------------------------------
# Chunk 11 — Project name aliases & resolution
# ---------------------------------------------------------------------------

import difflib as _difflib
import sqlite3 as _sqlite3
from dataclasses import dataclass as _dc


@_dc
class ProjectMatch:
    project_code: str
    confidence:   float
    match_method: str


_COMMON_WORDS = frozenset([
    "the", "at", "of", "and", "&", "a", "an", "in", "on", "by", "for",
    "project", "building", "tower", "centre", "center", "complex",
])


def generate_aliases(project_code: str, project_name: str) -> list[str]:
    """
    Generate abbreviations and common variants for a project name.

    Examples:
        "Kampung Admiralty" → ["KA", "Kampung", "Admiralty", "KampungAdmiralty"]
        "Marina One"        → ["M1", "MO", "Marina"]
    """
    if not project_name:
        return []

    aliases: list[str] = []
    # Split name into significant words
    words = [w for w in re.split(r"[\s\-]+", project_name) if w and w.lower() not in _COMMON_WORDS]

    # Initialism: first letter of each word
    if len(words) >= 2:
        initialism = "".join(w[0].upper() for w in words)
        aliases.append(initialism)

    # Individual significant words
    for w in words:
        if len(w) >= 3:
            aliases.append(w)

    # CamelCase concatenation
    if len(words) >= 2:
        camel = "".join(w.capitalize() for w in words)
        aliases.append(camel)

    # Project code number variants (e.g. "261" already in project_code — add it)
    m = re.search(r"\d{3,4}", project_code)
    if m:
        num = m.group()
        if num not in aliases:
            aliases.append(num)

    # Deduplicate, case-insensitive
    seen: set[str] = set()
    result: list[str] = []
    for a in aliases:
        if a.lower() not in seen and len(a) >= 2:
            seen.add(a.lower())
            result.append(a)

    return result


def add_alias(
    conn: _sqlite3.Connection,
    project_code: str,
    alias: str,
    alias_type: str = "manual",
    source: str = "manual",
) -> None:
    """Add a single alias entry (idempotent via UNIQUE constraint)."""
    try:
        conn.execute(
            "INSERT OR IGNORE INTO project_aliases (project_code, alias, alias_type, source) "
            "VALUES (?, ?, ?, ?)",
            (project_code, alias, alias_type, source),
        )
        conn.commit()
    except Exception as e:
        logger.warning("add_alias failed for %s/%s: %s", project_code, alias, e)


def generate_aliases_for_all(conn: _sqlite3.Connection) -> int:
    """Re-generate inferred aliases for all projects in project_cards table."""
    try:
        cards = conn.execute(
            "SELECT project_code, name FROM project_cards WHERE name IS NOT NULL"
        ).fetchall()
    except Exception:
        return 0

    added = 0
    for card in cards:
        code = card["project_code"]
        name = card["name"] or ""
        for alias in generate_aliases(code, name):
            try:
                conn.execute(
                    "INSERT OR IGNORE INTO project_aliases "
                    "(project_code, alias, alias_type, source) VALUES (?, ?, ?, ?)",
                    (code, alias, "abbreviation", "inferred"),
                )
                added += 1
            except Exception:
                pass
    conn.commit()
    return added


def resolve_project_from_path(
    file_path: str | Path,
    conn: _sqlite3.Connection | None = None,
    cfg_obj=None,
) -> ProjectMatch | None:
    """
    Resolve project from file path using layered matching.

    Resolution order (first match wins):
      1. PATH ROOT MATCH (confidence=1.0) — known root_path from project_cards
      2. PROJECT NUMBER IN PATH (confidence=0.95) — 3-4 digit number in path
      3. ALIAS MATCH (confidence=0.85) — project_aliases table
      4. FUZZY NAME MATCH (confidence=0.70) — SequenceMatcher on path segments

    Returns None (+ logs to unmatched_files.log) if no match found.
    Never raises.
    """
    try:
        path = Path(file_path)
        path_str = str(path).replace("\\", "/")

        _cfg = cfg_obj
        if _cfg is None:
            from config import cfg
            _cfg = cfg

        _own_conn = conn is None
        if conn is None:
            from core.db import get_connection
            conn = get_connection(_cfg.get_db_path())

        try:
            # ── 1. Root path match ─────────────────────────────────────────────
            try:
                roots = conn.execute(
                    "SELECT project_code, root_path FROM project_cards "
                    "WHERE root_path IS NOT NULL"
                ).fetchall()
                for r in roots:
                    rp = str(r["root_path"]).replace("\\", "/")
                    if path_str.startswith(rp):
                        return ProjectMatch(r["project_code"], 1.0, "root_path")
            except Exception:
                pass

            # ── 2. Project number in path ──────────────────────────────────────
            parts = path.parts
            try:
                known_codes = conn.execute(
                    "SELECT project_code FROM project_cards WHERE project_code IS NOT NULL"
                ).fetchall()
                known_set = {r["project_code"] for r in known_codes}
                for part in parts:
                    m = re.search(r"\b(\d{3,4})\b", part)
                    if m and m.group(1) in known_set:
                        return ProjectMatch(m.group(1), 0.95, "project_number")
            except Exception:
                pass

            # ── 3. Alias match ─────────────────────────────────────────────────
            try:
                aliases = conn.execute(
                    "SELECT project_code, alias FROM project_aliases"
                ).fetchall()
                # Sort by alias length (longer aliases match more specifically)
                aliases_sorted = sorted(aliases, key=lambda r: -len(r["alias"]))
                path_lower = path_str.lower()
                for a in aliases_sorted:
                    alias_l = a["alias"].lower()
                    if len(alias_l) >= 3 and alias_l in path_lower:
                        return ProjectMatch(a["project_code"], 0.85, "alias_match")
            except Exception:
                pass

            # ── 4. Fuzzy name match ────────────────────────────────────────────
            try:
                cards = conn.execute(
                    "SELECT project_code, name FROM project_cards WHERE name IS NOT NULL"
                ).fetchall()
                for part in parts[1:]:   # skip drive letter
                    part_lower = part.lower()
                    for card in cards:
                        name_l = (card["name"] or "").lower()
                        if not name_l:
                            continue
                        ratio = _difflib.SequenceMatcher(None, part_lower, name_l).ratio()
                        if ratio > 0.80:
                            return ProjectMatch(card["project_code"], 0.70, "fuzzy_name")
            except Exception:
                pass

            # No match — log to unmatched_files.log
            _log_unmatched(file_path, _cfg)
            return None

        finally:
            if _own_conn:
                conn.close()

    except Exception as e:
        logger.debug("resolve_project_from_path error: %s", e)
        return None


def _log_unmatched(file_path: str | Path, cfg_obj=None) -> None:
    """Append to unmatched_files.log."""
    try:
        _cfg = cfg_obj
        if _cfg is None:
            from config import cfg
            _cfg = cfg
        log_dir = _cfg.work_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "unmatched_files.log", "a", encoding="utf-8") as f:
            f.write(f"{file_path}\n")
    except Exception:
        pass
