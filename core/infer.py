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
from functools import lru_cache
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

@lru_cache(maxsize=4096)
def _find_yaml_for_dir(dir_path: Path, roots_tuple: tuple) -> tuple | None:
    """
    Cached inner lookup: walk up from dir_path looking for tiga_project.yaml.
    Keyed on (dir_path, roots_tuple) — same directory always returns same result.
    Returns tuple of dict items (hashable) or None.
    """
    roots = set(roots_tuple)
    check = dir_path

    for _ in range(_MAX_WALK_DEPTH):
        yaml_file = check / _OVERRIDE_FILENAME
        if yaml_file.exists():
            try:
                with yaml_file.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict) and "project_id" in data:
                    return tuple(sorted(data.items()))
            except Exception:
                pass   # corrupt YAML — keep walking

        if check in roots:
            break
        parent = check.parent
        if parent == check:   # filesystem root
            break
        check = parent

    return None


def _find_project_yaml(
    path: Path,
    cfg_obj: Config,
) -> dict[str, Any] | None:
    """
    Walk up from path.parent looking for tiga_project.yaml.
    Stop when we reach (or pass) an index_root boundary or hit _MAX_WALK_DEPTH.
    Returns parsed dict or None.
    Results are cached per directory to avoid repeated NAS stat calls.
    """
    roots_tuple = tuple(r.resolve() for r in cfg_obj.index_roots)
    result = _find_yaml_for_dir(path.parent.resolve(), roots_tuple)
    return dict(result) if result is not None else None


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

@lru_cache(maxsize=256)
def _sibling_score(project_folder: str, root_dir: Path) -> float:
    """
    Check whether >50% of peer directories in root_dir share the same
    top token as project_folder. Returns 0.15 if yes, else 0.0.
    Cached per (project_folder, root_dir) — the NAS iterdir() is called once
    per unique root, not once per file.
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
