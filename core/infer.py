"""
core/infer.py — Project name, building typology, and title inference.

Rule-based + config-driven. No LLM. Deterministic.
Fail-safe: always returns a string (never raises).
"""

from __future__ import annotations

import re
from pathlib import Path

from config import cfg


# ---------------------------------------------------------------------------
# Project inference
# ---------------------------------------------------------------------------

def infer_project(path: Path) -> str:
    """
    Walk up the path to find which index_root it lives under, then return
    the immediate subdirectory name as the project.

    Applies regex patterns + keyword boosts from project_inference config.
    Falls back to parent directory name, then 'Unknown'.
    """
    try:
        resolved = path.resolve()

        for root in cfg.index_roots:
            try:
                rel = resolved.relative_to(root.resolve())
            except ValueError:
                continue

            if not rel.parts:
                # File is directly in the root — use root name
                return root.name or "Unknown"

            project_folder = rel.parts[0]

            if not cfg.project_inference_enable:
                return project_folder

            # Score confidence
            confidence = 0.0

            for pat_cfg in cfg.project_patterns:
                try:
                    if re.search(pat_cfg["regex"], project_folder):
                        confidence += float(pat_cfg.get("weight", 0.5))
                        break
                except re.error:
                    pass

            # Keyword boosts from the file/folder names in path
            path_text = " ".join(resolved.parts)
            for boost in cfg.project_keyword_boosts:
                if boost["keyword"].lower() in path_text.lower():
                    confidence += float(boost.get("weight", 0.1))

            # Return regardless of confidence — Unknown only if folder name is empty
            return project_folder or "Unknown"

        # Not under any index_root — use immediate parent
        return path.parent.name or "Unknown"

    except Exception:
        return "Unknown"


# ---------------------------------------------------------------------------
# Typology inference (building / project type)
# ---------------------------------------------------------------------------

def infer_typology(path: Path) -> str:
    """
    Infer architectural typology (healthcare, education, etc.)
    from path components using keyword_map from typology_inference config.
    Returns the best-matching typology label or 'Unknown'.
    """
    try:
        keyword_map = cfg.typology_keyword_map
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
# Title inference
# ---------------------------------------------------------------------------

def infer_title(path: Path) -> str:
    """
    Generate a readable title from the filename stem.
    Strips leading numbering prefixes; normalises separators.
    """
    try:
        stem = path.stem
        # Strip leading codes: "01_", "001 - ", "Rev.A ", "A-01 "
        stem = re.sub(r"^[\w]{0,6}[-_.\s]+", "", stem).strip()
        # Normalise separators
        stem = re.sub(r"[-_]+", " ", stem).strip()
        return stem if stem else path.stem
    except Exception:
        return path.stem
