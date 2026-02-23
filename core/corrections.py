"""
core/corrections.py — User correction system for project data.

Corrections are saved to {work_dir}/corrections.json.
When 3+ corrections show the same pattern, a generalised rule is written
to folder_semantics.yml automatically.

Format (corrections.json):
  [
    {
      "project_code": "186",
      "field": "client",
      "wrong": null,
      "correct": "Tianmu Real Estate",
      "source_path": null,
      "timestamp": "2025-01-01T12:00:00",
      "applied": false
    },
    ...
  ]

Public API
----------
  save_correction(project_code, field, correct_value, wrong_value=None,
                  source_path=None, work_dir=None)
  apply_corrections(project_code, conn=None, work_dir=None)
  extract_patterns_from_corrections(work_dir=None)
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path

from config import cfg as _module_cfg, Config

logger = logging.getLogger(__name__)

_CORRECTIONS_FILE = "corrections.json"
_PATTERN_THRESHOLD = 3   # min corrections before auto-generating a rule


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _corrections_path(work_dir: Path | None = None) -> Path:
    d = work_dir or _module_cfg.work_dir
    return Path(d) / _CORRECTIONS_FILE


def _load_corrections(work_dir: Path | None = None) -> list[dict]:
    path = _corrections_path(work_dir)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load corrections file: %s", e)
        return []


def _save_corrections(corrections: list[dict], work_dir: Path | None = None) -> None:
    path = _corrections_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(corrections, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# save_correction
# ---------------------------------------------------------------------------

def save_correction(
    project_code: str,
    field: str,
    correct_value: object,
    wrong_value: object = None,
    source_path: str | None = None,
    work_dir: Path | None = None,
) -> None:
    """
    Save a manual field correction.

    The correction is immediately applied to the project card (source="manual")
    so it takes effect right away. It is also recorded in corrections.json for
    the pattern-learning pass.
    """
    corrections = _load_corrections(work_dir)

    entry = {
        "project_code": project_code,
        "field": field,
        "wrong": wrong_value,
        "correct": correct_value,
        "source_path": source_path,
        "timestamp": datetime.now().isoformat(),
        "applied": False,
    }
    corrections.append(entry)
    _save_corrections(corrections, work_dir)
    logger.info("Saved correction: %s.%s → %r", project_code, field, correct_value)

    # Immediately apply to project card (source="manual" so it won't be overwritten)
    try:
        from core.project_card import upsert_project_card
        upsert_project_card({
            "project_code": project_code,
            field: correct_value,
            "_source": "manual",
            "_confidence": 1.0,
            "data_sources": {
                field: {"value": correct_value, "source": "manual", "confidence": 1.0}
            },
        })
        # Mark applied
        corrections[-1]["applied"] = True
        _save_corrections(corrections, work_dir)
    except Exception as e:
        logger.error("Failed to apply correction to project card: %s", e)


# ---------------------------------------------------------------------------
# apply_corrections
# ---------------------------------------------------------------------------

def apply_corrections(
    project_code: str,
    conn=None,
    work_dir: Path | None = None,
) -> int:
    """
    Re-apply all unapplied corrections for a project.
    Returns count of corrections applied.
    """
    corrections = _load_corrections(work_dir)
    pending = [
        c for c in corrections
        if c["project_code"] == project_code and not c.get("applied", False)
    ]

    if not pending:
        return 0

    from core.project_card import upsert_project_card
    applied_count = 0
    for corr in pending:
        try:
            upsert_project_card({
                "project_code": project_code,
                corr["field"]: corr["correct"],
                "_source": "manual",
                "_confidence": 1.0,
                "data_sources": {
                    corr["field"]: {
                        "value": corr["correct"],
                        "source": "manual",
                        "confidence": 1.0,
                    }
                },
            }, conn=conn)
            corr["applied"] = True
            applied_count += 1
        except Exception as e:
            logger.error("Failed to re-apply correction %s.%s: %s",
                         project_code, corr["field"], e)

    _save_corrections(corrections, work_dir)
    logger.info("Applied %d corrections for project %s", applied_count, project_code)
    return applied_count


# ---------------------------------------------------------------------------
# extract_patterns_from_corrections
# ---------------------------------------------------------------------------

def extract_patterns_from_corrections(
    work_dir: Path | None = None,
) -> list[str]:
    """
    Analyse corrections.json for recurring patterns.
    When 3+ corrections fix the same (field, pattern), generate a rule
    and append it to folder_semantics.yml.

    Returns list of rule descriptions that were generated.
    """
    corrections = _load_corrections(work_dir)
    if not corrections:
        return []

    # Group by (field, correct_value) — look for repeated manual fixes
    counter: Counter = Counter()
    for c in corrections:
        if c.get("applied") and c.get("correct"):
            key = (c["field"], str(c["correct"]))
            counter[key] += 1

    generated: list[str] = []
    semantics_path = (work_dir or _module_cfg.work_dir) / "folder_semantics.yml"

    for (field, correct_val), count in counter.items():
        if count < _PATTERN_THRESHOLD:
            continue

        # Only auto-generate folder_stage rules from stage field corrections
        if field != "folder_stage":
            continue

        # Check if this stage value is already in semantics
        if semantics_path.exists():
            existing = semantics_path.read_text(encoding="utf-8")
            if correct_val in existing:
                continue  # already covered

        # Collect the wrong values (folder names) that triggered this correction
        wrong_vals = [
            c["wrong"] for c in corrections
            if c["field"] == field
            and str(c["correct"]) == correct_val
            and c.get("wrong")
        ]

        if not wrong_vals:
            continue

        # Append to semantics file
        try:
            lines = [f"\n  # Auto-generated from {count} corrections ({datetime.now().date()})"]
            for val in set(str(v) for v in wrong_vals):
                lines.append(f'  - "{val}"')
            rule_block = "\n".join(lines)

            if semantics_path.exists():
                content = semantics_path.read_text(encoding="utf-8")
                # Find the stage section and append
                section_pattern = re.compile(
                    rf'^(\s+{re.escape(correct_val)}:\s*$)', re.MULTILINE
                )
                if section_pattern.search(content):
                    # Append after the section header's last entry
                    content = content + rule_block + "\n"
                else:
                    # Add a new entry under stage_synonyms
                    content += f"\n  # Auto-added stage: {correct_val}\n  {correct_val}:{rule_block}\n"
                semantics_path.write_text(content, encoding="utf-8")
            else:
                # Create minimal file
                minimal = (
                    f"stage_synonyms:\n"
                    f"  {correct_val}:{rule_block}\n"
                )
                semantics_path.write_text(minimal, encoding="utf-8")

            msg = f"New pattern rule generated for [{field}={correct_val}] from {count} corrections"
            logger.info(msg)
            generated.append(msg)

        except Exception as e:
            logger.error("Failed to write pattern rule: %s", e)

    return generated


# ---------------------------------------------------------------------------
# list_corrections (utility for CLI)
# ---------------------------------------------------------------------------

def list_corrections(
    project_code: str | None = None,
    work_dir: Path | None = None,
) -> list[dict]:
    """Return all corrections, optionally filtered by project_code."""
    corrections = _load_corrections(work_dir)
    if project_code:
        return [c for c in corrections if c["project_code"] == project_code]
    return corrections
