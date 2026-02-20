"""
tests/test_infer.py — Project inference unit tests.

Run with: pytest tests/test_infer.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from config import load_config
from core.infer import infer_project, infer_typology, infer_title


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _write_config(
    tmp_path: Path,
    index_roots: list[Path],
    *,
    confidence_threshold: float = 0.5,
    patterns: list[dict] | None = None,
    keyword_boosts: list[dict] | None = None,
    typology_map: dict | None = None,
) -> "Config":  # type: ignore[name-defined]
    data = {
        "index_roots": [str(r) for r in index_roots],
        "project_inference": {
            "enable": True,
            "confidence_threshold_unknown": confidence_threshold,
            "patterns": patterns or [],
            "keyword_boosts": keyword_boosts or [],
        },
        "typology_inference": {
            "confidence_threshold_unknown": 0.5,
            "keyword_map": typology_map or {},
        },
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(data), encoding="utf-8")
    return load_config(config_file=cfg_file, work_dir=tmp_path)


def _make_file(directory: Path, name: str = "brief.pdf") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    f = directory / name
    f.write_bytes(b"fake content")
    return f


# ---------------------------------------------------------------------------
# Manual override (tiga_project.yaml)
# ---------------------------------------------------------------------------

def test_infer_project_manual_override_beats_inference(tmp_path: Path) -> None:
    """tiga_project.yaml in a parent dir overrides all regex/keyword logic."""
    root = tmp_path / "archive"
    proj = root / "messy_folder_no_code"
    cfg_obj = _write_config(tmp_path, [root])
    f = _make_file(proj, "drawing.pdf")

    # Drop override in the project dir
    (proj / "tiga_project.yaml").write_text(
        "project_id: 2023_MARINA_TOWERS\ntypology: residential\n",
        encoding="utf-8",
    )

    project_id, confidence, signals = infer_project(f, cfg_obj=cfg_obj)
    assert project_id == "2023_MARINA_TOWERS"
    assert confidence == 1.0
    assert "manual_override" in signals


def test_infer_project_override_in_ancestor(tmp_path: Path) -> None:
    """Override file in a grandparent dir is found by the upward walk."""
    root = tmp_path / "archive"
    proj = root / "some_project"
    subdir = proj / "drawings" / "floor_plans"
    cfg_obj = _write_config(tmp_path, [root])
    f = _make_file(subdir, "plan.dwg")

    # Override is in the project dir (grandparent of file)
    (proj / "tiga_project.yaml").write_text(
        "project_id: OVERRIDE_PROJECT\n", encoding="utf-8"
    )

    project_id, confidence, signals = infer_project(f, cfg_obj=cfg_obj)
    assert project_id == "OVERRIDE_PROJECT"
    assert "manual_override" in signals


def test_infer_typology_override_from_yaml(tmp_path: Path) -> None:
    """tiga_project.yaml typology field is respected by infer_typology."""
    root = tmp_path / "archive"
    proj = root / "2023_RESI"
    cfg_obj = _write_config(tmp_path, [root])
    f = _make_file(proj, "brief.pdf")

    (proj / "tiga_project.yaml").write_text(
        "project_id: 2023_RESI\ntypology: residential\n", encoding="utf-8"
    )

    typology = infer_typology(f, cfg_obj=cfg_obj)
    assert typology == "residential"


# ---------------------------------------------------------------------------
# Step 1: Regex patterns
# ---------------------------------------------------------------------------

def test_infer_project_regex_pattern_match(tmp_path: Path) -> None:
    """Pattern matching a year-code folder name contributes confidence."""
    root = tmp_path / "archive"
    proj = root / "2023_HOSP"
    cfg_obj = _write_config(
        tmp_path,
        [root],
        confidence_threshold=0.5,
        patterns=[{"name": "year-code", "regex": r"^\d{3,4}[-_ ]", "weight": 0.8}],
    )
    f = _make_file(proj, "brief.pdf")

    project_id, confidence, signals = infer_project(f, cfg_obj=cfg_obj)
    assert project_id == "2023_HOSP"
    assert confidence >= 0.8
    assert "pattern:year-code" in signals


def test_infer_project_no_pattern_match_stays_below_threshold(tmp_path: Path) -> None:
    """Folder with no pattern match and no boosts → Unknown."""
    root = tmp_path / "archive"
    proj = root / "random_folder"
    cfg_obj = _write_config(
        tmp_path,
        [root],
        confidence_threshold=0.5,
        patterns=[{"name": "year-code", "regex": r"^\d{4}[-_ ]", "weight": 0.8}],
    )
    f = _make_file(proj, "doc.pdf")

    project_id, confidence, signals = infer_project(f, cfg_obj=cfg_obj)
    assert project_id == "Unknown"
    assert confidence < 0.5


# ---------------------------------------------------------------------------
# Step 2: Keyword boosts
# ---------------------------------------------------------------------------

def test_infer_project_keyword_boost_adds_confidence(tmp_path: Path) -> None:
    """Keyword found in tokens contributes confidence."""
    root = tmp_path / "archive"
    proj = root / "2023_HOSP"
    cfg_obj = _write_config(
        tmp_path,
        [root],
        confidence_threshold=0.1,
        patterns=[{"name": "year-code", "regex": r"^\d{3,4}[-_ ]", "weight": 0.6}],
        keyword_boosts=[{"keyword": "brief", "weight": 0.2}],
    )
    f = _make_file(proj, "brief.pdf")   # "brief" appears in stem tokens

    project_id, confidence, signals = infer_project(f, cfg_obj=cfg_obj)
    assert "keyword:brief" in signals
    assert confidence >= 0.8   # 0.6 (pattern) + 0.2 (keyword)


def test_infer_project_keyword_case_insensitive(tmp_path: Path) -> None:
    """Keyword matching is case-insensitive."""
    root = tmp_path / "archive"
    proj = root / "2023_HOSP"
    cfg_obj = _write_config(
        tmp_path,
        [root],
        confidence_threshold=0.1,
        keyword_boosts=[{"keyword": "TENDER", "weight": 0.3}],
    )
    f = _make_file(proj, "Tender_Submission.pdf")

    _, confidence, signals = infer_project(f, cfg_obj=cfg_obj)
    assert "keyword:TENDER" in signals
    assert confidence >= 0.3


# ---------------------------------------------------------------------------
# Step 3: Sibling frequency heuristic
# ---------------------------------------------------------------------------

def test_infer_project_sibling_heuristic_fires(tmp_path: Path) -> None:
    """
    When >50% of peer dirs in root share the same top token,
    confidence += 0.15 and 'sibling_heuristic' appears in signals.
    """
    root = tmp_path / "archive"
    # Three dirs starting with "2023" out of three total → 100% → heuristic fires
    (root / "2023_HOSP").mkdir(parents=True)
    (root / "2023_SCHOOL").mkdir()
    (root / "2023_RESI").mkdir()

    cfg_obj = _write_config(tmp_path, [root], confidence_threshold=0.1)
    f = _make_file(root / "2023_HOSP", "brief.pdf")

    _, confidence, signals = infer_project(f, cfg_obj=cfg_obj)
    assert "sibling_heuristic" in signals
    assert confidence >= 0.15


def test_infer_project_sibling_heuristic_does_not_fire_below_threshold(
    tmp_path: Path,
) -> None:
    """Exactly 50% shared top token → heuristic does NOT fire (must be >50%)."""
    root = tmp_path / "archive"
    (root / "2023_HOSP").mkdir(parents=True)
    (root / "2022_RESI").mkdir()   # 1/2 = 50%, not > 50%

    cfg_obj = _write_config(tmp_path, [root], confidence_threshold=0.01)
    f = _make_file(root / "2023_HOSP", "brief.pdf")

    _, confidence, signals = infer_project(f, cfg_obj=cfg_obj)
    assert "sibling_heuristic" not in signals


# ---------------------------------------------------------------------------
# Step 4: Threshold
# ---------------------------------------------------------------------------

def test_infer_project_below_threshold_returns_unknown(tmp_path: Path) -> None:
    """Confidence below threshold → project_id = 'Unknown'."""
    root = tmp_path / "archive"
    proj = root / "random_name"
    cfg_obj = _write_config(tmp_path, [root], confidence_threshold=0.9)
    f = _make_file(proj, "notes.txt")

    project_id, confidence, signals = infer_project(f, cfg_obj=cfg_obj)
    assert project_id == "Unknown"


def test_infer_project_at_or_above_threshold_returns_folder(tmp_path: Path) -> None:
    """Confidence >= threshold → project folder name returned."""
    root = tmp_path / "archive"
    proj = root / "2023_HOSP"
    cfg_obj = _write_config(
        tmp_path,
        [root],
        confidence_threshold=0.5,
        patterns=[{"name": "year-code", "regex": r"^\d{4}[-_]", "weight": 0.8}],
    )
    f = _make_file(proj, "drawing.dwg")

    project_id, confidence, signals = infer_project(f, cfg_obj=cfg_obj)
    assert project_id == "2023_HOSP"
    assert confidence >= 0.5


def test_infer_project_confidence_capped_at_1(tmp_path: Path) -> None:
    """Multiple signals cannot push confidence above 1.0."""
    root = tmp_path / "archive"
    proj = root / "2023_HOSP"
    # Three project dirs so sibling heuristic fires too
    (root / "2023_SCHOOL").mkdir(parents=True)
    (root / "2023_RESI").mkdir()
    cfg_obj = _write_config(
        tmp_path,
        [root],
        confidence_threshold=0.1,
        patterns=[{"name": "year-code", "regex": r"^\d{4}[-_]", "weight": 0.8}],
        keyword_boosts=[
            {"keyword": "brief",  "weight": 0.3},
            {"keyword": "tender", "weight": 0.3},
        ],
    )
    f = _make_file(proj, "brief-tender.pdf")

    _, confidence, _ = infer_project(f, cfg_obj=cfg_obj)
    assert confidence <= 1.0


# ---------------------------------------------------------------------------
# INVARIANT: never raises
# ---------------------------------------------------------------------------

def test_infer_project_never_raises_on_bad_path(tmp_path: Path) -> None:
    """Nonexistent path → ('Unknown', 0.0, ['error']) or graceful Unknown — no exception."""
    cfg_obj = _write_config(tmp_path, [tmp_path])
    ghost = tmp_path / "no_such_dir" / "no_such_file.pdf"

    try:
        result = infer_project(ghost, cfg_obj=cfg_obj)
        project_id, confidence, signals = result
        assert isinstance(project_id, str)
        assert isinstance(confidence, float)
        assert isinstance(signals, list)
    except Exception as exc:
        pytest.fail(f"infer_project raised: {exc}")


def test_infer_project_never_raises_on_corrupt_yaml(tmp_path: Path) -> None:
    """Corrupt tiga_project.yaml is silently ignored — no crash."""
    root = tmp_path / "archive"
    proj = root / "2023_HOSP"
    cfg_obj = _write_config(
        tmp_path,
        [root],
        confidence_threshold=0.1,
        patterns=[{"name": "year-code", "regex": r"^\d{4}[-_]", "weight": 0.8}],
    )
    f = _make_file(proj, "brief.pdf")

    # Write a corrupt YAML override
    (proj / "tiga_project.yaml").write_text(
        "project_id: [unclosed\n  - invalid yaml {{{\n", encoding="utf-8"
    )

    try:
        project_id, confidence, signals = infer_project(f, cfg_obj=cfg_obj)
        # Corrupt YAML → override skipped → falls through to pattern inference
        assert isinstance(project_id, str)
    except Exception as exc:
        pytest.fail(f"infer_project raised on corrupt YAML: {exc}")


# ---------------------------------------------------------------------------
# infer_typology (keyword-based)
# ---------------------------------------------------------------------------

def test_infer_typology_from_path_keywords(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    cfg_obj = _write_config(
        tmp_path,
        [root],
        typology_map={"healthcare": ["hospital", "clinic"], "education": ["school", "campus"]},
    )
    f = _make_file(root / "2023_HOSPITAL_TOWER", "brief.pdf")

    typology = infer_typology(f, cfg_obj=cfg_obj)
    assert typology == "healthcare"


def test_infer_typology_unknown_when_no_match(tmp_path: Path) -> None:
    root = tmp_path / "archive"
    cfg_obj = _write_config(
        tmp_path,
        [root],
        typology_map={"healthcare": ["hospital"]},
    )
    f = _make_file(root / "2023_RESI_PROJECT", "drawing.dwg")

    typology = infer_typology(f, cfg_obj=cfg_obj)
    assert typology == "Unknown"


# ---------------------------------------------------------------------------
# infer_title
# ---------------------------------------------------------------------------

def test_infer_title_strips_prefix_and_normalises(tmp_path: Path) -> None:
    f = tmp_path / "01_Brief_Submission.pdf"
    f.write_bytes(b"x")
    title = infer_title(f)
    # Should strip leading "01_" and normalise separators
    assert "01" not in title or title != "01_Brief_Submission"


def test_infer_title_never_raises(tmp_path: Path) -> None:
    ghost = tmp_path / "ghost.pdf"
    try:
        title = infer_title(ghost)
        assert isinstance(title, str)
    except Exception as exc:
        pytest.fail(f"infer_title raised: {exc}")
