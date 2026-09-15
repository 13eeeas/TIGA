"""
tests/test_index_state.py — Catalog honesty + pipeline progress reporting (#14).

Synthetic SQLite only. No NAS, no Ollama, no Streamlit.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from config import load_config
from core.db import file_id_from_path, get_connection, upsert_file
from core.index_state import (
    STALL_SECONDS,
    annotate_job,
    catalog_index_state,
    classify_project_state,
    honour_control,
    normalise_index_path,
    path_matches_root,
    status_counts_under_root,
    unknown_catalog,
)


def _cfg(tmp_path: Path, roots: list[str] | None = None):
    work = tmp_path / "work"
    work.mkdir()
    cfg_file = work / "config.yaml"
    cfg_file.write_text(
        yaml.dump({"index_roots": roots or [str(tmp_path / "nas")]}),
        encoding="utf-8",
    )
    return load_config(config_file=cfg_file, work_dir=work)


@pytest.fixture
def conn(tmp_path: Path):
    db = get_connection(tmp_path / "db" / "tiga.db")
    yield db
    db.close()


def _seed(
    conn,
    path: str,
    *,
    project: str | None,
    status: str,
    error_code: str | None = None,
    error_detail: str | None = None,
) -> str:
    fid = file_id_from_path(path)
    upsert_file(conn, {
        "file_id": fid,
        "file_path": path,
        "file_name": Path(path).name,
        "extension": Path(path).suffix or ".txt",
        "size_bytes": 10,
        "lane": "TEXT_EXTRACTABLE",
        "status": status,
        "project_id": project,
        "error_code": error_code,
        "error_detail": error_detail,
    })
    return fid


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def test_normalise_index_path_slash_and_unc() -> None:
    assert normalise_index_path(r"Z:\Projects\NUS") == "z:/projects/nus"
    assert normalise_index_path("//?/UNC/nas/share/NUS/") == "//nas/share/nus"
    assert normalise_index_path(Path("/tmp/Archive/")) == "/tmp/archive"


def test_path_matches_root_ignores_slash_and_case() -> None:
    assert path_matches_root(r"Z:\Projects\NUS\brief.pdf", "Z:/Projects")
    assert not path_matches_root("Z:/Other/file.pdf", "Z:/Projects")


# ---------------------------------------------------------------------------
# Catalog is the source of truth (not index_roots prefix match)
# ---------------------------------------------------------------------------

def test_catalog_counts_project_ids_when_roots_do_not_match_paths(conn) -> None:
    """Office bug: INDEXED files exist, root prefix match is 0, pipeline said 0."""
    _seed(conn, "//nas/share/186 - Tianmu/brief.pdf", project="186_TIANMU", status="INDEXED")
    _seed(conn, "//nas/share/261 NUS/ga.dwg", project="261_NUS", status="INDEXED")
    _seed(conn, "//nas/share/261 NUS/scan.pdf", project="261_NUS", status="DISCOVERED")

    # Configured roots look like mapped-drive paths that do not prefix-match UNC.
    catalog = catalog_index_state(conn, configured_roots=["Z:/Projects"])

    assert catalog["counts_known"] is True
    assert catalog["honesty"] == "catalog"
    assert catalog["indexed_projects"] == 2
    assert catalog["catalog_projects"] == 2
    assert catalog["configured_projects"] == 1
    assert catalog["unmatched_indexed_files"] == 2
    assert catalog["totals"]["files_indexed"] == 2
    assert catalog["totals"]["files_discovered"] == 1

    by_id = {p["project_id"]: p for p in catalog["projects"]}
    assert by_id["186_TIANMU"]["state"] == "indexed"
    assert by_id["186_TIANMU"]["files_indexed"] == 1
    assert by_id["261_NUS"]["state"] == "in_progress"
    assert by_id["261_NUS"]["files_indexed"] == 1
    assert by_id["261_NUS"]["files_in_progress"] == 1

    root = catalog["roots"][0]
    assert root["files_indexed"] == 0
    assert root["state"] == "empty"


def test_one_root_many_projects_indexed_projects_not_capped(conn, tmp_path: Path) -> None:
    """A single NAS root can contain many project_ids — do not cap at root count."""
    archive = tmp_path / "Projects"
    _seed(conn, str(archive / "A" / "a.txt"), project="PROJ_A", status="INDEXED")
    _seed(conn, str(archive / "B" / "b.txt"), project="PROJ_B", status="INDEXED")
    catalog = catalog_index_state(conn, configured_roots=[str(archive)])
    assert catalog["configured_projects"] == 1
    assert catalog["indexed_projects"] == 2
    assert catalog["indexed_projects"] > catalog["configured_projects"]


def test_discovered_not_indexed_is_in_progress_not_a_silent_zero(conn) -> None:
    _seed(conn, "/nas/NUS/brief.pdf", project="NUS", status="DISCOVERED")
    _seed(conn, "/nas/NUS/ga.dwg", project="NUS", status="EXTRACTED")
    catalog = catalog_index_state(conn, configured_roots=["/nas"])
    assert catalog["indexed_projects"] == 0
    assert catalog["in_progress_projects"] == 1
    assert catalog["totals"]["file_count"] == 2
    assert catalog["projects"][0]["state"] == "in_progress"


def test_failed_project_exposes_last_error_and_timestamp(conn) -> None:
    _seed(
        conn,
        "/nas/HICA/bad.pdf",
        project="HICA",
        status="FAILED",
        error_code="EXTRACT_ERROR",
        error_detail="PDF parser exploded",
    )
    catalog = catalog_index_state(conn, configured_roots=["/nas"])
    proj = catalog["projects"][0]
    assert proj["state"] == "failed"
    assert proj["last_error"] == "PDF parser exploded"
    assert proj["last_error_at"]
    assert catalog["failed_projects"] == 1


def test_unknown_catalog_does_not_report_zero_indexed() -> None:
    payload = unknown_catalog(["Z:/A", "Z:/B"], error="db locked")
    assert payload["counts_known"] is False
    assert payload["honesty"] == "unknown"
    assert payload["indexed_projects"] is None
    assert payload["catalog_projects"] is None
    assert payload["configured_projects"] == 2
    assert payload["counts_error"] == "db locked"


def test_classify_project_state_priority() -> None:
    assert classify_project_state(10, 2, 1, 13) == "in_progress"
    assert classify_project_state(10, 0, 1, 11) == "indexed"
    assert classify_project_state(0, 0, 3, 3) == "failed"
    assert classify_project_state(0, 0, 0, 0) == "empty"


def test_status_counts_under_root_slash_insensitive(conn) -> None:
    _seed(conn, "Z:/Projects/NUS/a.txt", project="NUS", status="INDEXED")
    _seed(conn, "Z:/Projects/NUS/b.txt", project="NUS", status="FAILED")
    counts = status_counts_under_root(conn, r"Z:\Projects")
    assert counts["indexed"] == 1
    assert counts["failed"] == 1


# ---------------------------------------------------------------------------
# Job progress / stall
# ---------------------------------------------------------------------------

def test_annotate_job_rate_and_eta() -> None:
    job = {
        "running": True,
        "stage": "extract",
        "processed": 50,
        "total": 100,
        "started_at": 1_000.0,
        "last_progress_at": 1_010.0,
        "errors": [],
    }
    annotate_job(job, now=1_010.0)
    assert job["phase"] == "extract"
    assert job["throughput"] == 5.0
    assert job["eta"] == 10
    assert job["stalled"] is False


def test_annotate_job_stalled_with_idle_seconds() -> None:
    started = 1_000.0
    last = started + 5
    now = last + STALL_SECONDS + 1
    job = {
        "running": True,
        "phase": "embed",
        "processed": 3,
        "total": 900,
        "started_at": started,
        "last_progress_at": last,
        "errors": ["ollama timeout"],
    }
    annotate_job(job, now=now)
    assert job["stalled"] is True
    assert job["idle_seconds"] == round(STALL_SECONDS + 1)
    assert job["last_error"] == "ollama timeout"


def test_annotate_job_idle_not_stalled() -> None:
    job = {
        "running": True,
        "processed": 1,
        "total": 10,
        "started_at": 0.0,
        "last_progress_at": 10.0,
        "errors": [],
    }
    annotate_job(job, now=40.0, stall_seconds=180)
    assert job["stalled"] is False
    assert job["idle_seconds"] == 30


def test_honour_control_cancel_and_pause() -> None:
    assert honour_control(lambda: True, lambda: False) == "cancelled"
    paused = {"n": 2}

    def is_paused() -> bool:
        paused["n"] -= 1
        return paused["n"] >= 0

    assert honour_control(lambda: False, is_paused, sleep_s=0.0) is None


# ---------------------------------------------------------------------------
# Pipeline emits progress events (synthetic, no NAS/Ollama)
# ---------------------------------------------------------------------------

def test_run_full_pipeline_emits_phase_progress(tmp_path: Path, conn) -> None:
    from core.index import run_full_pipeline

    cfg = _cfg(tmp_path)
    events: list[dict] = []

    with patch("core.discover.run_discover", return_value={"total": 0, "discovered": 0}), \
         patch("core.index._run_path_parse", return_value={}), \
         patch("core.index.run_index", return_value={"files_indexed": 0}), \
         patch("core.index._run_image_indexing", return_value={}), \
         patch("core.index._run_content_classification", return_value={}):
        run_full_pipeline(conn, cfg, progress=events.append)

    phases = [e.get("phase") for e in events]
    assert "discover" in phases
    assert "extract" in phases
    assert "embed" in phases


def test_run_fts_reports_progress(conn, tmp_path: Path) -> None:
    from core.index import run_fts

    _seed(conn, str(tmp_path / "a.txt"), project="NUS", status="EMBEDDED")
    events: list[dict] = []
    stats = run_fts(conn, progress=events.append)
    assert stats["files_indexed"] == 1
    assert events
    assert events[-1]["phase"] == "fts"
    assert events[-1]["processed"] == 1
    assert events[-1]["total"] == 1
