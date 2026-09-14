"""
tests/test_storage.py — Per-project TIGA operating storage (issue #8).

Synthetic work_dir + catalog fixtures. No NAS walk, no Ollama.

Run with: pytest tests/test_storage.py -v
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import yaml

from config import load_config
from core.db import file_id_from_path, get_connection, upsert_chunk, upsert_file
from core.project_card import upsert_project_card
from core.storage import (
    COST_DISCLAIMER,
    UNKNOWN_PROJECT,
    format_bytes,
    format_storage_report,
    get_storage_report,
    load_latest_snapshot,
    measure_operating_storage,
    save_snapshot,
    snapshot_path,
    walk_work_dir,
)


# ---------------------------------------------------------------------------
# Fixture builder
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path, roots: list[str] | None = None):
    archive = tmp_path / "archive"
    archive.mkdir(exist_ok=True)
    work = tmp_path / "work"
    work.mkdir(exist_ok=True)
    cfg_file = work / "config.yaml"
    cfg_file.write_text(
        yaml.dump({
            "index_roots": roots or [
                str(archive / "NUS"),
                str(archive / "NParks"),
                str(archive / "HICA"),
            ],
        }),
        encoding="utf-8",
    )
    return load_config(config_file=cfg_file, work_dir=work)


def _write_bytes(path: Path, n: int, fill: bytes = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(fill * n)


def _seed_file(
    conn,
    path: str,
    *,
    project: str | None,
    status: str,
    size: int,
    duplicate_of: str | None = None,
) -> str:
    fid = file_id_from_path(path)
    upsert_file(conn, {
        "file_id": fid,
        "file_path": path,
        "file_name": Path(path).name,
        "extension": Path(path).suffix,
        "size_bytes": size,
        "lane": "TEXT_EXTRACTABLE",
        "status": status,
        "project_id": project,
        "duplicate_of": duplicate_of,
    })
    if project:
        conn.execute(
            "UPDATE files SET project_code = ? WHERE file_id = ?",
            (project, fid),
        )
        conn.commit()
    return fid


def _seed_chunk(conn, file_id: str, ref: str, text: str, *, embedded: int = 1) -> None:
    chunk_id = hashlib.sha256(f"{file_id}::{ref}".encode()).hexdigest()
    upsert_chunk(conn, {
        "chunk_id": chunk_id,
        "file_id": file_id,
        "ref_value": ref,
        "text": text,
        "token_estimate": max(len(text) // 4, 1),
        "embedded": embedded,
    })


def build_storage_fixture(tmp_path: Path):
    """
    Synthetic TIGA work_dir + catalog covering:
      - NUS: fully indexed + Atlas overlay
      - NParks: incomplete indexing
      - HICA: fully indexed
      - unknown: missing project identity
      - shared SQLite / vectors, logs, temp, backup
    """
    cfg = _cfg(tmp_path)
    work = cfg.work_dir
    conn = get_connection(cfg.get_db_path())

    # Known-size operating artifacts (on top of the real SQLite file).
    _write_bytes(work / "db" / "sidecar.idx", 8_000, b"S")
    _write_bytes(work / "vectors" / "tiga_chunks.lance", 20_000, b"V")
    _write_bytes(work / "atlas" / "nus.overlay.json", 4_000, b"A")
    _write_bytes(work / "atlas" / "orphan.overlay.json", 500, b"O")
    _write_bytes(work / "logs" / "tiga.log", 2_000, b"L")
    _write_bytes(work / "tmp" / "scratch.dat", 1_500, b"T")
    _write_bytes(work / "backups" / "tiga.db.bak", 3_000, b"B")
    _write_bytes(work / "config.yaml", max(cfg.config_file.stat().st_size, 1))

    # NUS — complete
    for i in range(4):
        fid = _seed_file(
            conn,
            f"/archive/NUS/brief_{i}.txt",
            project="NUS",
            status="INDEXED",
            size=100_000,
        )
        _seed_chunk(conn, fid, "p1", "NUS campus brief " * 20, embedded=1)
        _seed_chunk(conn, fid, "p2", "NUS page reference " * 10, embedded=1)

    # NParks — incomplete (2 indexed, 3 discovered)
    for i in range(2):
        fid = _seed_file(
            conn,
            f"/archive/NParks/plan_{i}.pdf",
            project="NParks",
            status="INDEXED",
            size=250_000,
        )
        _seed_chunk(conn, fid, "p1", "NParks trail plan " * 15, embedded=1)
    for i in range(3):
        _seed_file(
            conn,
            f"/archive/NParks/pending_{i}.dwg",
            project="NParks",
            status="DISCOVERED",
            size=80_000,
        )

    # HICA — complete
    for i in range(3):
        fid = _seed_file(
            conn,
            f"/archive/HICA/spec_{i}.docx",
            project="HICA",
            status="INDEXED",
            size=50_000,
        )
        _seed_chunk(conn, fid, "p1", "HICA hospitality spec " * 12, embedded=1)

    # Missing project identity
    for i in range(2):
        fid = _seed_file(
            conn,
            f"/archive/loose/note_{i}.txt",
            project=None,
            status="INDEXED",
            size=10_000,
        )
        _seed_chunk(conn, fid, "p1", "unlabelled scrap " * 8, embedded=0)

    # Deduplicated copy must not inflate source archive size
    _seed_file(
        conn,
        "/archive/NUS/brief_0_copy.txt",
        project="NUS",
        status="SKIPPED",
        size=100_000,
        duplicate_of=file_id_from_path("/archive/NUS/brief_0.txt"),
    )

    upsert_project_card({"project_code": "NUS", "name": "NUS Campus"}, conn=conn)
    upsert_project_card({"project_code": "NParks", "name": "NParks"}, conn=conn)
    upsert_project_card({"project_code": "HICA", "name": "HICA"}, conn=conn)

    return cfg, conn


def _by_id(snapshot: dict) -> dict:
    return {p["project_id"]: p for p in snapshot["projects"]}


# ---------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------

def test_format_bytes_uses_explicit_units() -> None:
    assert format_bytes(512) == "512 B"
    assert format_bytes(2048).endswith("KiB")
    assert format_bytes(3 * 1024 ** 2).endswith("MiB")
    assert format_bytes(2 * 1024 ** 3).endswith("GiB")


# ---------------------------------------------------------------------------
# Reconciliation + shared overhead
# ---------------------------------------------------------------------------

def test_reconciliation_matches_physical_work_dir(tmp_path: Path) -> None:
    cfg, conn = build_storage_fixture(tmp_path)
    try:
        snap = measure_operating_storage(
            conn,
            cfg.work_dir,
            configured_projects=["NUS", "NParks", "HICA"],
        )
        walked = sum(size for _p, size, _c in walk_work_dir(cfg.work_dir))
        assert snap["totals"]["work_dir_bytes"] == walked
        recon = snap["reconciliation"]
        assert recon["ok"] is True
        assert recon["delta_bytes"] == 0
        assert (
            recon["allocated_bytes"] + recon["unattributed_bytes"]
            == walked
        )
    finally:
        conn.close()


def test_shared_overhead_present_and_explains_allocation(tmp_path: Path) -> None:
    cfg, conn = build_storage_fixture(tmp_path)
    try:
        snap = measure_operating_storage(conn, cfg.work_dir)
        shared = snap["shared"]
        assert shared["sqlite_attribution"] == "measured"
        assert shared["vector_attribution"] == "measured"
        assert shared["sqlite_bytes"] > 0
        assert shared["vector_bytes"] == 20_000
        assert "estimated" in shared["allocation_method"].lower()
        assert "logical" in shared["allocation_method"].lower()

        unattr = snap["unattributed"]
        # logs + tmp + backup + orphan atlas overlay
        assert unattr["bytes"] >= 2_000 + 1_500 + 3_000 + 500
        assert "reconcile" in unattr["explanation"].lower()

        by = _by_id(snap)
        # Per-project sqlite/vector shares are estimated, not measured copies.
        assert by["NUS"]["operating"]["attribution"] in {"estimated", "mixed"}
        assert by["NUS"]["operating"]["components"]["sqlite"]["attribution"] == "estimated"
        assert by["NUS"]["operating"]["components"]["atlas"]["attribution"] == "measured"
        assert by["NUS"]["operating"]["components"]["atlas"]["bytes"] == 4_000
    finally:
        conn.close()


def test_logical_text_not_added_to_disk_totals(tmp_path: Path) -> None:
    cfg, conn = build_storage_fixture(tmp_path)
    try:
        snap = measure_operating_storage(conn, cfg.work_dir)
        by = _by_id(snap)
        logical = by["NUS"]["logical_text_bytes"]
        assert logical > 0
        # Operating bytes come from work_dir files, not UTF-8 text length.
        assert by["NUS"]["operating"]["bytes"] != logical
        assert "not on-disk" in by["NUS"]["logical_text_note"].lower()
        walked = snap["totals"]["work_dir_bytes"]
        # Shared SQLite/FTS on disk is not the same number as catalog source
        # bytes or logical UTF-8 length — do not equate them.
        assert walked == snap["reconciliation"]["work_dir_bytes"]
        assert by["NUS"]["logical_text_bytes"] != walked
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Incomplete indexing + missing identity + configured projects
# ---------------------------------------------------------------------------

def test_incomplete_indexing_projects_completed_footprint(tmp_path: Path) -> None:
    cfg, conn = build_storage_fixture(tmp_path)
    try:
        snap = measure_operating_storage(conn, cfg.work_dir)
        parks = _by_id(snap)["NParks"]
        assert parks["file_count"] == 5
        assert parks["indexed_file_count"] == 2
        assert parks["coverage"] == pytest.approx(0.4)
        assert parks["projected_completed_bytes"] is not None
        assert parks["projected_completed_bytes"] > parks["operating"]["bytes"]
        assert "linear" in parks["projection_assumption"].lower()
        assert "coverage" in parks["projection_assumption"].lower()

        nus = _by_id(snap)["NUS"]
        assert nus["coverage"] == pytest.approx(0.8) or nus["indexed_file_count"] == 4
        # NUS has a skipped duplicate, so coverage is 4/5
        assert nus["file_count"] == 5
        assert nus["duplicate_file_count"] == 1
        # Duplicate source bytes excluded
        assert nus["source_archive_bytes"] == 400_000
    finally:
        conn.close()


def test_missing_project_identity_bucket(tmp_path: Path) -> None:
    cfg, conn = build_storage_fixture(tmp_path)
    try:
        snap = measure_operating_storage(conn, cfg.work_dir)
        unknown = _by_id(snap)[UNKNOWN_PROJECT]
        assert unknown["identity"] == "unknown"
        assert unknown["file_count"] == 2
        assert unknown["operating"]["bytes"] >= 0
        assert "missing project identity" in unknown["identity_note"].lower()
    finally:
        conn.close()


def test_configured_projects_always_listed(tmp_path: Path) -> None:
    cfg, conn = build_storage_fixture(tmp_path)
    try:
        snap = measure_operating_storage(
            conn,
            cfg.work_dir,
            configured_projects=["NUS", "NParks", "HICA", "EmptyRoot"],
        )
        ids = {p["project_id"] for p in snap["projects"]}
        assert {"NUS", "NParks", "HICA", "EmptyRoot", UNKNOWN_PROJECT} <= ids
        empty = _by_id(snap)["EmptyRoot"]
        assert empty["file_count"] == 0
        assert empty["operating"]["bytes"] == 0
        assert empty["projected_completed_bytes"] is None
    finally:
        conn.close()


def test_source_archive_is_catalog_not_nas_walk(tmp_path: Path) -> None:
    cfg, conn = build_storage_fixture(tmp_path)
    try:
        snap = measure_operating_storage(conn, cfg.work_dir)
        assert snap["measurement"]["scanned_source_archive"] is False
        assert snap["measurement"]["scanned_work_dir"] is True
        assert snap["totals"]["source_archive_attribution"] == "catalog"
        # Source total = NUS 400k + NParks 2*250k+3*80k + HICA 150k + unknown 20k
        assert snap["totals"]["source_archive_bytes"] == (
            400_000 + 500_000 + 240_000 + 150_000 + 20_000
        )
    finally:
        conn.close()


def test_snapshot_includes_units_counts_ratio_and_time(tmp_path: Path) -> None:
    cfg, conn = build_storage_fixture(tmp_path)
    try:
        when = datetime(2026, 9, 14, 2, 0, tzinfo=timezone.utc)
        snap = measure_operating_storage(conn, cfg.work_dir, now=when)
        assert snap["snapshot_at"].startswith("2026-09-14T02:00:00")
        assert "1024" in snap["units"]["gb"]
        assert snap["totals"]["work_dir_human"].split()[-1] in {"B", "KiB", "MiB", "GiB"}
        assert snap["totals"]["chunk_count"] > 0
        assert snap["totals"]["index_to_source_ratio"] is not None
        report = format_storage_report(snap)
        assert "TIGA operating storage" in report
        assert "NUS" in report and "NParks" in report and "HICA" in report
        assert "estimated stub" in report.lower() or "cost stubs" in report.lower()
    finally:
        conn.close()


def test_cost_fields_are_labeled_stubs(tmp_path: Path) -> None:
    cfg, conn = build_storage_fixture(tmp_path)
    try:
        snap = measure_operating_storage(conn, cfg.work_dir)
        assert COST_DISCLAIMER in snap["cost_disclaimer"]
        assert "not API billing" in snap["cost_disclaimer"]
        cost = _by_id(snap)["HICA"]["cost"]
        assert cost["label"] == "estimated stub"
        for key in ("initial_generation", "refresh", "per_query"):
            assert cost[key]["label"] == "estimated stub"
            assert "not invoiced" in cost[key]["basis"].lower()
            assert "storage gb" not in cost[key]["basis"].lower() or "not" in cost[key]["basis"].lower()
        assert "not derived from storage gb" in cost["initial_generation"]["basis"].lower()
    finally:
        conn.close()


def test_growth_versus_previous_snapshot(tmp_path: Path) -> None:
    cfg, conn = build_storage_fixture(tmp_path)
    try:
        first = measure_operating_storage(
            conn,
            cfg.work_dir,
            now=datetime(2026, 9, 1, tzinfo=timezone.utc),
        )
        _write_bytes(cfg.work_dir / "logs" / "extra.log", 4_000, b"E")
        second = measure_operating_storage(
            conn,
            cfg.work_dir,
            previous=first,
            now=datetime(2026, 9, 14, tzinfo=timezone.utc),
        )
        assert second["previous_snapshot_at"] == first["snapshot_at"]
        assert second["growth"]["work_dir_bytes"] == 4_000
        assert second["totals"]["work_dir_bytes"] == first["totals"]["work_dir_bytes"] + 4_000
    finally:
        conn.close()


def test_cached_report_skips_rewalk(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg, conn = build_storage_fixture(tmp_path)
    try:
        t0 = datetime(2026, 9, 14, 3, 0, tzinfo=timezone.utc)
        first = get_storage_report(conn, cfg.work_dir, refresh=True, now=t0)
        assert first["measurement"]["mode"] == "fresh"
        assert snapshot_path(cfg.work_dir).exists()

        calls = {"n": 0}
        real_walk = walk_work_dir

        def _counted(work_dir):
            calls["n"] += 1
            return real_walk(work_dir)

        monkeypatch.setattr("core.storage.walk_work_dir", _counted)
        cached = get_storage_report(
            conn,
            cfg.work_dir,
            refresh=False,
            max_age_seconds=900,
            now=t0 + timedelta(seconds=30),
        )
        assert cached["measurement"]["mode"] == "cached"
        assert cached["measurement"]["scanned_work_dir"] is False
        assert calls["n"] == 0

        # Force refresh after cache expires
        later = get_storage_report(
            conn,
            cfg.work_dir,
            refresh=False,
            max_age_seconds=10,
            now=t0 + timedelta(seconds=60),
        )
        assert later["measurement"]["mode"] == "fresh"
        assert calls["n"] == 1
        assert load_latest_snapshot(cfg.work_dir) is not None
    finally:
        conn.close()


def test_save_snapshot_roundtrip(tmp_path: Path) -> None:
    payload = {"schema_version": 1, "snapshot_at": "2026-09-14T00:00:00+00:00"}
    path = save_snapshot(tmp_path, payload)
    assert path == snapshot_path(tmp_path)
    loaded = load_latest_snapshot(tmp_path)
    assert loaded["snapshot_at"] == payload["snapshot_at"]
