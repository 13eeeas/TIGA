"""
tests/test_index_integrity.py — Ticket #3 first cut.

Synthetic processing-state regression (indexed / metadata_only / excluded /
duplicate / failed). No office-corpus claim. No contamination scoring.

Run: pytest tests/test_index_integrity.py -v
"""

from __future__ import annotations

import json
from pathlib import Path

from core.index_integrity import (
    DEFAULT_EXPECTED,
    FORCE_READ_ERROR_NAME,
    PROCESSING_STATES,
    audit_processing_states,
    classify_file,
    compare_to_expected,
    format_summary,
    run_integrity_benchmark,
    write_reports,
)


def test_classifier_indexed_requires_real_text() -> None:
    indexed = classify_file(
        rel_path="101_nus_synth/01_Reports/design_brief.txt",
        on_disk=True,
        excluded_by_glob=False,
        row={"status": "INDEXED", "lane": "TEXT_EXTRACTABLE"},
        chunks=[{"ref_value": "sec01", "text": "campus briefing note", "embedded": 1}],
    )
    assert indexed["processing_state"] == "indexed"
    assert indexed["missing_text"] is False

    flagged = classify_file(
        rel_path="report.txt",
        on_disk=True,
        excluded_by_glob=False,
        row={"status": "INDEXED", "lane": "TEXT_EXTRACTABLE"},
        chunks=[{"ref_value": "meta", "text": "report | txt | metadata only | path", "embedded": 1}],
    )
    assert flagged["processing_state"] == "metadata_only"
    assert flagged["missing_text"] is True


def test_classifier_metadata_only_lane() -> None:
    rec = classify_file(
        rel_path="101_nus_synth/06_Drawings/level_01.dwg",
        on_disk=True,
        excluded_by_glob=False,
        row={"status": "INDEXED", "lane": "METADATA_ONLY"},
        chunks=[{"ref_value": "meta", "text": "level_01 | dwg | metadata only", "embedded": 1}],
    )
    assert rec["processing_state"] == "metadata_only"
    assert rec["missing_text"] is True


def test_classifier_excluded_duplicate_failed() -> None:
    excluded = classify_file(
        rel_path="junk/~$lock.docx",
        on_disk=True,
        excluded_by_glob=True,
        row=None,
        chunks=[],
    )
    assert excluded["processing_state"] == "excluded"

    duplicate = classify_file(
        rel_path="copies/design_brief.txt",
        on_disk=True,
        excluded_by_glob=False,
        row={
            "status": "SKIPPED",
            "lane": "SKIPPED",
            "error_code": "DUPLICATE",
            "duplicate_of": "abc",
        },
        chunks=[],
    )
    assert duplicate["processing_state"] == "duplicate"

    failed = classify_file(
        rel_path=f"failed/{FORCE_READ_ERROR_NAME}",
        on_disk=True,
        excluded_by_glob=False,
        row={"status": "FAILED", "error_code": "READ_ERROR", "error_detail": "denied"},
        chunks=[],
    )
    assert failed["processing_state"] == "failed"
    assert failed["failure_class"] == "extraction"
    assert "READ_ERROR" in failed["reason"]


def test_classifier_stale_embeddings_visible() -> None:
    rec = classify_file(
        rel_path="brief.txt",
        on_disk=True,
        excluded_by_glob=False,
        row={"status": "INDEXED", "lane": "TEXT_EXTRACTABLE"},
        chunks=[{"ref_value": "sec01", "text": "real brief text", "embedded": 0}],
    )
    assert rec["processing_state"] == "indexed"
    assert rec["stale_embeddings"] is True


def test_integrity_pipeline_matches_expected_states(tmp_path: Path) -> None:
    report = run_integrity_benchmark(
        tmp_path / "integrity_sandbox",
        expected_path=DEFAULT_EXPECTED,
        mock_embed=True,
        verbose=False,
    )
    assert report["office_corpus_acceptance"] is False
    assert report["scope"]["cross_project_contamination"] == "not_in_scope"
    assert report["scope"]["retrieval_top3_gate"] == "not_claimed"
    assert report["expected"]["ok"], report["expected"]["mismatches"]
    assert report["ok"], report["errors"]
    assert report["coverage"]["unexplained"] == 0
    assert report["coverage"]["accounted"] == report["coverage"]["files_on_disk"]

    by_state = report["coverage"]["by_state"]
    assert by_state["indexed"] >= 3
    assert by_state["metadata_only"] >= 3
    assert by_state["excluded"] >= 2
    assert by_state["duplicate"] == 1
    assert by_state["failed"] == 1

    files = {r["rel_path"]: r for r in report["files"]}
    scanned = next(r for r in report["files"] if r["rel_path"].endswith("site_scan.pdf"))
    assert scanned["processing_state"] == "metadata_only"
    assert scanned["missing_text"] is True

    failed = next(r for r in report["files"] if r["rel_path"].endswith(FORCE_READ_ERROR_NAME))
    assert failed["processing_state"] == "failed"
    assert failed["failure_class"] == "extraction"

    lock = next(r for r in report["files"] if r["rel_path"].endswith("~$lock.docx"))
    assert lock["processing_state"] == "excluded"
    assert lock.get("db_status") is None

    for rec in report["files"]:
        assert rec["processing_state"] in PROCESSING_STATES + ("unexplained",)
        assert rec["rel_path"]
    assert files  # archive produced rows


def test_report_includes_sha_and_readable_summary(tmp_path: Path) -> None:
    report = run_integrity_benchmark(
        tmp_path / "integrity_sha",
        mock_embed=True,
        verbose=False,
    )
    assert report["build_sha"]
    assert report["build_sha"] != ""
    assert report["ticket"] == "3"
    assert report["office_corpus_acceptance"] is False

    json_path = Path(report["report_json"])
    txt_path = Path(report["report_txt"])
    assert json_path.exists()
    assert txt_path.exists()

    loaded = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded["build_sha"] == report["build_sha"]
    assert loaded["office_corpus_acceptance"] is False
    assert "latency_ms" in loaded
    assert "coverage" in loaded

    summary = txt_path.read_text(encoding="utf-8")
    assert report["build_sha"] in summary or report["build_sha_short"] in summary
    assert "NOT CLAIMED" in summary
    assert "indexed" in summary
    assert format_summary(report).count("NOT CLAIMED") == 1


def test_compare_to_expected_pair_order_independent() -> None:
    records = [
        {"rel_path": "101_nus_synth/01_Reports/design_brief.txt", "processing_state": "duplicate"},
        {"rel_path": "101_nus_synth/copies/design_brief.txt", "processing_state": "indexed"},
        {"rel_path": "101_nus_synth/junk/scratch.tmp", "processing_state": "excluded"},
    ]
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as td:
        manifest = Path(td) / "expected.yaml"
        manifest.write_text(
            "files:\n"
            "  - path: 101_nus_synth/01_Reports/design_brief.txt\n"
            "    processing_state: indexed\n"
            "    pair: brief_dup\n"
            "  - path: 101_nus_synth/copies/design_brief.txt\n"
            "    processing_state: duplicate\n"
            "    pair: brief_dup\n"
            "  - path: 101_nus_synth/junk/scratch.tmp\n"
            "    processing_state: excluded\n",
            encoding="utf-8",
        )
        result = compare_to_expected(records, manifest)
    assert result["ok"] is True
    assert result["matched"] == 3


def test_write_reports_roundtrip(tmp_path: Path) -> None:
    report = {
        "ticket": "3",
        "ts": "20260101T000000",
        "build_sha": "abc123def456",
        "office_corpus_acceptance": False,
        "office_corpus_note": "synthetic only",
        "coverage": {
            "files_on_disk": 1,
            "accounted": 1,
            "unexplained": 0,
            "by_state": {"indexed": 1, "metadata_only": 0, "excluded": 0, "duplicate": 0, "failed": 0},
        },
        "latency_ms": {"index": 1, "audit": 1, "total": 2},
        "expected": {"ok": True, "matched": 1, "expected_count": 1},
        "files": [],
        "ok": True,
        "corpus_snapshot": {"name": "integrity_archive"},
    }
    json_path, txt_path = write_reports(report, tmp_path)
    assert json.loads(json_path.read_text())["build_sha"] == "abc123def456"
    assert "abc123def456" in txt_path.read_text()


def test_audit_flags_db_only_missing_text(tmp_path: Path) -> None:
    """A files-table row with no disk file and no exclude match is unexplained."""
    from config import load_config
    from core.db import get_connection, file_id_from_path
    import yaml

    archive = tmp_path / "archive"
    archive.mkdir()
    (archive / "keep.txt").write_text("hello", encoding="utf-8")
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"index_roots": [str(archive)]}), encoding="utf-8")
    cfg = load_config(config_file=cfg_file, work_dir=tmp_path)

    conn = get_connection(tmp_path / "db" / "tiga.db")
    ghost = (archive / "missing.txt").resolve().as_posix()
    conn.execute(
        "INSERT INTO files (file_id, file_path, file_name, extension, status) "
        "VALUES (?, ?, 'missing.txt', '.txt', 'INDEXED')",
        (file_id_from_path(ghost), ghost),
    )
    conn.commit()
    records = audit_processing_states(conn, archive, cfg)
    conn.close()
    ghosts = [r for r in records if "missing.txt" in r["rel_path"]]
    assert ghosts
    assert ghosts[0]["on_disk"] is False
    assert ghosts[0]["missing_text"] is True
    assert ghosts[0]["processing_state"] == "metadata_only"
