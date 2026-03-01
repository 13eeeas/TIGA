"""
tests/test_discover.py — Discovery pipeline unit tests.

Run with: pytest tests/test_discover.py -v
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from config import load_config
from core.db import get_connection, file_id_from_path
from core.discover import run_discover, _classify_lane


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_config(tmp_path: Path, index_roots: list[Path], **overrides) -> Path:
    data: dict = {
        "index_roots": [str(r) for r in index_roots],
        "max_file_mb": 10,
        "include_globs": ["**/*"],
        "exclude_globs": ["**/.git/**", "**/~$*", "**/*.tmp"],
        "lane_rules": {
            "text_extractable_exts": [".pdf", ".docx", ".txt", ".md"],
            "metadata_only_exts": [".dwg", ".rvt", ".ifc", ".jpg", ".png"],
        },
    }
    data.update(overrides)
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump(data), encoding="utf-8")
    return cfg_file


@pytest.fixture
def conn(tmp_path: Path):
    db_path = tmp_path / "db" / "tiga.db"
    c = get_connection(db_path)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Required tests
# ---------------------------------------------------------------------------

def test_discovers_files_in_single_root(tmp_path: Path, conn) -> None:
    """Files under a single index_root are discovered and written to the DB."""
    root = tmp_path / "archive"
    root.mkdir()
    (root / "brief.pdf").write_bytes(b"fake pdf")
    (root / "drawing.dwg").write_bytes(b"fake dwg")
    (root / "notes.txt").write_text("some notes")

    cfg_file = _write_config(tmp_path, [root])
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    stats = run_discover(conn, [root], cfg_obj=cfg_obj)

    rows = conn.execute("SELECT file_name, status, lane FROM files ORDER BY file_name").fetchall()
    file_names = {r["file_name"] for r in rows}
    assert "brief.pdf"  in file_names
    assert "drawing.dwg" in file_names
    assert "notes.txt"  in file_names
    assert all(r["status"] == "DISCOVERED" for r in rows)
    assert stats["discovered"] == 3
    assert stats["skipped"]   == 0
    assert stats["failed"]    == 0


def test_discovers_files_in_multiple_roots(tmp_path: Path, conn) -> None:
    """Files in two separate roots are both discovered and written to DB."""
    root1 = tmp_path / "archive1"
    root2 = tmp_path / "archive2"
    root1.mkdir()
    root2.mkdir()
    (root1 / "brief.pdf").write_bytes(b"pdf content")
    (root2 / "drawing.dwg").write_bytes(b"dwg content")
    (root2 / "report.docx").write_bytes(b"docx content")

    cfg_file = _write_config(tmp_path, [root1, root2])
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    stats = run_discover(conn, [root1, root2], cfg_obj=cfg_obj)

    count = conn.execute("SELECT COUNT(*) FROM files").fetchone()[0]
    assert count == 3
    assert stats["discovered"] == 3
    assert stats["total"] == 3


# ---------------------------------------------------------------------------
# Oversized file handling
# ---------------------------------------------------------------------------

def test_oversized_file_marked_skipped(tmp_path: Path, conn) -> None:
    """File exceeding max_file_mb appears in DB as SKIPPED/OVERSIZED."""
    root = tmp_path / "archive"
    root.mkdir()
    big = root / "huge.pdf"
    big.write_bytes(b"x" * (5 * 1024 * 1024))   # 5 MB

    # Set limit to 1 MB so the file is oversized
    cfg_file = _write_config(tmp_path, [root], max_file_mb=1)
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    stats = run_discover(conn, [root], cfg_obj=cfg_obj)

    row = conn.execute("SELECT * FROM files WHERE file_name='huge.pdf'").fetchone()
    assert row is not None, "SKIPPED file must appear in DB"
    assert row["status"]     == "SKIPPED"
    assert row["lane"]       == "SKIPPED"
    assert row["error_code"] == "OVERSIZED"
    assert stats["skipped"]  == 1
    assert stats["discovered"] == 0


# ---------------------------------------------------------------------------
# Exclude globs
# ---------------------------------------------------------------------------

def test_excluded_file_not_in_db(tmp_path: Path, conn) -> None:
    """Files matching exclude_globs are silently dropped — not in DB."""
    root = tmp_path / "archive"
    root.mkdir()
    (root / "~$lockfile.docx").write_bytes(b"lock")   # matches ~$* glob
    (root / "normal.pdf").write_bytes(b"normal")

    cfg_file = _write_config(tmp_path, [root])
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    run_discover(conn, [root], cfg_obj=cfg_obj)

    names = {r["file_name"] for r in conn.execute("SELECT file_name FROM files").fetchall()}
    assert "~$lockfile.docx" not in names
    assert "normal.pdf" in names


def test_excluded_directory_is_pruned_from_walk(tmp_path: Path, conn) -> None:
    """Directory segments in exclude globs are pruned before file processing."""
    root = tmp_path / "archive"
    (root / "skipme").mkdir(parents=True)
    (root / "keepme").mkdir(parents=True)
    (root / "skipme" / "hidden.pdf").write_bytes(b"x")
    (root / "keepme" / "visible.pdf").write_bytes(b"y")

    cfg_file = _write_config(tmp_path, [root], exclude_globs=["**/skipme/**"])
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    stats = run_discover(conn, [root], cfg_obj=cfg_obj)

    names = {r["file_name"] for r in conn.execute("SELECT file_name FROM files").fetchall()}
    assert "hidden.pdf" not in names
    assert "visible.pdf" in names
    assert stats["total"] == 1


# ---------------------------------------------------------------------------
# Lane classification
# ---------------------------------------------------------------------------

def test_text_extractable_ext_gets_correct_lane(tmp_path: Path, conn) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "report.pdf").write_bytes(b"pdf content")

    cfg_file = _write_config(tmp_path, [root])
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    run_discover(conn, [root], cfg_obj=cfg_obj)

    row = conn.execute("SELECT lane FROM files WHERE file_name='report.pdf'").fetchone()
    assert row["lane"] == "TEXT_EXTRACTABLE"


def test_metadata_only_ext_gets_correct_lane(tmp_path: Path, conn) -> None:
    root = tmp_path / "archive"
    root.mkdir()
    (root / "floor.dwg").write_bytes(b"dwg content")

    cfg_file = _write_config(tmp_path, [root])
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    run_discover(conn, [root], cfg_obj=cfg_obj)

    row = conn.execute("SELECT lane FROM files WHERE file_name='floor.dwg'").fetchone()
    assert row["lane"] == "METADATA_ONLY"


def test_unknown_ext_defaults_to_metadata_only(tmp_path: Path, conn) -> None:
    """Unknown extension → METADATA_ONLY (safe default, file is not dropped)."""
    root = tmp_path / "archive"
    root.mkdir()
    (root / "model.xyz").write_bytes(b"some data")   # .xyz not in any lane list

    cfg_file = _write_config(tmp_path, [root])
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    run_discover(conn, [root], cfg_obj=cfg_obj)

    row = conn.execute("SELECT * FROM files WHERE file_name='model.xyz'").fetchone()
    assert row is not None, "Unknown-ext file must still appear in DB"
    assert row["lane"]   == "METADATA_ONLY"
    assert row["status"] == "DISCOVERED"


# ---------------------------------------------------------------------------
# Incremental / fingerprint logic
# ---------------------------------------------------------------------------

def test_incremental_unchanged_indexed_file_skipped(tmp_path: Path, conn) -> None:
    """
    File already INDEXED with same fingerprint → no re-processing (unchanged).
    """
    root = tmp_path / "archive"
    root.mkdir()
    f = root / "stable.pdf"
    f.write_bytes(b"stable content")

    cfg_file = _write_config(tmp_path, [root])
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    # First pass → DISCOVERED
    stats1 = run_discover(conn, [root], cfg_obj=cfg_obj)
    assert stats1["discovered"] == 1

    # Manually advance status to INDEXED (as if pipeline completed)
    posix = f.resolve().as_posix()
    fid = file_id_from_path(posix)
    conn.execute(
        "UPDATE files SET status='INDEXED' WHERE file_id=?", (fid,)
    )
    conn.commit()

    # Second pass with same file → should be unchanged
    stats2 = run_discover(conn, [root], cfg_obj=cfg_obj)
    assert stats2["unchanged"] == 1
    assert stats2["discovered"] == 0


def test_modified_file_rediscovered(tmp_path: Path, conn) -> None:
    """File that was INDEXED but has changed fingerprint → re-DISCOVERED."""
    root = tmp_path / "archive"
    root.mkdir()
    f = root / "changing.pdf"
    f.write_bytes(b"version 1")

    cfg_file = _write_config(tmp_path, [root])
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    # First discover + advance to INDEXED
    run_discover(conn, [root], cfg_obj=cfg_obj)
    posix = f.resolve().as_posix()
    fid = file_id_from_path(posix)
    conn.execute("UPDATE files SET status='INDEXED' WHERE file_id=?", (fid,))
    conn.commit()

    # Modify the file
    f.write_bytes(b"version 2 - different content")

    stats = run_discover(conn, [root], cfg_obj=cfg_obj)
    assert stats["discovered"] == 1
    assert stats["unchanged"]  == 0


def test_incremental_fast_path_skips_fingerprint(tmp_path: Path, conn) -> None:
    """Indexed file with unchanged size+mtime skips expensive fingerprint hashing."""
    root = tmp_path / "archive"
    root.mkdir()
    f = root / "stable-fast.pdf"
    f.write_bytes(b"same")

    cfg_file = _write_config(tmp_path, [root])
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    # First discover + advance to INDEXED
    run_discover(conn, [root], cfg_obj=cfg_obj)
    posix = f.resolve().as_posix()
    fid = file_id_from_path(posix)
    conn.execute("UPDATE files SET status='INDEXED' WHERE file_id=?", (fid,))
    conn.commit()

    # Second discover should not call compute_fingerprint
    with patch("core.discover.compute_fingerprint", side_effect=AssertionError("should not hash unchanged indexed file")):
        stats = run_discover(conn, [root], cfg_obj=cfg_obj)

    assert stats["unchanged"] == 1
    assert stats["discovered"] == 0


# ---------------------------------------------------------------------------
# READ_ERROR handling
# ---------------------------------------------------------------------------

def test_fingerprint_read_error_marks_failed(tmp_path: Path, conn) -> None:
    """compute_fingerprint OSError → file recorded as FAILED/READ_ERROR."""
    root = tmp_path / "archive"
    root.mkdir()
    (root / "corrupt.pdf").write_bytes(b"content")

    cfg_file = _write_config(tmp_path, [root])
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    with patch("core.discover.compute_fingerprint", side_effect=OSError("read error")):
        stats = run_discover(conn, [root], cfg_obj=cfg_obj)

    row = conn.execute("SELECT * FROM files WHERE file_name='corrupt.pdf'").fetchone()
    assert row is not None, "FAILED file must appear in DB"
    assert row["status"]     == "FAILED"
    assert row["error_code"] == "READ_ERROR"
    assert stats["failed"]   == 1


# ---------------------------------------------------------------------------
# Nonexistent root is gracefully skipped
# ---------------------------------------------------------------------------

def test_nonexistent_root_skipped_gracefully(tmp_path: Path, conn) -> None:
    """Missing index_root logs a warning but does not crash the pipeline."""
    ghost = tmp_path / "does_not_exist"
    cfg_file = _write_config(tmp_path, [ghost])
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    stats = run_discover(conn, [ghost], cfg_obj=cfg_obj)
    assert stats["total"] == 0


# ---------------------------------------------------------------------------
# _classify_lane unit test
# ---------------------------------------------------------------------------

def test_classify_lane_helper(tmp_path: Path) -> None:
    cfg_file = _write_config(tmp_path, [tmp_path])
    cfg_obj = load_config(config_file=cfg_file, work_dir=tmp_path)

    assert _classify_lane(".pdf",  cfg_obj) == "TEXT_EXTRACTABLE"
    assert _classify_lane(".dwg",  cfg_obj) == "METADATA_ONLY"
    assert _classify_lane(".xyz",  cfg_obj) == "METADATA_ONLY"   # unknown
    assert _classify_lane("",     cfg_obj) == "METADATA_ONLY"   # no extension
