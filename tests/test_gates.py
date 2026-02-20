"""
tests/test_gates.py — Citation validator unit tests.

Run with: pytest tests/test_gates.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.db import file_id_from_path, get_connection, upsert_chunk, upsert_file
from core.eval import validate_citation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def archive(tmp_path: Path) -> Path:
    """Real file at <tmp_path>/archive/project/brief.pdf"""
    dest = tmp_path / "archive" / "project"
    dest.mkdir(parents=True)
    (dest / "brief.pdf").write_bytes(b"%PDF-1.4 fake content")
    return tmp_path / "archive"


@pytest.fixture
def db_with_chunk(tmp_path: Path, archive: Path) -> Path:
    """SQLite DB that has a file + chunk row for project/brief.pdf#p1"""
    db_path = tmp_path / "db" / "tiga.db"
    conn = get_connection(db_path)
    abs_path = (archive / "project" / "brief.pdf").resolve()
    fid = file_id_from_path(abs_path.as_posix())
    upsert_file(conn, {
        "file_id":   fid,
        "file_path": abs_path.as_posix(),
        "file_name": "brief.pdf",
        "extension": ".pdf",
    })
    upsert_chunk(conn, {
        "chunk_id":  f"{fid}::p1",
        "file_id":   fid,
        "ref_value": "p1",
        "text":      "hospital tender brief",
    })
    conn.close()
    return db_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_citation_validator_valid_citation_passes(
    archive: Path,
    db_with_chunk: Path,
) -> None:
    """A well-formed citation whose file and chunk both exist → True."""
    result = validate_citation(
        "project/brief.pdf#p1",
        str(db_with_chunk),
        [str(archive)],
    )
    assert result is True


def test_citation_validator_root_label_stripped(
    archive: Path,
    db_with_chunk: Path,
) -> None:
    """[rootLabel] prefix must be stripped before validation."""
    result = validate_citation(
        "[rootA] project/brief.pdf#p1",
        str(db_with_chunk),
        [str(archive)],
    )
    assert result is True


def test_citation_validator_missing_file_fails(
    tmp_path: Path,
    db_with_chunk: Path,
) -> None:
    """File does not exist on disk → False."""
    result = validate_citation(
        "project/nonexistent.pdf#p1",
        str(db_with_chunk),
        [str(tmp_path / "archive")],
    )
    assert result is False


def test_citation_validator_missing_chunk_fails(
    archive: Path,
    db_with_chunk: Path,
) -> None:
    """File exists but ref_value is not in DB → False."""
    result = validate_citation(
        "project/brief.pdf#p99",    # p99 was never indexed
        str(db_with_chunk),
        [str(archive)],
    )
    assert result is False


def test_citation_validator_never_raises_on_bad_input() -> None:
    """Garbage input must return False without raising — ever."""
    bad_inputs: list = [
        "",
        "   ",
        "no_hash_at_all",
        "#only_ref",
        "rel/path#",
        "[unclosed bracket path#ref",
        None,
        12345,
        object(),
    ]
    for bad in bad_inputs:
        try:
            result = validate_citation(bad, "/nonexistent/tiga.db", [])  # type: ignore[arg-type]
            assert result is False, (
                f"Expected False for {bad!r}, got {result!r}"
            )
        except Exception as exc:
            pytest.fail(f"validate_citation raised for {bad!r}: {exc}")


def test_citation_validator_file_in_second_root(tmp_path: Path) -> None:
    """File under the second root (not the first) must still validate."""
    root1 = tmp_path / "archive1"
    root2 = tmp_path / "archive2"
    root1.mkdir()
    (root2 / "project").mkdir(parents=True)
    f = root2 / "project" / "drawing.pdf"
    f.write_bytes(b"fake pdf")

    db_path = tmp_path / "db" / "tiga.db"
    conn = get_connection(db_path)
    abs_path = f.resolve()
    fid = file_id_from_path(abs_path.as_posix())
    upsert_file(conn, {
        "file_id":   fid,
        "file_path": abs_path.as_posix(),
        "file_name": "drawing.pdf",
    })
    upsert_chunk(conn, {
        "chunk_id":  f"{fid}::s01",
        "file_id":   fid,
        "ref_value": "s01",
        "text":      "ground floor plan",
    })
    conn.close()

    result = validate_citation(
        "project/drawing.pdf#s01",
        str(db_path),
        [str(root1), str(root2)],   # root1 first — should fall through to root2
    )
    assert result is True


def test_citation_validator_empty_root_list_fails(
    archive: Path,
    db_with_chunk: Path,
) -> None:
    """Empty root_paths → file cannot be found → False."""
    result = validate_citation("project/brief.pdf#p1", str(db_with_chunk), [])
    assert result is False


def test_citation_validator_no_hash_fails(
    archive: Path,
    db_with_chunk: Path,
) -> None:
    """Citation without '#' separator → False."""
    result = validate_citation(
        "project/brief.pdf",
        str(db_with_chunk),
        [str(archive)],
    )
    assert result is False
