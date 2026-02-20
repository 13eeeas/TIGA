"""
tests/test_query.py — Hybrid search pipeline unit tests.

Run with: pytest tests/test_query.py -v

Vector tests are skipped when Ollama is unavailable.
All tests use a real in-memory/tmp SQLite DB seeded with fixture data.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from config import load_config
from core.db import get_connection, file_id_from_path
from core.query import search, _make_citation, _normalise_bm25, _normalise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path, roots: list[Path] | None = None):
    _roots = roots or [tmp_path]
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.dump({"index_roots": [str(r) for r in _roots]}),
        encoding="utf-8",
    )
    return load_config(config_file=cfg_file, work_dir=tmp_path)


@pytest.fixture
def conn(tmp_path: Path):
    db = get_connection(tmp_path / "db" / "tiga.db")
    yield db
    db.close()


def _seed(
    conn,
    root: Path,
    rel: str,
    text: str,
    project_id: str = "Unknown",
    typology: str = "Unknown",
    ref_value: str = "p1",
    exists_on_disk: bool = True,
) -> tuple[str, str]:
    """
    Create file on disk (optional) + insert files/chunks rows (INDEXED).
    Returns (file_id, chunk_id).
    """
    dest = root / rel
    if exists_on_disk:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")

    posix = dest.resolve().as_posix()
    file_id = file_id_from_path(posix)
    c_hash = hashlib.sha256(text.encode()).hexdigest()
    chunk_id = hashlib.sha256(f"{file_id}::{ref_value}".encode()).hexdigest()

    conn.execute(
        "INSERT OR IGNORE INTO files "
        "(file_id, file_path, file_name, extension, status, project_id, typology) "
        "VALUES (?, ?, ?, '.txt', 'INDEXED', ?, ?)",
        (file_id, posix, dest.name, project_id, typology),
    )
    conn.execute(
        "INSERT OR IGNORE INTO chunks "
        "(chunk_id, file_id, ref_value, text, content_hash, embedded) "
        "VALUES (?, ?, ?, ?, ?, 1)",
        (chunk_id, file_id, ref_value, text, c_hash),
    )
    conn.commit()
    return file_id, chunk_id


# ---------------------------------------------------------------------------
# Unit tests for helper functions
# ---------------------------------------------------------------------------

def test_normalise_bm25_best_gets_1():
    """Most-negative bm25 raw score (best match) must normalise to 1.0."""
    raw = [-2.0, -1.0, -0.5]
    norm = _normalise_bm25(raw)
    assert abs(norm[0] - 1.0) < 1e-9   # -2.0 is best
    assert abs(norm[-1] - 0.0) < 1e-9  # -0.5 is worst


def test_normalise_vec_best_gets_1():
    """Highest vector score (after 1-distance) must normalise to 1.0."""
    scores = [0.9, 0.6, 0.3]
    norm = _normalise(scores)
    assert abs(norm[0] - 1.0) < 1e-9
    assert abs(norm[-1] - 0.0) < 1e-9


def test_make_citation_single_root(tmp_path: Path) -> None:
    """Single root → citation has no bracket prefix."""
    roots = [tmp_path / "archive"]
    c = _make_citation(
        str(roots[0] / "2023_HOSP" / "brief.pdf"),
        "2023_HOSP/brief.pdf",
        "p1",
        roots,
    )
    assert c == "2023_HOSP/brief.pdf#p1"
    assert "[" not in c


def test_make_citation_multi_root(tmp_path: Path) -> None:
    """Multiple roots → citation has [root_label] bracket prefix."""
    root1 = tmp_path / "archive1"
    root2 = tmp_path / "archive2"
    roots = [root1, root2]
    c = _make_citation(
        str(root1 / "brief.pdf"),
        "brief.pdf",
        "p1",
        roots,
    )
    assert c.startswith("[archive1]")
    assert "brief.pdf#p1" in c


# ---------------------------------------------------------------------------
# Integration search tests (BM25 lane — no Ollama required)
# ---------------------------------------------------------------------------

def test_search_returns_top_k(tmp_path: Path, conn) -> None:
    """Search with top_k=3 returns at most 3 results."""
    cfg_obj = _cfg(tmp_path)
    root = tmp_path
    for i in range(8):
        _seed(conn, root, f"file{i}.txt",
              f"hospital architecture design brief {i}",
              project_id=f"2023_P{i:02d}")

    with patch("core.vectors.query_vector", return_value=[]):
        results = search("hospital", top_k=3, conn=conn, cfg_obj=cfg_obj)

    assert len(results) <= 3
    assert all("citation" in r for r in results)
    assert all("#" in r["citation"] for r in results)


def test_citation_format_single_root(tmp_path: Path, conn) -> None:
    """Single index_root → citations have no [bracket] prefix."""
    cfg_obj = _cfg(tmp_path, roots=[tmp_path])
    _seed(conn, tmp_path, "2023_HOSP/brief.txt",
          "hospital architecture brief project")

    with patch("core.vectors.query_vector", return_value=[]):
        results = search("hospital", conn=conn, cfg_obj=cfg_obj)

    assert results, "Expected at least one result"
    for r in results:
        assert "[" not in r["citation"], \
            f"Single-root citation must not have bracket prefix: {r['citation']}"
        assert "#" in r["citation"]
        assert "2023_HOSP" in r["citation"]


def test_citation_format_multi_root_has_bracket_prefix(tmp_path: Path, conn) -> None:
    """Multiple index_roots → citations include [root_label] prefix."""
    root1 = tmp_path / "archive1"
    root2 = tmp_path / "archive2"
    root1.mkdir(); root2.mkdir()
    cfg_obj = _cfg(tmp_path, roots=[root1, root2])

    _seed(conn, root1, "brief.txt", "hospital architecture design")

    with patch("core.vectors.query_vector", return_value=[]):
        results = search("hospital", conn=conn, cfg_obj=cfg_obj)

    assert results, "Expected at least one result"
    assert any("[archive1]" in r["citation"] for r in results), \
        f"Multi-root citation must have [archive1] prefix: {[r['citation'] for r in results]}"


def test_bm25_only_when_vector_unavailable(tmp_path: Path, conn) -> None:
    """Vector lane failure → BM25-only results returned, no crash."""
    cfg_obj = _cfg(tmp_path)
    _seed(conn, tmp_path, "brief.txt", "hospital architecture")

    with patch("core.vectors.query_vector", side_effect=Exception("Ollama down")):
        results = search("hospital", conn=conn, cfg_obj=cfg_obj)

    # Must not raise; BM25 alone produces results
    assert isinstance(results, list)
    assert len(results) >= 1
    for r in results:
        assert r["vector_score"] == 0.0, "No vector contribution when lane failed"


def test_filters_applied_to_search(tmp_path: Path, conn) -> None:
    """project_id filter must restrict results to that project only."""
    cfg_obj = _cfg(tmp_path)
    root = tmp_path

    _seed(conn, root, "hosp/brief.txt",   "architecture design hospital",
          project_id="2023_HOSP")
    _seed(conn, root, "school/brief.txt", "architecture design school",
          project_id="2022_SCHOOL")

    with patch("core.vectors.query_vector", return_value=[]):
        results = search(
            "hospital",
            filters={"project_id": "2023_HOSP"},
            conn=conn,
            cfg_obj=cfg_obj,
        )

    assert results, "Expected at least one result for 2023_HOSP"
    assert all(r["project_id"] == "2023_HOSP" for r in results), \
        "Filter must exclude non-matching projects"


def test_citation_validation_excludes_invalid(tmp_path: Path, conn) -> None:
    """
    A chunk whose file does NOT exist on disk gets an invalid citation
    and must be excluded from results.
    """
    cfg_obj = _cfg(tmp_path)
    root = tmp_path

    # Valid chunk: file exists on disk
    _seed(conn, root, "valid.txt", "hospital architecture brief valid",
          exists_on_disk=True)

    # Invalid chunk: file path is fake (file does NOT exist on disk)
    fake_path = (root / "ghost.txt").resolve().as_posix()
    fake_fid = file_id_from_path(fake_path)
    fake_cid = hashlib.sha256(f"{fake_fid}::p1".encode()).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO files "
        "(file_id, file_path, file_name, extension, status) "
        "VALUES (?, ?, 'ghost.txt', '.txt', 'INDEXED')",
        (fake_fid, fake_path),
    )
    conn.execute(
        "INSERT OR IGNORE INTO chunks "
        "(chunk_id, file_id, ref_value, text, embedded) "
        "VALUES (?, ?, 'p1', 'hospital architecture brief ghost', 1)",
        (fake_cid, fake_fid),
    )
    conn.commit()

    with patch("core.vectors.query_vector", return_value=[]):
        results = search("hospital", conn=conn, cfg_obj=cfg_obj)

    # The ghost file must not appear in results
    assert all("ghost" not in r["rel_path"] for r in results), \
        "Invalid-citation chunk must be excluded from results"
    # All returned citations must be valid (file exists + chunk in DB)
    assert all("#" in r["citation"] for r in results)


def test_hybrid_scoring_uses_both_lanes(tmp_path: Path, conn) -> None:
    """
    When vector lane returns a score, final_score must reflect both lanes.
    final = alpha * vector_score + (1-alpha) * bm25_score
    """
    cfg_obj = _cfg(tmp_path)
    root = tmp_path
    file_id, chunk_id = _seed(conn, root, "brief.txt", "hospital architecture")

    fake_vec_result = [{
        "chunk_id": chunk_id,
        "distance": 0.1,   # close match → score 0.9
        "metadata": {"file_id": file_id, "rel_path": "brief.txt",
                     "ref_value": "p1", "project_id": "Unknown",
                     "typology": "Unknown", "ext": ".txt", "root_id": ""},
    }]

    with patch("core.vectors.query_vector", return_value=fake_vec_result):
        results = search("hospital", conn=conn, cfg_obj=cfg_obj)

    assert results, "Expected at least one result"
    r = results[0]
    # With vector contributing, final_score should reflect hybrid
    assert r["vector_score"] > 0.0, "Vector score should be > 0"
    # final = alpha * vec + (1-alpha) * bm25
    expected = (cfg_obj.hybrid_alpha * r["vector_score"]
                + (1.0 - cfg_obj.hybrid_alpha) * r["bm25_score"])
    assert abs(r["final_score"] - expected) < 0.01


def test_session_query_saved_to_messages(tmp_path: Path, conn) -> None:
    """When session_id is given, the user query is saved to the messages table."""
    cfg_obj = _cfg(tmp_path)
    _seed(conn, tmp_path, "brief.txt", "hospital design architecture")
    session_id = "sess-test-001"

    with patch("core.vectors.query_vector", return_value=[]):
        search("hospital design", session_id=session_id,
               conn=conn, cfg_obj=cfg_obj)

    msgs = conn.execute(
        "SELECT role, content FROM messages WHERE session_id=?",
        (session_id,),
    ).fetchall()
    assert any(
        m["role"] == "user" and "hospital" in m["content"] for m in msgs
    ), f"User query must be saved to messages, got: {list(msgs)}"
