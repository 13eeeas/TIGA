"""
tests/test_index.py — Indexing pipeline unit tests.

Run with: pytest tests/test_index.py -v

All tests use real fixture files and mock Ollama via
patch("core.vectors.upsert_chunk") so no running Ollama is required.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from contextlib import contextmanager
from unittest.mock import patch, MagicMock

import pytest
import yaml

from config import load_config
from core.db import get_connection, file_id_from_path, fts_search
from core.index import run_embed, run_fts, run_index, run_rebuild

FIXTURES = Path(__file__).parent / "fixtures"


class _FakeMergeBuilder:
    def when_matched_update_all(self):
        return self

    def when_not_matched_insert_all(self):
        return self

    def execute(self, _rows):
        return None


class _FakeTable:
    def merge_insert(self, _key):
        return _FakeMergeBuilder()


@contextmanager
def _mock_embed_success():
    """Mock embedding + LanceDB merge path so no Ollama service is required."""
    with patch("core.index._vectors.embed_texts_batched", side_effect=lambda texts, _cfg: [[0.1] * 768 for _ in texts]), \
         patch("core.index._vectors._get_chunk_table", return_value=_FakeTable()), \
         patch("lancedb.connect", return_value=MagicMock()):
        yield


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _cfg(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.dump({"index_roots": [str(tmp_path)]}), encoding="utf-8"
    )
    return load_config(config_file=cfg_file, work_dir=tmp_path)


@pytest.fixture
def conn(tmp_path: Path):
    db = get_connection(tmp_path / "db" / "tiga.db")
    yield db
    db.close()


def _seed_extracted_file(
    conn,
    tmp_path: Path,
    fixture: str = "sample.txt",
    chunk_text: str = "architecture design brief project",
) -> tuple[str, str]:
    """
    Copy a fixture into tmp_path, create a files row (status=EXTRACTED),
    and insert one chunk. Returns (file_id, chunk_id).
    """
    src = FIXTURES / fixture
    dest = tmp_path / fixture
    dest.write_bytes(src.read_bytes())

    posix = dest.resolve().as_posix()
    file_id = file_id_from_path(posix)

    conn.execute(
        "INSERT OR IGNORE INTO files "
        "(file_id, file_path, file_name, extension, status) "
        "VALUES (?, ?, ?, '.txt', 'EXTRACTED')",
        (file_id, posix, dest.name),
    )

    chunk_id = hashlib.sha256(f"{file_id}::sec01".encode()).hexdigest()
    c_hash = hashlib.sha256(chunk_text.encode()).hexdigest()
    conn.execute(
        "INSERT OR IGNORE INTO chunks "
        "(chunk_id, file_id, ref_value, text, content_hash) "
        "VALUES (?, ?, 'sec01', ?, ?)",
        (chunk_id, file_id, chunk_text, c_hash),
    )
    conn.commit()
    return file_id, chunk_id


# ---------------------------------------------------------------------------
# Step 1: run_embed
# ---------------------------------------------------------------------------

def test_run_embed_sets_embedded_flag(tmp_path: Path, conn) -> None:
    """Successful embed → chunk.embedded=1."""
    cfg_obj = _cfg(tmp_path)
    file_id, chunk_id = _seed_extracted_file(conn, tmp_path)

    with _mock_embed_success():
        stats = run_embed(conn, cfg_obj)

    row = conn.execute(
        "SELECT embedded FROM chunks WHERE chunk_id=?", (chunk_id,)
    ).fetchone()
    assert row["embedded"] == 1
    assert stats["chunks_embedded"] == 1
    assert stats["chunks_skipped"] == 0


def test_run_embed_file_advances_to_embedded(tmp_path: Path, conn) -> None:
    """After all chunks embedded → file status becomes EMBEDDED."""
    cfg_obj = _cfg(tmp_path)
    file_id, _ = _seed_extracted_file(conn, tmp_path)

    with _mock_embed_success():
        stats = run_embed(conn, cfg_obj)

    status = conn.execute(
        "SELECT status FROM files WHERE file_id=?", (file_id,)
    ).fetchone()["status"]
    assert status == "EMBEDDED"
    assert stats["files_embedded"] == 1


def test_run_embed_chunk_error_skips_not_fails_file(tmp_path: Path, conn) -> None:
    """
    embed_texts_batched returning [None] → chunk stays embedded=0,
    file stays EXTRACTED (not all chunks done).
    """
    cfg_obj = _cfg(tmp_path)
    file_id, chunk_id = _seed_extracted_file(conn, tmp_path)

    with patch("core.index._vectors.embed_texts_batched", return_value=[None]):
        stats = run_embed(conn, cfg_obj)

    row = conn.execute(
        "SELECT embedded FROM chunks WHERE chunk_id=?", (chunk_id,)
    ).fetchone()
    assert row["embedded"] == 0, "Failed chunk must remain embedded=0"

    file_status = conn.execute(
        "SELECT status FROM files WHERE file_id=?", (file_id,)
    ).fetchone()["status"]
    assert file_status == "EXTRACTED", \
        "File must stay EXTRACTED when not all chunks are embedded"

    assert stats["chunks_skipped"] == 1
    assert stats["files_embedded"] == 0


def test_run_embed_skips_already_embedded_chunks(tmp_path: Path, conn) -> None:
    """Chunk already embedded=1 → no re-embed requested, file→EMBEDDED."""
    cfg_obj = _cfg(tmp_path)
    file_id, chunk_id = _seed_extracted_file(conn, tmp_path)

    # Pre-mark chunk as embedded
    conn.execute("UPDATE chunks SET embedded=1 WHERE chunk_id=?", (chunk_id,))
    conn.commit()

    # embed_texts_batched must not be called (no pending chunks to embed)
    with patch("core.index._vectors.embed_texts_batched") as mock_batched:
        stats = run_embed(conn, cfg_obj)
        mock_batched.assert_not_called()

    assert stats["chunks_embedded"] == 0
    # File should advance (all chunks are embedded=1, remaining=0)
    file_status = conn.execute(
        "SELECT status FROM files WHERE file_id=?", (file_id,)
    ).fetchone()["status"]
    assert file_status == "EMBEDDED"


# ---------------------------------------------------------------------------
# Step 2: run_fts
# ---------------------------------------------------------------------------

def test_run_fts_advances_embedded_to_indexed(tmp_path: Path, conn) -> None:
    """File in EMBEDDED status → run_fts sets status to INDEXED."""
    cfg_obj = _cfg(tmp_path)
    file_id, chunk_id = _seed_extracted_file(conn, tmp_path)

    # Simulate embed step having completed
    conn.execute("UPDATE chunks SET embedded=1 WHERE chunk_id=?", (chunk_id,))
    conn.execute(
        "UPDATE files SET status='EMBEDDED' WHERE file_id=?", (file_id,)
    )
    conn.commit()

    stats = run_fts(conn, cfg_obj)

    file_status = conn.execute(
        "SELECT status FROM files WHERE file_id=?", (file_id,)
    ).fetchone()["status"]
    assert file_status == "INDEXED"
    assert stats["files_indexed"] == 1


def test_run_fts_chunk_searchable_via_fts(tmp_path: Path, conn) -> None:
    """
    Chunk inserted via seed → FTS trigger populates chunks_fts →
    after run_fts, fts_search finds the chunk.
    """
    cfg_obj = _cfg(tmp_path)
    file_id, chunk_id = _seed_extracted_file(
        conn, tmp_path, chunk_text="hospital brief tender submission"
    )
    conn.execute("UPDATE chunks SET embedded=1 WHERE chunk_id=?", (chunk_id,))
    conn.execute(
        "UPDATE files SET status='EMBEDDED' WHERE file_id=?", (file_id,)
    )
    conn.commit()

    run_fts(conn, cfg_obj)

    results = fts_search(conn, "hospital", top_k=5)
    assert any(r["chunk_id"] == chunk_id for r in results), \
        f"Expected chunk_id {chunk_id[:12]}... in FTS results, got: {results}"


# ---------------------------------------------------------------------------
# Combined pipeline: run_index
# ---------------------------------------------------------------------------

def test_run_index_full_pipeline(tmp_path: Path, conn) -> None:
    """EXTRACTED → INDEXED through one run_index call (mocked Ollama)."""
    cfg_obj = _cfg(tmp_path)
    file_id, chunk_id = _seed_extracted_file(conn, tmp_path)

    with _mock_embed_success():
        stats = run_index(conn, cfg_obj)

    file_status = conn.execute(
        "SELECT status FROM files WHERE file_id=?", (file_id,)
    ).fetchone()["status"]
    assert file_status == "INDEXED"
    assert stats["files_embedded"] == 1
    assert stats["chunks_embedded"] == 1
    assert stats["files_indexed"] == 1


def test_run_index_idempotent(tmp_path: Path, conn) -> None:
    """
    Running run_index twice:
      first run  → files_indexed=1
      second run → files_indexed=0 (nothing left to process)
    FTS must not contain duplicate entries for the same chunk.
    """
    cfg_obj = _cfg(tmp_path)
    file_id, chunk_id = _seed_extracted_file(
        conn, tmp_path, chunk_text="idempotency test architecture"
    )

    with _mock_embed_success():
        stats1 = run_index(conn, cfg_obj)
        stats2 = run_index(conn, cfg_obj)

    assert stats1["files_indexed"] == 1
    # Second run: file already INDEXED → nothing to embed or FTS-index
    assert stats2["files_indexed"] == 0
    assert stats2["chunks_embedded"] == 0

    # FTS must have exactly one hit, not duplicated
    results = fts_search(conn, "idempotency", top_k=20)
    hits = [r for r in results if r["chunk_id"] == chunk_id]
    assert len(hits) == 1, f"FTS must not duplicate chunk entries: {len(hits)} hits"


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------

def test_run_rebuild_resets_and_reindexes(tmp_path: Path, conn) -> None:
    """
    After full index, run_rebuild:
      - resets INDEXED → EXTRACTED
      - resets embedded=0
      - re-runs index pipeline → back to INDEXED
    """
    cfg_obj = _cfg(tmp_path)
    file_id, chunk_id = _seed_extracted_file(conn, tmp_path)

    # First index pass
    with _mock_embed_success():
        run_index(conn, cfg_obj)

    pre_status = conn.execute(
        "SELECT status FROM files WHERE file_id=?", (file_id,)
    ).fetchone()["status"]
    assert pre_status == "INDEXED"

    # Rebuild — mock both vector upsert and LanceDB reset
    with _mock_embed_success(), \
         patch("core.index._reset_lancedb"):
        stats = run_rebuild(conn, cfg_obj)

    post_status = conn.execute(
        "SELECT status FROM files WHERE file_id=?", (file_id,)
    ).fetchone()["status"]
    assert post_status == "INDEXED", \
        "After rebuild file must be back to INDEXED"
    assert stats["files_indexed"] >= 1
    assert stats["chunks_embedded"] >= 1

    # Chunk should still be searchable after rebuild
    results = fts_search(conn, "architecture", top_k=5)
    assert any(r["chunk_id"] == chunk_id for r in results)
