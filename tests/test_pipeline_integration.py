"""
tests/test_pipeline_integration.py — End-to-end index + search benchmark on fixture archive.

No NAS, no Ollama, no API. Proves pipeline logic before real indexing.

Run: pytest tests/test_pipeline_integration.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.pipeline_validate import (
    DEFAULT_DUAL_BENCHMARK,
    DEFAULT_FIXTURE_ARCHIVE,
    DEFAULT_SEARCH_BENCHMARK,
    mock_embed_if_needed,
    run_index_validation,
    run_search_benchmark,
    setup_sandbox,
)
from core.db import get_connection


@pytest.fixture
def sandbox(tmp_path: Path):
    cfg, archive = setup_sandbox(tmp_path / "validate_sandbox", DEFAULT_FIXTURE_ARCHIVE)
    conn = get_connection(cfg.get_db_path())
    yield cfg, conn, archive
    conn.close()


def test_fixture_archive_has_three_projects() -> None:
    roots = list(DEFAULT_FIXTURE_ARCHIVE.iterdir())
    assert len(roots) >= 3


def test_full_index_pipeline_on_fixtures(sandbox) -> None:
    cfg, conn, _archive = sandbox
    with mock_embed_if_needed(True):
        report = run_index_validation(cfg, conn, mock_embed=True)
    assert report["ok"], report["errors"]
    assert report["files_indexed"] >= 4
    assert report["chunk_count"] >= 4
    assert report["fts_row_count"] >= 4


def test_search_benchmark_without_api(sandbox) -> None:
    cfg, conn, _archive = sandbox
    with mock_embed_if_needed(True):
        run_index_validation(cfg, conn, mock_embed=True)
        report = run_search_benchmark(
            cfg, DEFAULT_SEARCH_BENCHMARK, verbose=False
        )
    assert report["invalid_citations"] == 0
    assert report["top5_recall"] >= 0.8, (
        f"Expected >=80% recall on fixture benchmark, got {report['top5_recall_pct']}%"
    )


def test_dual_literal_paraphrase_benchmark(sandbox) -> None:
    """Google-close Hunt: literal + paraphrase gates without API."""
    cfg, conn, _archive = sandbox
    with mock_embed_if_needed(True):
        run_index_validation(cfg, conn, mock_embed=True)
        report = run_search_benchmark(cfg, DEFAULT_DUAL_BENCHMARK, verbose=False)
    assert report["mode"] == "search_only_dual"
    assert report["literal_recall_pct"] >= 90.0, report
    assert report["paraphrase_recall_pct"] >= 80.0, (
        f"Paraphrase recall {report['paraphrase_recall_pct']}% — "
        "domain expand / path boost should clear 80% on fixtures"
    )
    assert report["gateway_pass"] is True


def test_idempotent_reindex(sandbox) -> None:
    cfg, conn, _archive = sandbox
    with mock_embed_if_needed(True):
        first = run_index_validation(cfg, conn, mock_embed=True)
        second = run_index_validation(cfg, conn, mock_embed=True)
    assert first["files_indexed"] >= 4
    assert second["files_indexed"] >= 4
