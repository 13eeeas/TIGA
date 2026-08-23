"""Tests for corpus-adaptive query generation and POC test flow."""

from __future__ import annotations

from pathlib import Path

import pytest

from core.db import get_connection
from core.pipeline_validate import (
    DEFAULT_FIXTURE_ARCHIVE,
    mock_embed_if_needed,
    run_index_validation,
    setup_sandbox,
)
from core.query_generator import generate_queries_for_projects
from core.poc_test import import_poc_test_bundle, run_poc_test


@pytest.fixture
def indexed_sandbox(tmp_path: Path):
    cfg, archive = setup_sandbox(tmp_path / "poc_sandbox", DEFAULT_FIXTURE_ARCHIVE)
    conn = get_connection(cfg.get_db_path())
    with mock_embed_if_needed(True):
        run_index_validation(cfg, conn, mock_embed=True)
    yield cfg, conn, archive
    conn.close()


def test_generate_queries_from_fixture_corpus(indexed_sandbox) -> None:
    cfg, conn, _archive = indexed_sandbox
    rows = conn.execute(
        "SELECT DISTINCT project_id FROM files WHERE status='INDEXED'"
    ).fetchall()
    pids = [r["project_id"] for r in rows if r["project_id"] != "Unknown"]
    if not pids:
        pids = ["261_tianmu", "262_hotel", "263_school"]
    report = generate_queries_for_projects(conn, pids, cfg.index_roots)
    assert report.queries
    d = report.to_dict()
    assert d["literal"] >= 1
    assert d["paraphrase"] >= 1
    for q in report.queries:
        assert q.expected_paths
        assert q.query


def test_poc_test_stress_and_export(indexed_sandbox) -> None:
    cfg, conn, archive = indexed_sandbox
    selected = [
        {
            "path": str(archive / "261_tianmu"),
            "name": "261_tianmu",
            "project_id": "261_tianmu",
        }
    ]
    result = run_poc_test(
        selected,
        cfg,
        skip_index=True,
        verbose=False,
    )
    assert result["generation"]["total"] >= 1
    assert "literal_recall_pct" in result["stress"]
    assert Path(result["export_path"]).exists()

    imp = import_poc_test_bundle(Path(result["export_path"]), cfg=cfg)
    assert Path(imp["playbook"]).exists()
