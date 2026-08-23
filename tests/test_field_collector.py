"""tests/test_field_collector.py — Office field data collector."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import yaml

from config import load_config
from core.db import get_connection
from core.field_collector import (
    add_label,
    collect_status,
    export_bundle,
    import_bundle,
    load_labels,
    record_search_event,
)


def _cfg(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.dump({
            "index_roots": [str(tmp_path / "archive")],
            "field_collect": {"enabled": True},
        }),
        encoding="utf-8",
    )
    return load_config(config_file=cfg_file, work_dir=tmp_path)


def test_record_and_export_import_roundtrip(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()

    record_search_event(
        query="tianmu brief",
        mode="semantic",
        project_code="261",
        confidence=0.4,
        duration_ms=120.0,
        session_id="s1",
        results=[
            {
                "rel_path": "261/brief.txt",
                "file_path": str(tmp_path / "archive" / "261/brief.txt"),
                "citation": "261/brief.txt#p1",
                "project_id": "261",
                "final_score": 0.7,
                "snippet": "residential units",
            }
        ],
        cfg_obj=cfg,
    )

    add_label(
        "tianmu brief",
        ["261/brief.txt"],
        notes="gold",
        cfg_obj=cfg,
    )

    conn = get_connection(cfg.get_db_path())
    conn.execute(
        "INSERT INTO feedback (query, rating, comment) VALUES ('bad query', -1, 'wrong file')"
    )
    conn.commit()

    zip_path = export_bundle(conn, cfg_obj=cfg)
    conn.close()
    assert zip_path.exists()

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        assert "manifest.json" in names
        assert "search_events.jsonl" in names
        assert "office_eval.yaml" in names
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["counts"]["search_events"] == 1

    # Import into fresh work dir
    dev = tmp_path / "dev"
    dev.mkdir()
    (dev / "archive").mkdir()
    dev_cfg = _cfg(dev)
    dev_cfg.ensure_dirs()
    result = import_bundle(zip_path, cfg_obj=dev_cfg)
    assert result["merged"]["events_added"] == 1
    assert result["merged"]["labels_added"] >= 1
    assert len(load_labels(dev_cfg)) >= 1


def test_collect_status(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    cfg.ensure_dirs()
    conn = get_connection(cfg.get_db_path())
    st = collect_status(conn, cfg)
    conn.close()
    assert st["enabled"] is True
    assert "search_events" in st
