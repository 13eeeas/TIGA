"""
tests/test_ocr.py — selective OCR pass and confidence gate.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from config import load_config
from core.db import get_connection, upsert_file
from core.ocr import OcrResult, load_ocr_review_queue, run_ocr_pass


def _cfg(tmp_path: Path, *, enabled: bool = True):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.dump({
            "index_roots": [str(tmp_path)],
            "ocr": {
                "enabled": enabled,
                "on_empty_pdf": False,
                "confidence_threshold": 0.55,
                "max_files_per_run": 10,
            },
        }),
        encoding="utf-8",
    )
    return load_config(config_file=cfg_file, work_dir=tmp_path)


@pytest.fixture
def conn(tmp_path: Path):
    db = get_connection(tmp_path / "db" / "tiga.db")
    yield db
    db.close()


def test_ocr_pass_disabled_is_noop(tmp_path: Path, conn) -> None:
    cfg_obj = _cfg(tmp_path, enabled=False)
    stats = run_ocr_pass(conn, cfg_obj)
    assert stats["disabled"] == 1
    assert stats["indexed"] == 0


def test_ocr_pass_indexes_high_confidence(tmp_path: Path, conn) -> None:
    cfg_obj = _cfg(tmp_path, enabled=True)
    img = tmp_path / "scan_schedule.png"
    img.write_bytes(b"fake")
    upsert_file(conn, {
        "file_id": "fid-ocr-1",
        "file_path": str(img),
        "file_name": "scan_schedule.png",
        "project_id": "261",
        "typology": "Unknown",
        "extension": ".png",
        "lane": "METADATA_ONLY",
        "status": "EXTRACTED",
    })
    conn.commit()

    fake = OcrResult(
        text="Door schedule type A width 900 height 2100",
        confidence=0.9,
        source="image",
    )
    with patch("core.ocr.ocr_image_with_confidence", return_value=fake), \
         patch("core.ocr.classify_image", create=True):
        # classify_image is imported inside run_ocr_pass from core.extract
        with patch("core.extract.classify_image") as mock_cls:
            mock_cls.return_value = type(
                "IC", (), {"needs_ocr": True, "image_type": "scanned_document"}
            )()
            stats = run_ocr_pass(conn, cfg_obj)

    assert stats["indexed"] == 1
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE file_id=?", ("fid-ocr-1",)
    ).fetchone()
    assert row["n"] >= 1
    meta = conn.execute(
        "SELECT ocr_processed, ocr_text_length FROM files WHERE file_id=?",
        ("fid-ocr-1",),
    ).fetchone()
    assert meta["ocr_processed"] == 1
    assert meta["ocr_text_length"] > 0


def test_ocr_pass_queues_low_confidence(tmp_path: Path, conn) -> None:
    cfg_obj = _cfg(tmp_path, enabled=True)
    img = tmp_path / "blurry_scan.png"
    img.write_bytes(b"fake")
    upsert_file(conn, {
        "file_id": "fid-ocr-2",
        "file_path": str(img),
        "file_name": "blurry_scan.png",
        "project_id": "261",
        "typology": "Unknown",
        "extension": ".png",
        "lane": "METADATA_ONLY",
        "status": "EXTRACTED",
    })
    conn.commit()

    fake = OcrResult(text="??? xx", confidence=0.2, source="image")
    with patch("core.ocr.ocr_image_with_confidence", return_value=fake), \
         patch("core.extract.classify_image") as mock_cls:
        mock_cls.return_value = type(
            "IC", (), {"needs_ocr": True, "image_type": "scanned_document"}
        )()
        stats = run_ocr_pass(conn, cfg_obj)

    assert stats["queued_review"] == 1
    queue = load_ocr_review_queue(cfg_obj, limit=5)
    assert any(e.get("file_id") == "fid-ocr-2" for e in queue)
    chunks = conn.execute(
        "SELECT COUNT(*) AS n FROM chunks WHERE file_id=?", ("fid-ocr-2",)
    ).fetchone()
    assert chunks["n"] == 0
