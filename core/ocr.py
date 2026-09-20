"""
core/ocr.py — Gated OCR via pytesseract + Pillow.

Selective Tier-C ingest: only runs when ocr.enabled is true, and only on
candidate images / empty PDFs. Low-confidence pulls go to ocr_review.log
instead of being auto-indexed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config import cfg as _module_cfg, Config
from core.db import upsert_chunk

logger = logging.getLogger(__name__)

_IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".bmp", ".webp",
}
_OCR_REVIEW_LOG = "ocr_review.log"


@dataclass
class OcrResult:
    text: str
    confidence: float  # 0–1
    source: str = "image"


def ocr_image(path: Path, cfg_obj: Config | None = None) -> str:
    """
    Run OCR on a single image file.
    Returns extracted text or empty string on failure.
    Raises RuntimeError if OCR is disabled in config.
    """
    result = ocr_image_with_confidence(path, cfg_obj)
    return result.text


def ocr_image_with_confidence(
    path: Path,
    cfg_obj: Config | None = None,
) -> OcrResult:
    """OCR an image and estimate mean word confidence (0–1)."""
    _cfg = cfg_obj or _module_cfg
    if not _cfg.ocr_enabled:
        raise RuntimeError(
            "OCR is disabled. Set ocr.enabled: true in config.yaml to use it."
        )

    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        if _cfg.tesseract_cmd and _cfg.tesseract_cmd != "tesseract":
            pytesseract.pytesseract.tesseract_cmd = _cfg.tesseract_cmd

        img = Image.open(str(path))
        text = (pytesseract.image_to_string(img) or "").strip()
        conf = _mean_tesseract_confidence(pytesseract, img)
        return OcrResult(text=text, confidence=conf, source="image")
    except ImportError as e:
        logger.error("OCR dependencies not installed: %s", e)
        return OcrResult(text="", confidence=0.0, source="image")
    except Exception as e:
        logger.warning("OCR failed for %s: %s", path.name, e)
        return OcrResult(text="", confidence=0.0, source="image")


def _mean_tesseract_confidence(pytesseract: Any, img: Any) -> float:
    try:
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        confs = [
            float(c)
            for c in data.get("conf", [])
            if str(c) not in ("-1", "") and float(c) >= 0
        ]
        if not confs:
            return 0.0
        return max(0.0, min(1.0, (sum(confs) / len(confs)) / 100.0))
    except Exception:
        return 0.5


def ocr_pdf_pages(
    path: Path,
    dpi: int = 200,
    cfg_obj: Config | None = None,
) -> str:
    """
    Render each PDF page as an image and OCR it.
    For scanned PDFs where pypdf returns no text.
    Requires pdf2image + poppler (not in base requirements — install separately).
    """
    result = ocr_pdf_pages_with_confidence(path, dpi=dpi, cfg_obj=cfg_obj)
    return result.text


def ocr_pdf_pages_with_confidence(
    path: Path,
    dpi: int = 200,
    cfg_obj: Config | None = None,
) -> OcrResult:
    _cfg = cfg_obj or _module_cfg
    if not _cfg.ocr_enabled:
        raise RuntimeError("OCR is disabled in config.")

    try:
        from pdf2image import convert_from_path  # type: ignore
        import pytesseract  # type: ignore

        if _cfg.tesseract_cmd and _cfg.tesseract_cmd != "tesseract":
            pytesseract.pytesseract.tesseract_cmd = _cfg.tesseract_cmd

        images = convert_from_path(str(path), dpi=dpi)
        pages: list[str] = []
        confs: list[float] = []
        for img in images:
            pages.append(pytesseract.image_to_string(img) or "")
            confs.append(_mean_tesseract_confidence(pytesseract, img))
        text = "\n".join(pages).strip()
        conf = sum(confs) / len(confs) if confs else 0.0
        return OcrResult(text=text, confidence=conf, source="pdf")
    except ImportError as e:
        logger.error("pdf2image or poppler not installed: %s", e)
        return OcrResult(text="", confidence=0.0, source="pdf")
    except Exception as e:
        logger.warning("PDF OCR failed for %s: %s", path.name, e)
        return OcrResult(text="", confidence=0.0, source="pdf")


def _append_ocr_review(cfg_obj: Config, entry: dict[str, Any]) -> None:
    log_dir = cfg_obj.work_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / _OCR_REVIEW_LOG
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _chunk_ocr_text(file_id: str, text: str, source: str) -> list[dict[str, Any]]:
    """Split OCR text into ~1200-char chunks for upsert."""
    cleaned = (text or "").strip()
    if not cleaned:
        return []
    size = 1200
    chunks: list[dict[str, Any]] = []
    for i in range(0, len(cleaned), size):
        body = cleaned[i : i + size].strip()
        if not body:
            continue
        idx = len(chunks) + 1
        ref = f"ocr{idx:02d}"
        chunk_id = hashlib.sha256(f"{file_id}::{ref}".encode()).hexdigest()
        chunks.append({
            "chunk_id": chunk_id,
            "file_id": file_id,
            "ref_value": ref,
            "text": body,
            "token_estimate": len(body.split()),
            "content_hash": hashlib.sha256(body.encode()).hexdigest(),
            "source": source,
        })
    return chunks


def _mark_ocr_state(
    conn: sqlite3.Connection,
    file_id: str,
    *,
    processed: int,
    text_length: int,
) -> None:
    conn.execute(
        "UPDATE files SET ocr_processed=?, ocr_text_length=? WHERE file_id=?",
        (processed, text_length, file_id),
    )


def run_ocr_pass(
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,
    *,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, int]:
    """
    Selective OCR pass over pending image / empty-PDF candidates.

    High-confidence text → upsert chunks + mark ocr_processed=1.
    Low-confidence / empty → append ocr_review.log and leave unprocessed.
    """
    _cfg = cfg_obj or _module_cfg
    stats = {
        "candidates": 0,
        "indexed": 0,
        "queued_review": 0,
        "skipped": 0,
        "disabled": 0,
    }

    if not _cfg.ocr_enabled:
        stats["disabled"] = 1
        logger.info("OCR pass skipped — ocr.enabled is false")
        return stats

    from core.extract import classify_image

    threshold = float(getattr(_cfg, "ocr_confidence_threshold", 0.55))
    max_files = int(getattr(_cfg, "ocr_max_files_per_run", 50))

    # Image candidates: METADATA_ONLY images not yet OCR-processed
    image_rows = conn.execute(
        "SELECT file_id, file_path, extension FROM files "
        "WHERE COALESCE(ocr_processed, 0)=0 "
        "AND lane='METADATA_ONLY' "
        "AND LOWER(extension) IN (" + ",".join("?" * len(_IMAGE_EXTS)) + ") "
        "LIMIT ?",
        [*list(_IMAGE_EXTS), max_files],
    ).fetchall()

    candidates: list[tuple[str, str, str]] = []
    for row in image_rows:
        path = Path(row["file_path"])
        try:
            img_cls = classify_image(path)
        except Exception:
            img_cls = None
        # Prefer scans / schedule-like paths; still allow unknown when OCR on
        if img_cls is None or img_cls.needs_ocr or img_cls.image_type in (
            "scanned_document", "unknown",
        ):
            candidates.append((row["file_id"], row["file_path"], "image"))

    remaining = max(0, max_files - len(candidates))
    if remaining and getattr(_cfg, "ocr_on_empty_pdf", False):
        pdf_rows = conn.execute(
            "SELECT f.file_id, f.file_path FROM files f "
            "WHERE COALESCE(f.ocr_processed, 0)=0 "
            "AND f.lane='TEXT_EXTRACTABLE' "
            "AND LOWER(f.extension)='.pdf' "
            "AND NOT EXISTS ("
            "  SELECT 1 FROM chunks c WHERE c.file_id=f.file_id "
            "  AND LENGTH(COALESCE(c.text,'')) > 40"
            ") "
            "LIMIT ?",
            (remaining,),
        ).fetchall()
        for row in pdf_rows:
            candidates.append((row["file_id"], row["file_path"], "pdf"))

    stats["candidates"] = len(candidates)
    if progress:
        progress({
            "phase": "ocr",
            "processed": 0,
            "total": len(candidates),
            "detail": f"{len(candidates)} OCR candidates",
        })

    for idx, (file_id, file_path, kind) in enumerate(candidates, start=1):
        path = Path(file_path)
        try:
            if kind == "pdf":
                result = ocr_pdf_pages_with_confidence(path, cfg_obj=_cfg)
            else:
                result = ocr_image_with_confidence(path, cfg_obj=_cfg)

            text = (result.text or "").strip()
            conf = float(result.confidence)

            if not text or conf < threshold:
                _append_ocr_review(_cfg, {
                    "timestamp": datetime.now().isoformat(),
                    "file_id": file_id,
                    "file_path": file_path,
                    "kind": kind,
                    "confidence": round(conf, 3),
                    "text_preview": text[:240],
                    "status": "queued",
                    "reason": "empty" if not text else "low_confidence",
                })
                _mark_ocr_state(conn, file_id, processed=0, text_length=len(text))
                stats["queued_review"] += 1
            else:
                for chunk in _chunk_ocr_text(file_id, text, kind):
                    upsert_chunk(conn, chunk)
                _mark_ocr_state(conn, file_id, processed=1, text_length=len(text))
                # Ensure downstream index/embed picks the file up
                conn.execute(
                    "UPDATE files SET status='EXTRACTED' "
                    "WHERE file_id=? AND status IN ('EXTRACTED','INDEXED','EMBEDDED','DISCOVERED')",
                    (file_id,),
                )
                stats["indexed"] += 1
        except Exception as e:
            logger.warning("OCR candidate failed %s: %s", file_path, e)
            stats["skipped"] += 1

        if progress:
            progress({
                "phase": "ocr",
                "processed": idx,
                "total": len(candidates),
            })

    conn.commit()
    logger.info("OCR pass complete: %s", stats)
    return stats


def load_ocr_review_queue(
    cfg_obj: Config | None = None,
    *,
    limit: int = 20,
) -> list[dict[str, Any]]:
    _cfg = cfg_obj or _module_cfg
    path = _cfg.work_dir / "logs" / _OCR_REVIEW_LOG
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except Exception:
                continue
    return entries[-limit:]
