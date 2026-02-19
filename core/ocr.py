"""
core/ocr.py — Gated OCR via pytesseract + Pillow.

NEVER called from the main pipeline. Must be invoked explicitly.
Requires: Tesseract binary installed + pytesseract + Pillow packages.
"""

from __future__ import annotations

import logging
from pathlib import Path

from config import cfg

logger = logging.getLogger(__name__)


def ocr_image(path: Path) -> str:
    """
    Run OCR on a single image file.
    Returns extracted text or empty string on failure.
    Raises RuntimeError if OCR is disabled in config.
    """
    if not cfg.ocr_enabled:
        raise RuntimeError(
            "OCR is disabled. Set ocr.enabled: true in config.yaml to use it."
        )

    try:
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore

        if cfg.tesseract_cmd and cfg.tesseract_cmd != "tesseract":
            pytesseract.pytesseract.tesseract_cmd = cfg.tesseract_cmd

        img = Image.open(str(path))
        text = pytesseract.image_to_string(img)
        return text.strip()
    except ImportError as e:
        logger.error("OCR dependencies not installed: %s", e)
        return ""
    except Exception as e:
        logger.warning("OCR failed for %s: %s", path.name, e)
        return ""


def ocr_pdf_pages(path: Path, dpi: int = 200) -> str:
    """
    Render each PDF page as an image and OCR it.
    For scanned PDFs where pypdf returns no text.
    Requires pdf2image + poppler (not in base requirements — install separately).
    """
    if not cfg.ocr_enabled:
        raise RuntimeError("OCR is disabled in config.")

    try:
        from pdf2image import convert_from_path  # type: ignore

        images = convert_from_path(str(path), dpi=dpi)
        pages = []
        for img in images:
            import pytesseract  # type: ignore
            pages.append(pytesseract.image_to_string(img))
        return "\n".join(pages).strip()
    except ImportError as e:
        logger.error("pdf2image or poppler not installed: %s", e)
        return ""
    except Exception as e:
        logger.warning("PDF OCR failed for %s: %s", path.name, e)
        return ""
