"""
core/extract.py — Text extraction from supported file types.

Returns (text, surrogate) tuple:
  text      : extracted body text (may be empty)
  surrogate : human-readable stand-in used for search when text is empty

OCR is NOT called here. See core/ocr.py for opt-in OCR.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Extraction dispatch
# ---------------------------------------------------------------------------

def extract(path: Path) -> tuple[str, str]:
    """
    Extract text from a file.
    Returns (text, surrogate). Neither will be None.
    Errors are caught and logged; returns ("", surrogate) on failure.
    """
    ext = path.suffix.lower()

    if ext == ".pdf":
        return _extract_pdf(path)
    elif ext in (".docx", ".doc"):
        return _extract_docx(path)
    elif ext in (".pptx", ".ppt"):
        return _extract_pptx(path)
    elif ext in (".txt", ".md"):
        return _extract_text(path)
    else:
        # Unsupported — return surrogate only
        surrogate = _build_surrogate(path, "no text extraction available")
        return "", surrogate


# ---------------------------------------------------------------------------
# Format extractors
# ---------------------------------------------------------------------------

def _extract_pdf(path: Path) -> tuple[str, str]:
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(str(path), strict=False)
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:
                pages.append("")
        text = "\n".join(pages).strip()
        surrogate = _build_surrogate(path, f"{len(reader.pages)} pages")
        return text, surrogate
    except Exception as e:
        logger.warning("PDF extraction failed for %s: %s", path.name, e)
        return "", _build_surrogate(path, "PDF extraction failed")


def _extract_docx(path: Path) -> tuple[str, str]:
    try:
        from docx import Document  # type: ignore

        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        text = "\n".join(paragraphs).strip()
        surrogate = _build_surrogate(path, f"{len(paragraphs)} paragraphs")
        return text, surrogate
    except Exception as e:
        logger.warning("DOCX extraction failed for %s: %s", path.name, e)
        return "", _build_surrogate(path, "DOCX extraction failed")


def _extract_pptx(path: Path) -> tuple[str, str]:
    try:
        from pptx import Presentation  # type: ignore

        prs = Presentation(str(path))
        slides_text: list[str] = []
        for slide in prs.slides:
            parts: list[str] = []
            for shape in slide.shapes:
                if shape.has_text_frame:
                    for para in shape.text_frame.paragraphs:
                        line = para.text.strip()
                        if line:
                            parts.append(line)
            if parts:
                slides_text.append(" ".join(parts))
        text = "\n".join(slides_text).strip()
        surrogate = _build_surrogate(path, f"{len(prs.slides)} slides")
        return text, surrogate
    except Exception as e:
        logger.warning("PPTX extraction failed for %s: %s", path.name, e)
        return "", _build_surrogate(path, "PPTX extraction failed")


def _extract_text(path: Path) -> tuple[str, str]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        surrogate = _build_surrogate(path, "plain text")
        return text, surrogate
    except Exception as e:
        logger.warning("Text extraction failed for %s: %s", path.name, e)
        return "", _build_surrogate(path, "text read failed")


# ---------------------------------------------------------------------------
# Surrogate builder
# ---------------------------------------------------------------------------

def _build_surrogate(path: Path, note: str) -> str:
    """
    Build a descriptive surrogate for files with no extractable text.
    Used by FTS and vector indexing so the file is still discoverable.
    """
    parts = list(path.parts)
    # Include up to 4 path components for context
    context_parts = parts[-4:] if len(parts) >= 4 else parts
    breadcrumb = " > ".join(context_parts)
    return f"{path.stem} | {path.suffix.lstrip('.')} | {note} | {breadcrumb}"


# ---------------------------------------------------------------------------
# Word count
# ---------------------------------------------------------------------------

def word_count(text: str) -> int:
    return len(text.split()) if text else 0
