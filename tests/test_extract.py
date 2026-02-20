"""
tests/test_extract.py — Extraction pipeline unit tests.

Run with: pytest tests/test_extract.py -v
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from config import load_config
from core.db import get_connection, file_id_from_path
from core.extract import extract_chunks, run_extract, MAX_PDF_CHARS

# ---------------------------------------------------------------------------
# Fixture paths
# ---------------------------------------------------------------------------

FIXTURES = Path(__file__).parent / "fixtures"


def _cfg(tmp_path: Path):
    """Minimal config for tests — points index_root at tmp_path."""
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


# ---------------------------------------------------------------------------
# extract_chunks — ref format tests
# ---------------------------------------------------------------------------

def test_extract_chunks_pdf_returns_page_refs() -> None:
    """sample.pdf (3 pages) → refs p1, p2, p3."""
    pairs = extract_chunks(FIXTURES / "sample.pdf")
    refs = [ref for ref, _ in pairs]
    assert "p1" in refs
    assert "p2" in refs
    assert "p3" in refs
    # All refs start with 'p'
    assert all(r.startswith("p") for r in refs)


def test_extract_chunks_pptx_returns_slide_refs() -> None:
    """sample.pptx (3 slides) → refs s01, s02, s03."""
    pairs = extract_chunks(FIXTURES / "sample.pptx")
    refs = [ref for ref, _ in pairs]
    assert "s01" in refs
    assert "s02" in refs
    assert "s03" in refs
    assert len(pairs) == 3


def test_extract_chunks_docx_splits_on_headings() -> None:
    """sample.docx (2 headings) → refs sec01, sec02."""
    pairs = extract_chunks(FIXTURES / "sample.docx")
    refs = [ref for ref, _ in pairs]
    assert "sec01" in refs
    assert "sec02" in refs
    assert len(pairs) == 2


def test_extract_chunks_txt_splits_on_sections() -> None:
    """sample.txt (3 # headings) → at least refs sec01, sec02, sec03."""
    pairs = extract_chunks(FIXTURES / "sample.txt")
    refs = [ref for ref, _ in pairs]
    assert "sec01" in refs
    assert "sec02" in refs
    assert "sec03" in refs


def test_extract_chunks_long_pdf_page_splits_with_suffix(tmp_path: Path) -> None:
    """
    A PDF page exceeding MAX_PDF_CHARS chars gets split into p1a, p1b, ...
    """
    from fpdf import FPDF

    # Create a single-page PDF with text well over MAX_PDF_CHARS
    long_text = ("Architecture is the learned game, correct and magnificent, "
                 "of forms assembled in the light. ") * 120  # ~6000+ chars
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=10)
    pdf.multi_cell(0, 5, long_text)
    pdf_path = tmp_path / "long_page.pdf"
    pdf.output(str(pdf_path))

    pairs = extract_chunks(pdf_path)
    refs = [ref for ref, _ in pairs]

    # Should NOT have bare "p1" — it should be p1a, p1b, ...
    assert "p1" not in refs, "long page should be split into p1a, p1b, ..."
    assert any(r.startswith("p1") and len(r) > 2 for r in refs), \
        f"expected p1a/p1b style refs, got: {refs}"
    # Text should be preserved across splits
    combined = " ".join(text for _, text in pairs)
    assert len(combined) > MAX_PDF_CHARS


# ---------------------------------------------------------------------------
# run_extract — DB integration tests
# ---------------------------------------------------------------------------

def test_run_extract_writes_chunks_to_db(tmp_path: Path, conn) -> None:
    """run_extract writes chunk rows with correct file_id and ref_values."""
    cfg_obj = _cfg(tmp_path)
    src = FIXTURES / "sample.txt"

    # Copy fixture into index_root so infer_project can find it
    dest = tmp_path / "sample.txt"
    dest.write_bytes(src.read_bytes())

    posix = dest.resolve().as_posix()
    file_id = file_id_from_path(posix)

    # Seed a files row (discover would normally do this)
    conn.execute(
        "INSERT OR IGNORE INTO files (file_id, file_path, file_name, status) "
        "VALUES (?, ?, ?, 'DISCOVERED')",
        (file_id, posix, dest.name),
    )
    conn.commit()

    stats = run_extract(conn, file_id, dest, "TEXT_EXTRACTABLE", cfg_obj=cfg_obj)

    chunks = conn.execute(
        "SELECT ref_value, text FROM chunks WHERE file_id=? ORDER BY ref_value",
        (file_id,),
    ).fetchall()

    assert stats["new"] >= 3, f"Expected >=3 new chunks, got {stats}"
    assert stats["failed"] == 0
    refs = {r["ref_value"] for r in chunks}
    assert "sec01" in refs
    assert "sec02" in refs
    # All chunks have non-empty text
    assert all(r["text"].strip() for r in chunks)


def test_run_extract_idempotent(tmp_path: Path, conn) -> None:
    """
    Running run_extract twice on the same unchanged file:
      first run  → all new
      second run → all unchanged, zero new
    """
    cfg_obj = _cfg(tmp_path)
    src = FIXTURES / "sample.docx"
    dest = tmp_path / "sample.docx"
    dest.write_bytes(src.read_bytes())

    posix = dest.resolve().as_posix()
    file_id = file_id_from_path(posix)
    conn.execute(
        "INSERT OR IGNORE INTO files (file_id, file_path, file_name, status) "
        "VALUES (?, ?, ?, 'DISCOVERED')",
        (file_id, posix, dest.name),
    )
    conn.commit()

    stats1 = run_extract(conn, file_id, dest, "TEXT_EXTRACTABLE", cfg_obj=cfg_obj)
    stats2 = run_extract(conn, file_id, dest, "TEXT_EXTRACTABLE", cfg_obj=cfg_obj)

    assert stats1["new"] > 0
    assert stats2["unchanged"] == stats1["new"]
    assert stats2["new"] == 0
    assert stats2["updated"] == 0


def test_run_extract_updates_changed_chunks(tmp_path: Path, conn) -> None:
    """
    Modifying a file between two extract runs triggers updated > 0 on the second run.
    """
    cfg_obj = _cfg(tmp_path)
    dest = tmp_path / "report.txt"
    dest.write_text(
        "# Section One\nOriginal content for the first section.\n",
        encoding="utf-8",
    )

    posix = dest.resolve().as_posix()
    file_id = file_id_from_path(posix)
    conn.execute(
        "INSERT OR IGNORE INTO files (file_id, file_path, file_name, status) "
        "VALUES (?, ?, ?, 'DISCOVERED')",
        (file_id, posix, dest.name),
    )
    conn.commit()

    stats1 = run_extract(conn, file_id, dest, "TEXT_EXTRACTABLE", cfg_obj=cfg_obj)
    assert stats1["new"] >= 1

    # Modify the file content
    dest.write_text(
        "# Section One\nRevised content with additional project information.\n",
        encoding="utf-8",
    )

    stats2 = run_extract(conn, file_id, dest, "TEXT_EXTRACTABLE", cfg_obj=cfg_obj)
    assert stats2["updated"] >= 1, f"Expected updates, got: {stats2}"
    assert stats2["failed"] == 0


# ---------------------------------------------------------------------------
# Older format support
# ---------------------------------------------------------------------------

def test_extract_chunks_doc_graceful_on_binary(tmp_path: Path) -> None:
    """.doc with OLE2 magic bytes (true binary) → graceful empty list, no crash."""
    f = tmp_path / "legacy.doc"
    f.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512)
    pairs = extract_chunks(f)
    assert isinstance(pairs, list), "extract_chunks must return a list"
    # True binary .doc returns [] gracefully


def test_extract_chunks_ppt_graceful(tmp_path: Path) -> None:
    """.ppt (no pure-Python parser) → graceful empty list, no crash."""
    f = tmp_path / "legacy.ppt"
    f.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1" + b"\x00" * 512)
    pairs = extract_chunks(f)
    assert isinstance(pairs, list)
    assert pairs == [], "PPT must return empty list (no pure-Python parser)"


def test_extract_chunks_xlsx_returns_sheet_refs(tmp_path: Path) -> None:
    """xlsx with data rows → returns sheet01 ref containing cell text."""
    try:
        import openpyxl
    except ImportError:
        pytest.skip("openpyxl not installed")

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Project", "Status", "Start Date"])
    ws.append(["2023_HOSP", "Active", "2023-01-15"])
    ws.append(["2022_RESI", "Complete", "2022-06-01"])
    path = tmp_path / "schedule.xlsx"
    wb.save(str(path))

    pairs = extract_chunks(path)
    refs = [ref for ref, _ in pairs]
    assert "sheet01" in refs, f"Expected sheet01 in {refs}"
    combined_text = " ".join(t for _, t in pairs)
    assert "2023_HOSP" in combined_text


def test_extract_chunks_xls_returns_sheet_refs(tmp_path: Path) -> None:
    """xls (xlrd) — if xlrd available, returns sheet refs with cell content."""
    try:
        import xlrd
        import xlwt  # noqa: F401 — only skip if xlwt not available for fixture creation
    except ImportError:
        pytest.skip("xlrd/xlwt not installed")

    import xlwt
    wb = xlwt.Workbook()
    ws = wb.add_sheet("Sheet1")
    ws.write(0, 0, "Project")
    ws.write(0, 1, "Budget")
    ws.write(1, 0, "2023_HOSP")
    ws.write(1, 1, 1500000)
    path = tmp_path / "budget.xls"
    wb.save(str(path))

    pairs = extract_chunks(path)
    refs = [ref for ref, _ in pairs]
    assert "sheet01" in refs, f"Expected sheet01 in {refs}"
    combined_text = " ".join(t for _, t in pairs)
    assert "2023_HOSP" in combined_text
