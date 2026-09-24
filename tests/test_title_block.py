"""Title-block field parsing from stored PDF page text."""

from core.db import get_connection, upsert_chunk, upsert_file
from core.title_block import backfill_title_fields, parse_title_block


def test_parse_title_block_prefers_tail():
    text = ("notes about level 2 " * 40) + "\nREV C\n2024-06-15\n"
    parsed = parse_title_block(text)
    assert parsed["revision"] == 3
    assert parsed["file_date"] == "2024-06-15"


def test_parse_numeric_revision():
    parsed = parse_title_block("Sheet title\nRevision: 07\n")
    assert parsed["revision"] == 7


def test_backfill_updates_only_empty_fields(tmp_path):
    conn = get_connection(tmp_path / "db" / "tiga.db")
    upsert_file(conn, {
        "file_id": "pdf-1",
        "file_path": "F:/Shared/Projects/283 HICA/drawing.pdf",
        "file_name": "drawing.pdf",
        "extension": ".pdf",
        "lane": "TEXT_EXTRACTABLE",
        "status": "INDEXED",
    })
    upsert_chunk(conn, {
        "chunk_id": "c1",
        "file_id": "pdf-1",
        "ref_value": "p1",
        "text": "floor plan notes " * 20 + " REV A  2023-11-02",
        "content_hash": "abc",
    })
    conn.commit()

    stats = backfill_title_fields(conn)
    row = conn.execute(
        "SELECT revision, file_date FROM files WHERE file_id='pdf-1'"
    ).fetchone()
    assert stats["revision"] == 1
    assert stats["file_date"] == 1
    assert row["revision"] == 1
    assert row["file_date"] == "2023-11-02"

    again = backfill_title_fields(conn)
    assert again["revision"] == 0
    conn.close()
