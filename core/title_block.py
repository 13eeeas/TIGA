"""
Title-block fields from text already extracted from PDF page 1.

Does not open CAD, render folders, or re-embed. Letter revisions map
A→1 … Z→26 so they fit the integer revision column.
"""

from __future__ import annotations

import logging
import re
import sqlite3

from core.path_parser import _extract_date

logger = logging.getLogger(__name__)

_REV_RE = re.compile(
    r"\brev(?:ision)?\.?\s*[:\-]?\s*([A-Z]|\d{1,2})\b",
    re.IGNORECASE,
)


def revision_token_to_int(token: str) -> int | None:
    raw = token.strip()
    if raw.isdigit():
        value = int(raw)
        return value if 0 <= value <= 99 else None
    if len(raw) == 1 and raw.isalpha():
        return ord(raw.upper()) - ord("A") + 1
    return None


def parse_title_block(text: str) -> dict[str, int | str | None]:
    """Read revision and date from the start and end of page text.

    PDF extractors usually emit the title block last, so the tail is preferred.
    """
    found: dict[str, int | str | None] = {"revision": None, "file_date": None}
    if not text or not text.strip():
        return found

    windows = [text[-2000:], text[:800]]
    for window in windows:
        if found["revision"] is None:
            match = _REV_RE.search(window)
            if match:
                found["revision"] = revision_token_to_int(match.group(1))
        if found["file_date"] is None:
            found["file_date"] = _extract_date(window)
        if found["revision"] is not None and found["file_date"] is not None:
            break
    return found


def backfill_title_fields(conn: sqlite3.Connection) -> dict[str, int]:
    """Fill empty revision / file_date from stored PDF page-1 chunk text."""
    stats = {"seen": 0, "revision": 0, "file_date": 0}
    rows = conn.execute(
        """
        SELECT f.file_id, f.revision, f.file_date, c.text
        FROM files f
        JOIN chunks c ON c.file_id = f.file_id
        WHERE LOWER(f.extension) = '.pdf'
          AND c.ref_value LIKE 'p1%'
          AND LENGTH(c.text) > 80
          AND (f.revision IS NULL OR f.file_date IS NULL OR f.file_date = '')
        """
    )
    updates: list[tuple] = []
    for row in rows:
        stats["seen"] += 1
        parsed = parse_title_block(row["text"] or "")
        revision = row["revision"]
        file_date = row["file_date"]
        changed_rev = False
        changed_date = False
        if revision is None and parsed["revision"] is not None:
            revision = parsed["revision"]
            changed_rev = True
        if (not file_date) and parsed["file_date"]:
            file_date = parsed["file_date"]
            changed_date = True
        if changed_rev or changed_date:
            updates.append((revision, file_date, row["file_id"]))
            stats["revision"] += int(changed_rev)
            stats["file_date"] += int(changed_date)

    if updates:
        conn.executemany(
            "UPDATE files SET revision=?, file_date=? WHERE file_id=?",
            updates,
        )
        conn.commit()
    logger.info("title-block backfill: %s", stats)
    return stats
