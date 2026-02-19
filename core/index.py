"""
core/index.py — Full indexing pipeline orchestrator.

Flow: discover → extract → infer → db upsert → vector upsert
Skips files unchanged since last index (mtime check).
"""

from __future__ import annotations

import datetime
import logging
import sqlite3
from pathlib import Path
from typing import Callable

from tqdm import tqdm

from config import cfg
from core.db import get_connection, upsert_document, get_by_path
from core.discover import discover
from core.extract import extract, word_count
from core.infer import infer_project, infer_typology, infer_title
from core.vectors import get_collection, upsert_documents

logger = logging.getLogger(__name__)


def _mtime_str(path: Path) -> str:
    ts = path.stat().st_mtime
    return datetime.datetime.utcfromtimestamp(ts).isoformat()


def _needs_reindex(conn: sqlite3.Connection, path: Path, mtime: str) -> bool:
    row = get_by_path(conn, path.as_posix())
    if row is None:
        return True
    return row["modified_at"] != mtime


def run_index(
    index_roots: list[Path] | None = None,
    force: bool = False,
    progress_cb: Callable[[str], None] | None = None,
) -> dict[str, int]:
    """
    Index all supported files.

    Args:
        index_roots: Override cfg.index_roots.
        force:       Re-index even if file is unchanged.
        progress_cb: Optional callback for live progress messages.

    Returns:
        Stats dict: indexed, skipped, failed.
    """
    cfg.ensure_dirs()
    conn = get_connection(cfg.get_db_path())
    collection = get_collection()

    files = discover(index_roots)
    if not files:
        logger.warning("No files discovered. Check index_roots in config.yaml.")
        return {"indexed": 0, "skipped": 0, "failed": 0}

    stats = {"indexed": 0, "skipped": 0, "failed": 0}
    vector_batch: list[dict] = []

    for path in tqdm(files, desc="Indexing", unit="file"):
        try:
            mtime = _mtime_str(path)

            if not force and not _needs_reindex(conn, path, mtime):
                stats["skipped"] += 1
                continue

            if progress_cb:
                progress_cb(f"Indexing: {path.name}")

            text, surrogate = extract(path)
            project  = infer_project(path)
            typology = infer_typology(path)
            title    = infer_title(path)
            wc       = word_count(text)

            doc = {
                "file_path":    path.as_posix(),
                "file_name":    path.name,
                "extension":    path.suffix.lower(),
                "project":      project,
                "typology":     typology,
                "title":        title,
                "surrogate":    surrogate,
                "text_content": text,
                "word_count":   wc,
                "file_size":    path.stat().st_size,
                "modified_at":  mtime,
            }

            upsert_document(conn, doc)
            vector_batch.append(doc)
            stats["indexed"] += 1

            if len(vector_batch) >= cfg.batch_size:
                upsert_documents(vector_batch, collection)
                vector_batch.clear()

        except Exception as e:
            logger.error("Failed to index %s: %s", path.name, e)
            stats["failed"] += 1

    if vector_batch:
        upsert_documents(vector_batch, collection)

    conn.close()
    logger.info(
        "Index complete — indexed: %d, skipped: %d, failed: %d",
        stats["indexed"], stats["skipped"], stats["failed"],
    )
    return stats
