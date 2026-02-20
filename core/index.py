"""
core/index.py — Two-step indexing pipeline.

Step 1 — run_embed: Vector embedding (Ollama → LanceDB)
  Query: files WHERE status='EXTRACTED'
         chunks WHERE file_id=<id> AND embedded=0
  For each chunk:
    vectors.upsert_chunk() → True/False
    On success  → set chunk.embedded=1
    On failure  → log warning, skip chunk, keep embedded=0
                  do NOT fail the file
  After all chunks for a file have embedded=0 count = 0:
    UPDATE files SET status='EMBEDDED'

Step 2 — run_fts: FTS5 keyword index (SQLite)
  Query: files WHERE status='EMBEDDED'
  FTS is auto-populated by triggers when chunks are upserted during extract.
  This step ensures sync and advances file status:
    UPDATE files SET status='INDEXED'

run_index(conn, cfg_obj): Step 1 then Step 2
run_rebuild(conn, cfg_obj):
  1. Drop and recreate chunks_fts + triggers
  2. Rebuild FTS from chunks content table
  3. Reset chunks.embedded = 0
  4. Reset files INDEXED|EMBEDDED → EXTRACTED
  5. Drop LanceDB tiga_chunks table
  6. Re-run run_index

IDEMPOTENCY
  Embed: WHERE embedded=0 naturally skips already-embedded chunks.
  FTS:   Content table + triggers ensure no duplicates on normal upsert.
         Rebuild uses FTS5 'rebuild' command to re-sync cleanly.
  Status: transitions only go forward (EXTRACTED→EMBEDDED→INDEXED),
          except during rebuild which explicitly resets backward.

STATUS FLOW
  DISCOVERED → EXTRACTED → EMBEDDED → INDEXED
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

from config import cfg as _module_cfg, Config
from core.db import (
    set_chunk_embedded,
    set_file_status,
    log_event,
)
import core.vectors as _vectors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compute_rel_path(file_path: str, cfg: Config) -> str:
    """Return path relative to the first matching index_root, else basename."""
    p = Path(file_path)
    for root in cfg.index_roots:
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            continue
    return p.name


def _rebuild_fts(conn: sqlite3.Connection) -> None:
    """
    Drop and recreate chunks_fts virtual table + triggers,
    then repopulate from the chunks content table.
    """
    # executescript issues implicit COMMIT before DDL — safe here
    conn.executescript("""
        DROP TRIGGER IF EXISTS chunks_fts_insert;
        DROP TRIGGER IF EXISTS chunks_fts_update;
        DROP TRIGGER IF EXISTS chunks_fts_delete;
        DROP TABLE IF EXISTS chunks_fts;

        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            text,
            chunk_id UNINDEXED,
            content='chunks',
            content_rowid='rowid',
            tokenize='porter ascii'
        );

        CREATE TRIGGER IF NOT EXISTS chunks_fts_insert
        AFTER INSERT ON chunks BEGIN
            INSERT INTO chunks_fts(rowid, text, chunk_id)
            VALUES (new.rowid, new.text, new.chunk_id);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_fts_update
        AFTER UPDATE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, chunk_id)
            VALUES ('delete', old.rowid, old.text, old.chunk_id);
            INSERT INTO chunks_fts(rowid, text, chunk_id)
            VALUES (new.rowid, new.text, new.chunk_id);
        END;

        CREATE TRIGGER IF NOT EXISTS chunks_fts_delete
        AFTER DELETE ON chunks BEGIN
            INSERT INTO chunks_fts(chunks_fts, rowid, text, chunk_id)
            VALUES ('delete', old.rowid, old.text, old.chunk_id);
        END;
    """)
    # Repopulate FTS from the content table (chunks)
    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
    conn.commit()


def _reset_lancedb(cfg_obj: Config | None = None) -> None:
    """Drop the tiga_chunks LanceDB table so it is recreated fresh on next embed."""
    try:
        import lancedb  # type: ignore
        _cfg = cfg_obj or _module_cfg
        db = lancedb.connect(str(_cfg.get_vector_dir()))
        if "tiga_chunks" in db.table_names():
            db.drop_table("tiga_chunks")
            logger.info("LanceDB tiga_chunks table dropped for rebuild")
    except Exception as e:
        logger.warning("Failed to reset LanceDB (non-fatal): %s", e)


# ---------------------------------------------------------------------------
# Step 1: Vector embedding
# ---------------------------------------------------------------------------

def run_embed(
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,
) -> dict[str, int]:
    """
    Embed all un-embedded chunks of EXTRACTED files.

    Returns:
        {"files_embedded": N, "chunks_embedded": N, "chunks_skipped": N}
    """
    _cfg = cfg_obj or _module_cfg
    stats: dict[str, int] = {
        "files_embedded": 0,
        "chunks_embedded": 0,
        "chunks_skipped": 0,
    }

    extracted = conn.execute(
        """SELECT file_id, file_path, extension,
                  COALESCE(project_id, 'Unknown') AS project_id,
                  COALESCE(typology, 'Unknown')   AS typology
           FROM files WHERE status = 'EXTRACTED'"""
    ).fetchall()

    for file_row in extracted:
        file_id = file_row["file_id"]

        pending = conn.execute(
            "SELECT chunk_id, ref_value, text FROM chunks "
            "WHERE file_id = ? AND embedded = 0",
            (file_id,),
        ).fetchall()

        rel_path = _compute_rel_path(file_row["file_path"], _cfg)

        for chunk in pending:
            success = _vectors.upsert_chunk(
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                metadata={
                    "file_id":    file_id,
                    "ref_value":  chunk["ref_value"],
                    "rel_path":   rel_path,
                    "project_id": file_row["project_id"],
                    "typology":   file_row["typology"],
                    "ext":        file_row["extension"] or "",
                    "root_id":    "",
                },
                cfg=_cfg,
            )
            if success:
                set_chunk_embedded(conn, chunk["chunk_id"], 1)
                stats["chunks_embedded"] += 1
            else:
                logger.warning(
                    "Chunk embed failed — skipping chunk %s (file %s)",
                    chunk["chunk_id"][:12], file_id[:12],
                )
                stats["chunks_skipped"] += 1

        # Advance file only when ALL chunks are embedded
        remaining = conn.execute(
            "SELECT COUNT(*) FROM chunks WHERE file_id = ? AND embedded = 0",
            (file_id,),
        ).fetchone()[0]

        if remaining == 0:
            set_file_status(conn, file_id, "EMBEDDED")
            log_event(conn, "EMBEDDED", file_id=file_id)
            stats["files_embedded"] += 1

    logger.info(
        "run_embed complete — files_embedded=%d chunks_embedded=%d chunks_skipped=%d",
        stats["files_embedded"], stats["chunks_embedded"], stats["chunks_skipped"],
    )
    return stats


# ---------------------------------------------------------------------------
# Step 2: FTS5 keyword index
# ---------------------------------------------------------------------------

def run_fts(
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,  # kept for API consistency
) -> dict[str, int]:
    """
    Advance EMBEDDED files to INDEXED.

    FTS entries are auto-populated by triggers when chunks are upserted
    during run_extract. This step finalises file status and (in rebuild
    scenarios) ensures FTS is in sync via the content table.

    Returns:
        {"files_indexed": N}
    """
    stats: dict[str, int] = {"files_indexed": 0}

    embedded_files = conn.execute(
        "SELECT file_id FROM files WHERE status = 'EMBEDDED'"
    ).fetchall()

    for row in embedded_files:
        file_id = row["file_id"]
        set_file_status(conn, file_id, "INDEXED")
        log_event(conn, "INDEXED", file_id=file_id)
        stats["files_indexed"] += 1

    logger.info("run_fts complete — files_indexed=%d", stats["files_indexed"])
    return stats


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def run_index(
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,
) -> dict[str, int]:
    """
    Step 1 (embed) then Step 2 (FTS). Returns merged stats.
    Expects files already in EXTRACTED status (called by run_full_pipeline).
    """
    s1 = run_embed(conn, cfg_obj)
    s2 = run_fts(conn, cfg_obj)
    return {**s1, **s2}


def run_full_pipeline(
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,
) -> dict[str, int]:
    """
    Complete incremental pipeline:
      Phase 0 — Discover new/changed files → DISCOVERED
      Phase 1 — Extract text + chunks      → EXTRACTED
      Phase 2 — Embed chunks (Ollama)      → EMBEDDED
      Phase 3 — FTS5 keyword index         → INDEXED

    Idempotent: unchanged files (same fingerprint, already INDEXED) are skipped.
    Returns merged stats from all phases.
    """
    from pathlib import Path as _Path
    from core.discover import run_discover
    from core.extract import run_extract

    _cfg = cfg_obj or _module_cfg

    # ── Phase 0: discover ────────────────────────────────────────────────────
    logger.info("run_full_pipeline: phase 0 — discover")
    d_stats = run_discover(conn, _cfg.index_roots, _cfg)

    # ── Phase 1: extract DISCOVERED files ───────────────────────────────────
    logger.info("run_full_pipeline: phase 1 — extract")
    e_stats = {"files_extracted": 0, "files_extract_failed": 0, "chunks_new": 0}

    discovered = conn.execute(
        "SELECT file_id, file_path, lane FROM files WHERE status = 'DISCOVERED'"
    ).fetchall()

    total_discovered = len(discovered)
    for idx, row in enumerate(discovered, start=1):
        path = _Path(row["file_path"])
        lane = row["lane"] or "METADATA_ONLY"
        result = run_extract(conn, row["file_id"], path, lane, _cfg)
        if result.get("failed", 0) == 0:
            e_stats["files_extracted"] += 1
            e_stats["chunks_new"] += result.get("new", 0)
        else:
            e_stats["files_extract_failed"] += 1
        if idx % 500 == 0:
            logger.info(
                "run_full_pipeline: extract progress %d/%d  chunks_new=%d",
                idx, total_discovered, e_stats["chunks_new"],
            )

    # ── Phases 2 + 3: embed + FTS ────────────────────────────────────────────
    logger.info("run_full_pipeline: phases 2+3 — embed + FTS")
    i_stats = run_index(conn, _cfg)

    return {**d_stats, **e_stats, **i_stats}


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------

def run_rebuild(
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,
) -> dict[str, int]:
    """
    Full rebuild:
      1. Drop + recreate chunks_fts and triggers
      2. Rebuild FTS from chunks content table
      3. Reset chunks.embedded = 0
      4. Reset INDEXED|EMBEDDED files → EXTRACTED
      5. Drop LanceDB tiga_chunks table
      6. Re-run run_index

    Returns merged index stats.
    """
    logger.info("run_rebuild: dropping and rebuilding FTS index")
    _rebuild_fts(conn)

    logger.info("run_rebuild: resetting chunk embedded flags")
    conn.execute("UPDATE chunks SET embedded = 0")
    conn.commit()

    logger.info("run_rebuild: resetting file statuses to EXTRACTED")
    conn.execute(
        "UPDATE files SET status = 'EXTRACTED' "
        "WHERE status IN ('INDEXED', 'EMBEDDED')"
    )
    conn.commit()

    logger.info("run_rebuild: resetting LanceDB vector table")
    _reset_lancedb(cfg_obj)

    logger.info("run_rebuild: re-running index pipeline")
    return run_index(conn, cfg_obj)
