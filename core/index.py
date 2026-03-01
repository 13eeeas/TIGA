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
import re
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from config import cfg as _module_cfg, Config
from core.db import (
    set_chunk_embedded,
    set_file_status,
    log_event,
)
import core.vectors as _vectors

logger = logging.getLogger(__name__)

# Control chars that cause Ollama HTTP 400: keep tab(9), LF(10), CR(13)
_CTRL_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str) -> str:
    """Replace null bytes and control characters with a space."""
    return _CTRL_RE.sub(' ', text)


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
    Embed all un-embedded chunks of EXTRACTED files using batch API.

    Collects all pending chunks across all EXTRACTED files, sends them in
    batches to Ollama /api/embed (much faster than per-chunk calls), then
    bulk-upserts to LanceDB and updates SQLite status.

    Returns:
        {"files_embedded": N, "chunks_embedded": N, "chunks_skipped": N}
    """
    _cfg = cfg_obj or _module_cfg
    stats: dict[str, int] = {
        "files_embedded": 0,
        "chunks_embedded": 0,
        "chunks_skipped": 0,
    }

    # Collect all pending chunks from all EXTRACTED files in one query
    pending_rows = conn.execute(
        """SELECT c.chunk_id, c.ref_value, c.text,
                  f.file_id, f.file_path, f.extension,
                  COALESCE(f.project_id, 'Unknown') AS project_id,
                  COALESCE(f.typology,   'Unknown') AS typology
           FROM chunks c
           JOIN files  f ON f.file_id = c.file_id
           WHERE f.status = 'EXTRACTED' AND c.embedded = 0"""
    ).fetchall()

    if not pending_rows:
        logger.info("run_embed: no pending chunks to embed")
        # Still advance any EXTRACTED files whose chunks are all already embedded
        for frow in conn.execute(
            "SELECT file_id FROM files WHERE status='EXTRACTED'"
        ).fetchall():
            remaining = conn.execute(
                "SELECT COUNT(*) FROM chunks WHERE file_id=? AND embedded=0",
                (frow["file_id"],),
            ).fetchone()[0]
            if remaining == 0:
                set_file_status(conn, frow["file_id"], "EMBEDDED")
                log_event(conn, "EMBEDDED", file_id=frow["file_id"])
                stats["files_embedded"] += 1
        return stats

    # Separate empty-text chunks — Ollama rejects them (HTTP 400), mark done without vector
    embeddable = [row for row in pending_rows if row["text"] and row["text"].strip()]
    empty_chunk_ids = [row["chunk_id"] for row in pending_rows if not (row["text"] and row["text"].strip())]
    if empty_chunk_ids:
        logger.info("run_embed: skipping %d empty-text chunks (marking embedded=1)", len(empty_chunk_ids))
        conn.executemany(
            "UPDATE chunks SET embedded = 1 WHERE chunk_id = ?",
            [(cid,) for cid in empty_chunk_ids],
        )
        conn.commit()

    logger.info("run_embed: embedding %d pending chunks in batches", len(embeddable))

    # Batch embed all texts (sanitize control chars that cause Ollama HTTP 400)
    texts = [_sanitize(row["text"]) for row in embeddable]
    embeddings = _vectors.embed_texts_batched(texts, _cfg) if texts else []

    # Open LanceDB once for bulk upsert
    import lancedb as _lancedb
    db = _lancedb.connect(str(_cfg.get_vector_dir()))
    table = _vectors._get_chunk_table(db)

    lancedb_rows: list[dict] = []
    succeeded_chunk_ids: list[str] = []
    failed_chunk_ids: list[str] = []

    for row, vec in zip(embeddable, embeddings):
        rel_path = _compute_rel_path(row["file_path"], _cfg)
        if vec is None:
            failed_chunk_ids.append(row["chunk_id"])
            continue
        lancedb_rows.append({
            "chunk_id":   row["chunk_id"],
            "file_id":    row["file_id"],
            "ref_value":  row["ref_value"],
            "rel_path":   rel_path,
            "project_id": row["project_id"],
            "typology":   row["typology"],
            "ext":        row["extension"] or "",
            "root_id":    "",
            "vector":     [float(v) for v in vec],
        })
        succeeded_chunk_ids.append(row["chunk_id"])

    # Bulk upsert to LanceDB in small batches to avoid Arrow spill/OOM errors
    _LANCE_BATCH = 200
    if lancedb_rows:
        for batch_start in range(0, len(lancedb_rows), _LANCE_BATCH):
            batch = lancedb_rows[batch_start : batch_start + _LANCE_BATCH]
            try:
                (
                    table.merge_insert("chunk_id")
                    .when_matched_update_all()
                    .when_not_matched_insert_all()
                    .execute(batch)
                )
                stats["chunks_embedded"] += len(batch)
            except Exception as e:
                logger.error(
                    "LanceDB batch upsert failed (batch %d-%d): %s — skipping batch",
                    batch_start, batch_start + len(batch), e,
                )
                for row_dict in batch:
                    failed_chunk_ids.append(row_dict["chunk_id"])
                    succeeded_chunk_ids.remove(row_dict["chunk_id"])

    stats["chunks_skipped"] += len(failed_chunk_ids)

    # Bulk update SQLite: mark embedded chunks
    if succeeded_chunk_ids:
        conn.executemany(
            "UPDATE chunks SET embedded = 1 WHERE chunk_id = ?",
            [(cid,) for cid in succeeded_chunk_ids],
        )
        conn.commit()

    # Advance file status for files where ALL chunks are now embedded
    file_ids_seen = {row["file_id"] for row in pending_rows}
    for file_id in file_ids_seen:
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


def _run_path_parse(conn: sqlite3.Connection, cfg_obj: Config) -> dict[str, int]:
    """
    Parse structured metadata from file paths for all DISCOVERED files
    that haven't been path-parsed yet (folder_stage IS NULL).
    Updates files table with project_code, folder_stage, discipline, etc.
    """
    stats = {"path_parsed": 0, "path_parse_errors": 0}
    try:
        from core.path_parser import parse_file_path, load_semantics, update_is_latest

        sem_path = cfg_obj.work_dir / "folder_semantics.yml"
        semantics = load_semantics(sem_path)

        # Build root→project_code mapping from project_cards and files
        root_map: dict[str, str] = {}
        try:
            rows = conn.execute(
                "SELECT project_code, root_path FROM project_cards "
                "WHERE root_path IS NOT NULL"
            ).fetchall()
            for r in rows:
                root_map[r["root_path"]] = r["project_code"]
        except Exception:
            pass

        # Process all files missing folder_stage
        pending = conn.execute(
            "SELECT file_id, file_path, project_id FROM files "
            "WHERE folder_stage IS NULL"
        ).fetchall()

        if not pending:
            return stats

        logger.info("_run_path_parse: parsing %d file paths", len(pending))

        for row in pending:
            try:
                file_path = row["file_path"]
                # Determine project root: match against known roots or index_roots
                project_root = ""
                project_code = row["project_id"]  # existing inference result

                for root in cfg_obj.index_roots:
                    root_str = str(root)
                    if file_path.startswith(root_str):
                        project_root = root_str
                        # Override project_code from root_map if available
                        if root_str in root_map:
                            project_code = root_map[root_str]
                        break

                if not project_root:
                    project_root = str(Path(file_path).parent)

                parsed = parse_file_path(file_path, project_root, semantics)
                if project_code and project_code != "Unknown":
                    parsed["project_code"] = project_code

                conn.execute(
                    """UPDATE files SET
                        project_code       = COALESCE(?, project_code),
                        folder_stage       = ?,
                        discipline         = ?,
                        doc_type           = ?,
                        revision           = ?,
                        file_date          = ?,
                        folder_date        = ?,
                        content_type       = ?,
                        is_issued          = ?,
                        is_superseded      = ?,
                        is_received        = ?,
                        canonical_category = ?
                       WHERE file_id = ?""",
                    (
                        parsed.get("project_code"),
                        parsed.get("folder_stage"),
                        parsed.get("discipline"),
                        parsed.get("doc_type"),
                        parsed.get("revision"),
                        parsed.get("file_date"),
                        parsed.get("folder_date"),
                        parsed.get("content_type"),
                        int(parsed.get("is_issued", 0)),
                        int(parsed.get("is_superseded", 0)),
                        int(parsed.get("is_received", 0)),
                        parsed.get("canonical_category"),
                        row["file_id"],
                    ),
                )
                stats["path_parsed"] += 1

                if stats["path_parsed"] % 500 == 0:
                    conn.commit()
                    logger.info("_run_path_parse: %d parsed…", stats["path_parsed"])

            except Exception as e:
                logger.warning("path_parse failed for %s: %s", row.get("file_path"), e)
                stats["path_parse_errors"] += 1

        conn.commit()

        # Update is_latest for all project codes seen
        try:
            update_is_latest(conn)
        except Exception as e:
            logger.warning("update_is_latest failed: %s", e)

        logger.info(
            "_run_path_parse complete — parsed=%d errors=%d",
            stats["path_parsed"], stats["path_parse_errors"],
        )
    except Exception as e:
        logger.warning("_run_path_parse skipped: %s", e)

    return stats


# ---------------------------------------------------------------------------
# Parallel extraction helpers
# ---------------------------------------------------------------------------

def _extract_chunks_worker(args: tuple) -> tuple:
    """
    Worker function for parallel text extraction.

    Runs in a subprocess — does NOT touch the database.  Returns raw chunk
    tuples to the main process which handles all DB writes.

    Args:
        args: (file_id, file_path_str, lane)

    Returns:
        (file_id, file_path_str, lane, chunks, error_str | None)
        where chunks = [(ref, text), ...]
    """
    file_id, file_path_str, lane = args
    try:
        from core.extract import extract_chunks
        from pathlib import Path as _P
        if lane == "TEXT_EXTRACTABLE":
            chunks = extract_chunks(_P(file_path_str))
        else:
            chunks = []  # meta chunk written by main process
        return file_id, file_path_str, lane, chunks, None
    except Exception as exc:
        return file_id, file_path_str, lane, [], str(exc)


def _run_parallel_extract(
    conn: sqlite3.Connection,
    discovered: list,
    extract_workers: int,
    cfg_obj: Config,
) -> dict[str, int]:
    """
    Extract text from DISCOVERED files using a ProcessPoolExecutor.

    Text extraction (PDF/DOCX/PPTX parsing) is CPU-bound and releases Python's
    GIL via native libraries, making it safe to parallelise with processes.
    All DB writes happen in the main process to avoid SQLite locking issues.

    Architecture:
      Worker processes: extract_chunks(path) → [(ref, text), ...]
      Main process:     infer_project/typology + write chunks + update status
    """
    from pathlib import Path as _Path
    from core.extract import run_extract

    stats = {"files_extracted": 0, "files_extract_failed": 0, "chunks_new": 0}
    total = len(discovered)

    if total == 0:
        return stats

    # Separate METADATA_ONLY from TEXT_EXTRACTABLE.
    # METADATA_ONLY files are cheap — handle them sequentially in main process.
    # TEXT_EXTRACTABLE files go to the worker pool for parallel parsing.
    meta_rows = [r for r in discovered if (r["lane"] or "METADATA_ONLY") != "TEXT_EXTRACTABLE"]
    text_rows = [r for r in discovered if (r["lane"] or "METADATA_ONLY") == "TEXT_EXTRACTABLE"]

    # Process METADATA_ONLY sequentially (each is just a surrogate meta chunk)
    for row in meta_rows:
        result = run_extract(
            conn, row["file_id"], _Path(row["file_path"]),
            row["lane"] or "METADATA_ONLY", cfg_obj,
        )
        if result.get("failed", 0) == 0:
            stats["files_extracted"] += 1
        else:
            stats["files_extract_failed"] += 1

    if not text_rows:
        return stats

    logger.info(
        "parallel_extract: %d text files across %d workers",
        len(text_rows), extract_workers,
    )

    # Submit text extraction jobs to worker pool
    args_list = [
        (row["file_id"], row["file_path"], row["lane"] or "TEXT_EXTRACTABLE")
        for row in text_rows
    ]

    completed = 0
    with ProcessPoolExecutor(max_workers=extract_workers) as pool:
        future_map = {pool.submit(_extract_chunks_worker, a): a for a in args_list}
        for future in as_completed(future_map):
            file_id, file_path_str, lane, chunks, err = future.result()
            completed += 1

            if err:
                logger.warning("parallel_extract worker error %s: %s", file_path_str, err)
                # Mark as FAILED via run_extract fallback (it handles error logging)
                run_extract(
                    conn, file_id, _Path(file_path_str), lane, cfg_obj,
                )
                stats["files_extract_failed"] += 1
            else:
                # run_extract does inference + DB write; but text extraction already
                # done — pass pre-extracted chunks via a thin wrapper that skips
                # re-extraction.  For now, call run_extract normally; the
                # pre-extracted chunks are a future optimisation when this path
                # has been validated in production.
                result = run_extract(
                    conn, file_id, _Path(file_path_str), lane, cfg_obj,
                )
                if result.get("failed", 0) == 0:
                    stats["files_extracted"] += 1
                    stats["chunks_new"] += result.get("new", 0)
                else:
                    stats["files_extract_failed"] += 1

            if completed % 100 == 0:
                logger.info(
                    "parallel_extract: %d/%d done  chunks_new=%d",
                    completed, len(text_rows), stats["chunks_new"],
                )

    logger.info(
        "parallel_extract complete: extracted=%d failed=%d chunks_new=%d",
        stats["files_extracted"], stats["files_extract_failed"], stats["chunks_new"],
    )
    return stats


def run_full_pipeline(
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,
) -> dict[str, int]:
    """
    Complete incremental pipeline:
      Phase 0 — Discover new/changed files → DISCOVERED
      Phase 1 — Extract text + chunks      → EXTRACTED (parallel if extract_workers > 1)
      Phase 2 — Embed chunks (Ollama)      → EMBEDDED
      Phase 3 — FTS5 keyword index         → INDEXED

    Idempotent: unchanged files (same fingerprint, already INDEXED) are skipped.
    Returns merged stats from all phases.
    """
    from pathlib import Path as _Path
    from core.discover import run_discover
    from core.extract import run_extract
    from core.scheduler import get_schedule_cfg

    _cfg = cfg_obj or _module_cfg

    # ── Phase 0: discover ────────────────────────────────────────────────────
    logger.info("run_full_pipeline: phase 0 — discover")
    d_stats = run_discover(conn, _cfg.index_roots, _cfg)

    # ── Phase 0b: parse file paths for structured metadata ───────────────────
    logger.info("run_full_pipeline: phase 0b — path parsing")
    _run_path_parse(conn, _cfg)

    # ── Phase 1: extract DISCOVERED files ───────────────────────────────────
    # Read extract_workers from scheduler (time-of-day aware) or config.
    logger.info("run_full_pipeline: phase 1 — extract")
    e_stats = {"files_extracted": 0, "files_extract_failed": 0, "chunks_new": 0}

    discovered = conn.execute(
        "SELECT file_id, file_path, lane FROM files WHERE status = 'DISCOVERED'"
    ).fetchall()

    total_discovered = len(discovered)

    try:
        schedule_cfg = get_schedule_cfg()
        extract_workers = schedule_cfg.extract_workers
    except Exception:
        extract_workers = getattr(_cfg, "extract_workers", 1)

    logger.info(
        "run_full_pipeline: %d files to extract, workers=%d",
        total_discovered, extract_workers,
    )

    if extract_workers > 1 and total_discovered > 0:
        e_stats.update(_run_parallel_extract(conn, discovered, extract_workers, _cfg))
    else:
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

    # ── Phase 4: Image context indexing (renders → synthetic descriptions) ────
    logger.info("run_full_pipeline: phase 4 — image context indexing")
    img_stats = _run_image_indexing(conn, _cfg)

    # ── Phase 5: Content-based classification fallback ───────────────────────
    logger.info("run_full_pipeline: phase 5 — content classification")
    cls_stats = _run_content_classification(conn, _cfg)

    return {**d_stats, **e_stats, **i_stats, **img_stats, **cls_stats}


# ---------------------------------------------------------------------------
# Rebuild
# ---------------------------------------------------------------------------

def _run_image_indexing(conn: sqlite3.Connection, cfg_obj: Config) -> dict[str, int]:
    """
    Chunk 9: For image files (METADATA_ONLY with image extensions) that haven't
    been given a synthetic description yet, generate one from the path context.
    Updates image_type and inserts a synthetic text chunk for semantic search.
    """
    stats = {"images_indexed": 0, "images_skipped": 0}
    try:
        from core.extract import classify_image, build_image_synthetic_description
        import hashlib as _hash

        image_exts = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".bmp"}
        # Find image files that are EXTRACTED (have meta chunk) but no image_type set
        rows = conn.execute(
            "SELECT f.file_id, f.file_path, f.extension "
            "FROM files f "
            "WHERE f.image_type IS NULL "
            "AND f.lane = 'METADATA_ONLY' "
            "AND LOWER(f.extension) IN (" +
            ",".join("?" * len(image_exts)) + ")",
            list(image_exts),
        ).fetchall()

        if not rows:
            return stats

        logger.info("_run_image_indexing: processing %d image files", len(rows))

        for row in rows:
            try:
                path = Path(row["file_path"])
                img_cls = classify_image(path)

                # Update image_type in files table
                conn.execute(
                    "UPDATE files SET image_type = ? WHERE file_id = ?",
                    (img_cls.image_type, row["file_id"]),
                )

                # For renders: generate synthetic description as chunk text
                if not img_cls.needs_ocr:
                    desc = build_image_synthetic_description(path, img_cls)
                    if desc:
                        # Upsert a synthetic text chunk for this image
                        from core.db import upsert_chunk as _upsert_chunk
                        chunk_id = _hash.sha256(
                            f"{row['file_id']}::image_context".encode()
                        ).hexdigest()
                        c_hash = _hash.sha256(desc.encode()).hexdigest()
                        existing = conn.execute(
                            "SELECT content_hash FROM chunks WHERE chunk_id=?",
                            (chunk_id,),
                        ).fetchone()
                        if not existing:
                            _upsert_chunk(conn, {
                                "chunk_id": chunk_id,
                                "file_id": row["file_id"],
                                "ref_value": "image_context",
                                "text": desc,
                                "token_estimate": len(desc.split()),
                                "content_hash": c_hash,
                            })
                            stats["images_indexed"] += 1
                        else:
                            stats["images_skipped"] += 1

            except Exception as e:
                logger.debug("image indexing failed for %s: %s", row.get("file_path"), e)
                stats["images_skipped"] += 1

        conn.commit()
        logger.info("_run_image_indexing: indexed=%d skipped=%d",
                    stats["images_indexed"], stats["images_skipped"])

    except Exception as e:
        logger.warning("_run_image_indexing skipped: %s", e)

    return stats


def _run_content_classification(conn: sqlite3.Connection, cfg_obj: Config) -> dict[str, int]:
    """
    Chunk 12: Content-based document classification fallback.
    For TEXT_EXTRACTABLE files where doc_type is NULL or 'unknown',
    read first 500 words from DB chunks and classify.
    Only applies classification if confidence > 0.75.
    """
    stats = {"content_classified": 0, "content_classified_low_conf": 0}
    try:
        from core.extract import classify_document_by_content

        # Find indexed files with unknown doc_type
        rows = conn.execute(
            """SELECT f.file_id, f.file_path,
                      (SELECT c.text FROM chunks c WHERE c.file_id = f.file_id
                       ORDER BY c.rowid LIMIT 1) AS first_chunk_text
               FROM files f
               WHERE f.lane = 'TEXT_EXTRACTABLE'
                 AND (f.doc_type IS NULL OR f.doc_type = 'unknown')
                 AND f.status = 'INDEXED'
               LIMIT 5000"""
        ).fetchall()

        if not rows:
            return stats

        logger.info("_run_content_classification: checking %d files", len(rows))

        for row in rows:
            text = row["first_chunk_text"] or ""
            if not text.strip():
                continue

            try:
                cls = classify_document_by_content(text, row["file_path"])
                method = "content"
                if cls.confidence >= 0.75:
                    conn.execute(
                        "UPDATE files SET doc_type = ?, classification_method = ?, "
                        "classification_confidence = ? WHERE file_id = ?",
                        (cls.doc_type, method, cls.confidence, row["file_id"]),
                    )
                    stats["content_classified"] += 1
                else:
                    conn.execute(
                        "UPDATE files SET classification_method = 'unknown', "
                        "classification_confidence = ? WHERE file_id = ?",
                        (cls.confidence, row["file_id"]),
                    )
                    stats["content_classified_low_conf"] += 1
            except Exception as e:
                logger.debug("content classification failed for %s: %s", row["file_path"], e)

        conn.commit()
        logger.info(
            "_run_content_classification: classified=%d low_conf=%d",
            stats["content_classified"], stats["content_classified_low_conf"],
        )

    except Exception as e:
        logger.warning("_run_content_classification skipped: %s", e)

    return stats


# ---------------------------------------------------------------------------
# Chunk 13 — Index health / staleness detection
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc


@_dc
class StalenessReport:
    project_code:           str | None
    last_full_index:        str          # ISO datetime or empty
    last_incremental_index: str          # ISO datetime or empty
    files_indexed:          int
    files_missing:          int          # indexed files no longer on disk
    files_new_estimate:     int          # rough estimate only
    staleness_score:        float        # 0.0 = fresh, 1.0 = very stale
    recommendation:         str          # "fresh" | "run incremental" | "run rebuild"


def check_index_staleness(
    conn: sqlite3.Connection,
    project_code: str | None = None,
    cfg_obj: Config | None = None,
    sample_limit: int = 500,
) -> StalenessReport:
    """
    Lightweight staleness check.
    Samples up to `sample_limit` indexed file paths to detect missing files.
    Does NOT scan the full archive (too slow for real-time use).
    """
    from datetime import datetime, timezone, timedelta

    _cfg = cfg_obj or _module_cfg

    try:
        # Last indexed timestamp
        if project_code:
            row = conn.execute(
                "SELECT MAX(e.ts) AS last_ts FROM events e "
                "JOIN files f ON f.file_id = e.file_id "
                "WHERE e.event_type = 'INDEXED' AND f.project_code = ?",
                (project_code,),
            ).fetchone()
        else:
            row = conn.execute(
                "SELECT MAX(ts) AS last_ts FROM events WHERE event_type = 'INDEXED'"
            ).fetchone()
        last_index = row["last_ts"] if row and row["last_ts"] else ""

        # Count indexed files
        if project_code:
            count_row = conn.execute(
                "SELECT COUNT(*) AS n FROM files WHERE status = 'INDEXED' AND project_code = ?",
                (project_code,),
            ).fetchone()
        else:
            count_row = conn.execute(
                "SELECT COUNT(*) AS n FROM files WHERE status = 'INDEXED'"
            ).fetchone()
        files_indexed = count_row["n"] if count_row else 0

        # Sample files and check if they still exist
        if project_code:
            sample_rows = conn.execute(
                "SELECT file_path FROM files WHERE status = 'INDEXED' "
                "AND project_code = ? LIMIT ?",
                (project_code, sample_limit),
            ).fetchall()
        else:
            sample_rows = conn.execute(
                "SELECT file_path FROM files WHERE status = 'INDEXED' "
                "ORDER BY RANDOM() LIMIT ?",
                (sample_limit,),
            ).fetchall()

        files_missing_sample = sum(
            1 for r in sample_rows if not Path(r["file_path"]).exists()
        )
        # Scale estimate to full population
        sample_size = len(sample_rows)
        if sample_size > 0:
            missing_rate = files_missing_sample / sample_size
            files_missing = int(missing_rate * files_indexed)
        else:
            files_missing = 0

        # Staleness score
        score = 0.0
        if last_index:
            try:
                last_dt = datetime.fromisoformat(last_index.replace("Z", "+00:00"))
                age_days = (datetime.now(timezone.utc) - last_dt.replace(
                    tzinfo=last_dt.tzinfo or timezone.utc
                )).days
                if age_days > 7:
                    score += 0.2
                if age_days > 30:
                    score += 0.3
            except Exception:
                score += 0.1
        else:
            score += 0.5  # never indexed

        missing_rate_total = files_missing / max(files_indexed, 1)
        if missing_rate_total > 0.1:
            score += 0.2
        if missing_rate_total > 0.02:
            score += 0.1

        score = min(score, 1.0)

        if score < 0.2:
            recommendation = "fresh"
        elif score < 0.5:
            recommendation = "run incremental"
        else:
            recommendation = "run rebuild"

        return StalenessReport(
            project_code=project_code,
            last_full_index=last_index,
            last_incremental_index=last_index,
            files_indexed=files_indexed,
            files_missing=files_missing,
            files_new_estimate=0,   # only computable by full disk scan
            staleness_score=round(score, 3),
            recommendation=recommendation,
        )

    except Exception as e:
        logger.warning("check_index_staleness failed: %s", e)
        return StalenessReport(
            project_code=project_code,
            last_full_index="",
            last_incremental_index="",
            files_indexed=0,
            files_missing=0,
            files_new_estimate=0,
            staleness_score=0.5,
            recommendation="run incremental",
        )


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
