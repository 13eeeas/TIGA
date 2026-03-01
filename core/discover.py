"""
core/discover.py — File discovery and classification for the index pipeline.

Two public functions:

  discover(index_roots)
      Dry-run list of candidate files. Used by `tiga.py discover` CLI.
      Returns list[Path]. Does NOT write to DB.

  run_discover(conn, index_roots, cfg_obj)
      Full pipeline step. Walks roots, classifies every file, writes to the
      files table (including SKIPPED and FAILED rows — nothing silently dropped
      except glob-excluded files). Returns stats dict.

Windows long paths:
  Disk operations use the \\\\?\\ prefix when path length > 248 chars.
  DB stores plain resolved POSIX strings (no prefix).
"""

from __future__ import annotations

import fnmatch
import logging
import os
import sqlite3
from pathlib import Path
from typing import Any

from config import cfg as _module_cfg, Config
from core.db import (
    compute_fingerprint,
    file_id_from_path,
    get_file_by_path,
    log_event,
    upsert_file,
)

logger = logging.getLogger(__name__)

_MAX_WIN_PATH = 248
_PROCESSED_STATUSES = frozenset(("EXTRACTED", "EMBEDDED", "INDEXED"))


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def _win_safe(path: Path) -> Path:
    """Apply \\?\\ prefix on Windows for paths exceeding MAX_WIN_PATH.

    UNC paths (\\server\share\...) must use \\?\\UNC\\server\\share\\...
    Local paths use \\?\\C:\\...
    """
    s = str(path.resolve())
    if len(s) > _MAX_WIN_PATH and not s.startswith("\\\\?\\"):
        if s.startswith("\\\\"):  # UNC path: \\server\share\...
            s = "\\\\?\\UNC\\" + s[2:]
        else:
            s = "\\\\?\\" + s
    return Path(s)


def _posix_for_db(path: Path) -> str:
    """Resolved POSIX string without \\?\\ prefix — used as stable DB key."""
    s = str(path.resolve())
    if s.startswith("\\\\?\\"):
        s = s[4:]
    return Path(s).as_posix()


# ---------------------------------------------------------------------------
# Glob filters
# ---------------------------------------------------------------------------

def _is_excluded(path: Path, exclude_globs: list[str]) -> bool:
    name = path.name
    parts = set(path.parts)
    for pattern in exclude_globs:
        segments = [s for s in pattern.split("/") if s not in ("**", "")]
        if not segments:
            continue
        file_pat = segments[-1]
        if fnmatch.fnmatch(name, file_pat):
            return True
        for seg in segments[:-1]:
            if seg in parts:
                return True
    return False


def _is_included(path: Path, include_globs: list[str]) -> bool:
    if not include_globs or include_globs == ["**/*"]:
        return True
    name = path.name
    return any(fnmatch.fnmatch(name, pat.split("/")[-1]) for pat in include_globs)


# ---------------------------------------------------------------------------
# Lane classification
# ---------------------------------------------------------------------------

def _classify_lane(ext: str, cfg_obj: Config) -> str:
    """
    Classify file lane from extension.
    Unknown extension → METADATA_ONLY (safe default — never silently dropped).
    """
    if ext in cfg_obj.text_extractable_exts:
        return "TEXT_EXTRACTABLE"
    if ext in cfg_obj.metadata_only_exts:
        return "METADATA_ONLY"
    return "METADATA_ONLY"   # safe default for unknown ext


# ---------------------------------------------------------------------------
# Pipeline discovery (writes to DB)
# ---------------------------------------------------------------------------

def run_discover(
    conn: sqlite3.Connection,
    index_roots: list[Path],
    cfg_obj: Config | None = None,
) -> dict[str, int]:
    """
    Walk index_roots, classify every file, upsert into the files table.

    All SKIPPED (oversized) and FAILED (unreadable) files appear in the DB.
    Glob-excluded files are the only ones never recorded.

    Incremental: if a file's fingerprint is unchanged and its status is
    EXTRACTED, EMBEDDED, or INDEXED, it is not re-processed.

    Returns:
        {"total": N, "discovered": N, "skipped": N, "failed": N, "unchanged": N}
    """
    _cfg = cfg_obj or _module_cfg
    stats: dict[str, int] = {
        "total":     0,
        "discovered": 0,
        "skipped":   0,
        "failed":    0,
        "unchanged": 0,
    }

    for root in index_roots:
        root = Path(root)
        if not root.exists():
            logger.warning("index_root does not exist, skipping: %s", root)
            continue
        if not root.is_dir():
            logger.warning("index_root is not a directory, skipping: %s", root)
            continue

        logger.info("Scanning: %s", root)
        try:
            _scan_root(root, conn, _cfg, stats)
        except PermissionError as e:
            logger.warning("Permission denied scanning %s: %s", root, e)
        except Exception as e:
            logger.error("Unexpected error scanning %s: %s", root, e)

    logger.info(
        "Discover complete — total: %d, discovered: %d, skipped: %d, "
        "failed: %d, unchanged: %d",
        stats["total"], stats["discovered"], stats["skipped"],
        stats["failed"], stats["unchanged"],
    )
    return stats


def _scan_root(
    root: Path,
    conn: sqlite3.Connection,
    cfg_obj: Config,
    stats: dict[str, int],
) -> None:
    existing_by_path = _load_existing_by_path(conn, root)
    exclude_dir_names = _exclude_dir_names(cfg_obj.exclude_globs)

    for path in _iter_files(root, exclude_dir_names):

        # a/b. Glob filters (excluded files are silently ignored — not in DB)
        safe = _win_safe(path)
        if _is_excluded(safe, cfg_obj.exclude_globs):
            continue
        if not _is_included(safe, cfg_obj.include_globs):
            continue

        stats["total"] += 1

        # c. Resolve POSIX DB key (no \\?\ prefix)
        posix = _posix_for_db(safe)
        fid = file_id_from_path(posix)
        ext = safe.suffix.lower()

        # Get stat (size + mtime)
        try:
            st = safe.stat()
            size_bytes  = st.st_size
            mtime_epoch = st.st_mtime
        except OSError as e:
            logger.warning("Cannot stat %s: %s", safe.name, e)
            upsert_file(conn, {
                "file_id":    fid,
                "file_path":  posix,
                "file_name":  safe.name,
                "extension":  ext,
                "status":     "FAILED",
                "error_code": "STAT_ERROR",
                "error_detail": str(e),
            })
            log_event(conn, "DISCOVER_FAILED", detail=f"STAT_ERROR: {e}", file_id=fid)
            stats["failed"] += 1
            continue

        # d. Oversized → SKIPPED (recorded in DB)
        if size_bytes > cfg_obj.max_file_bytes:
            upsert_file(conn, {
                "file_id":    fid,
                "file_path":  posix,
                "file_name":  safe.name,
                "extension":  ext,
                "size_bytes": size_bytes,
                "mtime_epoch": mtime_epoch,
                "lane":       "SKIPPED",
                "status":     "SKIPPED",
                "error_code": "OVERSIZED",
            })
            log_event(
                conn, "DISCOVER_SKIPPED",
                detail=f"OVERSIZED: {size_bytes} bytes", file_id=fid,
            )
            stats["skipped"] += 1
            continue

        # e. Lane classification (unknown ext → METADATA_ONLY)
        lane = _classify_lane(ext, cfg_obj)

        # f. Fast incremental skip by metadata before expensive hashing
        existing_row = existing_by_path.get(posix)
        if existing_row and _is_fast_unchanged(existing_row, size_bytes, mtime_epoch):
            stats["unchanged"] += 1
            continue

        # g. Fingerprint (IOError → FAILED, recorded in DB)
        try:
            fp = compute_fingerprint(str(safe))
        except OSError as e:
            upsert_file(conn, {
                "file_id":    fid,
                "file_path":  posix,
                "file_name":  safe.name,
                "extension":  ext,
                "size_bytes": size_bytes,
                "mtime_epoch": mtime_epoch,
                "lane":       lane,
                "status":     "FAILED",
                "error_code": "READ_ERROR",
                "error_detail": str(e),
            })
            log_event(conn, "DISCOVER_FAILED", detail=f"READ_ERROR: {e}", file_id=fid)
            stats["failed"] += 1
            continue

        # h. Incremental: skip if fingerprint unchanged and already processed
        existing = existing_row or get_file_by_path(conn, posix)
        if (
            existing
            and existing["fingerprint_sha256"] == fp
            and existing["status"] in _PROCESSED_STATUSES
        ):
            stats["unchanged"] += 1
            continue

        # Upsert as DISCOVERED
        upsert_file(conn, {
            "file_id":            fid,
            "file_path":          posix,
            "file_name":          safe.name,
            "extension":          ext,
            "size_bytes":         size_bytes,
            "mtime_epoch":        mtime_epoch,
            "fingerprint_sha256": fp,
            "lane":               lane,
            "status":             "DISCOVERED",
        })
        stats["discovered"] += 1


def _exclude_dir_names(exclude_globs: list[str]) -> set[str]:
    """Extract directory-name excludes from glob patterns for traversal pruning."""
    names: set[str] = set()
    for pattern in exclude_globs:
        for seg in pattern.split("/"):
            if not seg or seg in ("**", "*"):
                continue
            if any(ch in seg for ch in ("*", "?", "[", "]")):
                continue
            if "." in seg:
                continue
            names.add(seg)
    return names


def _iter_files(root: Path, exclude_dir_names: set[str] | None = None):
    """Yield files under root using os.scandir, pruning excluded directory names."""
    pruned = exclude_dir_names or set()
    stack = [root]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            if entry.name in pruned:
                                continue
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            yield Path(entry.path)
                    except OSError:
                        continue
        except (PermissionError, OSError):
            continue


def _load_existing_by_path(conn: sqlite3.Connection, root: Path) -> dict[str, dict[str, Any]]:
    """Load existing file rows for a root into a lookup map to reduce per-file DB calls."""
    prefix = root.resolve().as_posix().rstrip("/") + "/%"
    rows = conn.execute(
        """
        SELECT file_path, fingerprint_sha256, status, size_bytes, mtime_epoch
        FROM files
        WHERE file_path LIKE ?
        """,
        (prefix,),
    ).fetchall()
    return {r["file_path"]: dict(r) for r in rows}


def _is_fast_unchanged(existing: dict[str, Any], size_bytes: int, mtime_epoch: float) -> bool:
    """Fast unchanged check using metadata before expensive full-file hashing."""
    return (
        existing.get("status") in _PROCESSED_STATUSES
        and existing.get("size_bytes") == size_bytes
        and existing.get("mtime_epoch") == mtime_epoch
    )


# ---------------------------------------------------------------------------
# Dry-run discovery (CLI `tiga.py discover`)
# ---------------------------------------------------------------------------

def discover(index_roots: list[Path] | None = None) -> list[Path]:
    """
    Dry-run: return list of candidate file paths without writing to DB.
    Used by `python tiga.py discover`.
    """
    roots = index_roots or _module_cfg.index_roots
    found: list[Path] = []

    for root in roots:
        root = Path(root)
        if not root.exists() or not root.is_dir():
            logger.warning("index_root not found, skipping: %s", root)
            continue
        logger.info("Scanning (dry run): %s", root)
        try:
            for path in root.rglob("*"):
                if not path.is_file():
                    continue
                safe = _win_safe(path)
                if _is_excluded(safe, _module_cfg.exclude_globs):
                    continue
                if not _is_included(safe, _module_cfg.include_globs):
                    continue
                try:
                    if safe.stat().st_size <= _module_cfg.max_file_bytes:
                        found.append(safe)
                except OSError:
                    continue
        except PermissionError as e:
            logger.warning("Permission denied scanning %s: %s", root, e)

    found.sort()
    logger.info("Discovered %d candidate files", len(found))
    return found
