"""
core/discover.py — File discovery across index_roots with glob filtering.

Returns Path objects for every file that:
  - Lives under an index_root
  - Matches at least one include_glob
  - Does not match any exclude_glob
  - Does not exceed max_file_bytes
  - Has an extension in text_extractable_exts OR metadata_only_exts

Windows long paths are handled automatically.
"""

from __future__ import annotations

import fnmatch
import logging
from pathlib import Path

from config import cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Glob helpers
# ---------------------------------------------------------------------------

def _is_excluded(path: Path, exclude_globs: list[str]) -> bool:
    """
    Return True if path matches any exclude glob.
    Checks filename and all path components for directory-level patterns.
    """
    name = path.name
    parts = set(path.parts)

    for pattern in exclude_globs:
        segments = [s for s in pattern.split("/") if s not in ("**", "")]
        if not segments:
            continue
        # Last segment → filename pattern (e.g. '~$*', '*.tmp')
        file_pat = segments[-1]
        if fnmatch.fnmatch(name, file_pat):
            return True
        # All segments → check as directory component names
        for seg in segments[:-1]:
            if seg in parts:
                return True

    return False


def _is_included(path: Path, include_globs: list[str]) -> bool:
    """Return True if filename matches at least one include glob."""
    if not include_globs or include_globs == ["**/*"]:
        return True
    name = path.name
    return any(fnmatch.fnmatch(name, pat.split("/")[-1]) for pat in include_globs)


def _is_processable(path: Path) -> bool:
    """Return True if extension is in either extract or metadata lane."""
    ext = path.suffix.lower()
    return ext in cfg.text_extractable_exts or ext in cfg.metadata_only_exts


# ---------------------------------------------------------------------------
# Windows long-path safety
# ---------------------------------------------------------------------------

_MAX_WIN_PATH = 248


def _win_safe(path: Path) -> Path:
    s = str(path.resolve())
    if len(s) > _MAX_WIN_PATH and not s.startswith("\\\\?\\"):
        s = "\\\\?\\" + s
    return Path(s)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def discover(index_roots: list[Path] | None = None) -> list[Path]:
    """
    Walk all index_roots and return candidate files after all filters.

    Args:
        index_roots: Override cfg.index_roots.

    Returns:
        Sorted list of absolute Path objects.
    """
    roots = index_roots or cfg.index_roots
    found: list[Path] = []

    for root in roots:
        root = Path(root)
        if not root.exists():
            logger.warning("index_root not found, skipping: %s", root)
            continue
        if not root.is_dir():
            logger.warning("index_root is not a directory, skipping: %s", root)
            continue

        logger.info("Scanning: %s", root)
        _scan_root(root, found)

    found.sort()
    logger.info("Discovered %d processable files", len(found))
    return found


def _scan_root(root: Path, found: list[Path]) -> None:
    try:
        for path in root.rglob("*"):
            if not path.is_file():
                continue

            safe = _win_safe(path)

            if _is_excluded(safe, cfg.exclude_globs):
                continue
            if not _is_included(safe, cfg.include_globs):
                continue
            if not _is_processable(safe):
                continue

            try:
                size = safe.stat().st_size
                if size > cfg.max_file_bytes:
                    logger.debug("Skipping oversized file (%d MB): %s", size // (1024**2), safe.name)
                    continue
            except OSError:
                continue

            found.append(safe)

    except PermissionError as e:
        logger.warning("Permission denied scanning %s: %s", root, e)
