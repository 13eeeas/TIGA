"""
core/storage.py — TIGA per-project operating storage measurement.

Measures how much disk TIGA itself uses (work_dir / index), distinct from the
original NAS archive. Shared SQLite and vector-store totals are measured
exactly; per-project shares of those stores are estimated. Logical extracted
text length is reported separately and is never treated as on-disk size.

Public API
----------
  measure_operating_storage(conn, work_dir, ...) -> dict
  get_storage_report(conn, work_dir, refresh=False, ...) -> dict
  format_storage_report(snapshot) -> str
  format_bytes(n) -> str
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

# Snapshot cache — avoid walking work_dir on every page load.
DEFAULT_CACHE_TTL_SECONDS = 15 * 60
SNAPSHOT_SCHEMA_VERSION = 1
UNKNOWN_PROJECT = "unknown"

# Persistent operating artifacts vs temp / backup (issue #8).
_PERSISTENT_CATEGORIES = frozenset({
    "sqlite_index",
    "vector_store",
    "atlas",
    "other",
})
_TEMPORARY_CATEGORIES = frozenset({
    "logs",
    "caches",
    "temporary",
    "backups",
})

# Shared stores that are split across projects by weight.
_SHARED_STORE_CATEGORIES = frozenset({"sqlite_index", "vector_store"})

# Cost stubs — planning estimates, not invoices. Storage GB is not API billing.
_SGD_PER_USD = 1.35
_EMBED_USD_PER_MILLION_TOKENS = 0.02
_REFRESH_FRACTION = 0.15
_PER_QUERY_SGD = 0.003
_CHARS_PER_TOKEN = 4.0

COST_DISCLAIMER = (
    "Cost fields are estimated stubs for planning, not invoices. "
    "Storage GB is not API billing. Local extract/embed is treated as S$0 "
    "hardware time; figures assume a cheap embed/compose tariff if APIs are used."
)


# ---------------------------------------------------------------------------
# Units / formatting
# ---------------------------------------------------------------------------

def format_bytes(n: int | float | None) -> str:
    """Human-readable size with an explicit unit (B / KiB / MiB / GiB)."""
    if n is None:
        return "—"
    value = float(n)
    if value < 0:
        sign = "-"
        value = abs(value)
    else:
        sign = ""
    if value < 1024:
        return f"{sign}{int(value)} B"
    if value < 1024 ** 2:
        return f"{sign}{value / 1024:.1f} KiB"
    if value < 1024 ** 3:
        return f"{sign}{value / (1024 ** 2):.2f} MiB"
    return f"{sign}{value / (1024 ** 3):.3f} GiB"


def bytes_to_gb(n: int | float | None) -> float | None:
    """GiB using 1024³. Returned as a float for JSON dashboards."""
    if n is None:
        return None
    return round(float(n) / (1024 ** 3), 6)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def snapshot_path(work_dir: Path) -> Path:
    return Path(work_dir) / "reports" / "storage_snapshot.json"


# ---------------------------------------------------------------------------
# Project identity
# ---------------------------------------------------------------------------

def _normalize_project_id(raw: Any) -> str:
    """Map empty / Unknown / None to the unknown bucket."""
    if raw is None:
        return UNKNOWN_PROJECT
    text = str(raw).strip()
    if not text or text.lower() in {"unknown", "none", "null", "n/a", "-"}:
        return UNKNOWN_PROJECT
    return text


def _slug(code: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", code.lower()).strip("-") or UNKNOWN_PROJECT


# ---------------------------------------------------------------------------
# Work-dir walk (TIGA operating files only — never the NAS archive)
# ---------------------------------------------------------------------------

def classify_work_path(rel: Path) -> str:
    """
    Assign a work_dir-relative path to one storage category.
    Each file is classified once to avoid double-counting.
    """
    parts = rel.parts
    name = rel.name.lower()
    if name.endswith((".bak", ".old", ".orig")) or ".bak." in name:
        return "backups"
    if not parts:
        return "other"
    top = parts[0].lower()
    if top in {"backup", "backups", "_backup"}:
        return "backups"
    if top == "db":
        return "sqlite_index"
    if top in {"vectors", "vector"}:
        return "vector_store"
    if top == "atlas":
        return "atlas"
    if top == "logs":
        return "logs"
    if top in {"tmp", "temp", ".tmp"}:
        return "temporary"
    if top in {"cache", "caches"}:
        return "caches"
    if top in {"field_data", "field_exports", "field_imports", "poc_test", "reports"}:
        return "caches"
    return "other"


def walk_work_dir(work_dir: Path) -> list[tuple[Path, int, str]]:
    """Return (absolute path, size, category) for every regular file under work_dir."""
    root = Path(work_dir)
    found: list[tuple[Path, int, str]] = []
    if not root.exists():
        return found
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames[:] = [d for d in dirnames if d not in {"__pycache__", ".git"}]
        for name in filenames:
            path = Path(dirpath) / name
            try:
                if not path.is_file() or path.is_symlink():
                    continue
                size = path.stat().st_size
            except OSError:
                continue
            try:
                rel = path.relative_to(root)
            except ValueError:
                continue
            found.append((path, int(size), classify_work_path(rel)))
    return found


def _atlas_project_slug(path: Path) -> str | None:
    name = path.name
    lower = name.lower()
    for suffix in (".overlay.json", ".json", ".yml", ".yaml"):
        if lower.endswith(suffix):
            stem = name[: -len(suffix)]
            slug = _slug(stem)
            return None if slug == UNKNOWN_PROJECT else slug
    return None


# ---------------------------------------------------------------------------
# Catalog stats (source archive size from the index — no live NAS walk)
# ---------------------------------------------------------------------------

_PROJECT_SQL = """
SELECT
    COALESCE(NULLIF(TRIM(project_code), ''), NULLIF(TRIM(project_id), ''), '')
        AS raw_id,
    COUNT(*) AS file_count,
    SUM(CASE WHEN status = 'INDEXED' THEN 1 ELSE 0 END) AS indexed_file_count,
    SUM(CASE WHEN status IN ('INDEXED', 'EMBEDDED', 'EXTRACTED') THEN 1 ELSE 0 END)
        AS processed_file_count,
    SUM(
        CASE
            WHEN duplicate_of IS NULL OR duplicate_of = ''
            THEN COALESCE(size_bytes, 0)
            ELSE 0
        END
    ) AS source_bytes,
    SUM(
        CASE
            WHEN duplicate_of IS NOT NULL AND duplicate_of != '' THEN 1
            ELSE 0
        END
    ) AS duplicate_file_count
FROM files
GROUP BY raw_id
"""

_CHUNK_SQL = """
SELECT
    COALESCE(NULLIF(TRIM(f.project_code), ''), NULLIF(TRIM(f.project_id), ''), '')
        AS raw_id,
    COUNT(c.chunk_id) AS chunk_count,
    COALESCE(SUM(LENGTH(c.text)), 0) AS logical_text_bytes,
    SUM(CASE WHEN c.embedded = 1 THEN 1 ELSE 0 END) AS embedded_chunk_count,
    COALESCE(SUM(COALESCE(c.token_estimate, 0)), 0) AS token_estimate
FROM chunks c
JOIN files f ON f.file_id = c.file_id
GROUP BY raw_id
"""


def _catalog_projects(conn: sqlite3.Connection) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for r in conn.execute(_PROJECT_SQL):
        key = _normalize_project_id(r["raw_id"])
        bucket = rows.setdefault(key, {
            "project_id": key,
            "file_count": 0,
            "indexed_file_count": 0,
            "processed_file_count": 0,
            "source_bytes": 0,
            "duplicate_file_count": 0,
            "chunk_count": 0,
            "logical_text_bytes": 0,
            "embedded_chunk_count": 0,
            "token_estimate": 0,
        })
        bucket["file_count"] += int(r["file_count"] or 0)
        bucket["indexed_file_count"] += int(r["indexed_file_count"] or 0)
        bucket["processed_file_count"] += int(r["processed_file_count"] or 0)
        bucket["source_bytes"] += int(r["source_bytes"] or 0)
        bucket["duplicate_file_count"] += int(r["duplicate_file_count"] or 0)

    try:
        chunk_rows = conn.execute(_CHUNK_SQL).fetchall()
    except sqlite3.Error:
        chunk_rows = []
    for r in chunk_rows:
        key = _normalize_project_id(r["raw_id"])
        bucket = rows.setdefault(key, {
            "project_id": key,
            "file_count": 0,
            "indexed_file_count": 0,
            "processed_file_count": 0,
            "source_bytes": 0,
            "duplicate_file_count": 0,
            "chunk_count": 0,
            "logical_text_bytes": 0,
            "embedded_chunk_count": 0,
            "token_estimate": 0,
        })
        bucket["chunk_count"] += int(r["chunk_count"] or 0)
        bucket["logical_text_bytes"] += int(r["logical_text_bytes"] or 0)
        bucket["embedded_chunk_count"] += int(r["embedded_chunk_count"] or 0)
        bucket["token_estimate"] += int(r["token_estimate"] or 0)
    return rows


def _configured_project_ids(
    conn: sqlite3.Connection,
    extra: Iterable[str] | None = None,
) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for raw in extra or []:
        key = _normalize_project_id(raw)
        if key == UNKNOWN_PROJECT or key in seen:
            continue
        seen.add(key)
        found.append(key)
    try:
        for r in conn.execute(
            "SELECT project_code FROM project_cards ORDER BY project_code"
        ):
            key = _normalize_project_id(r["project_code"])
            if key == UNKNOWN_PROJECT or key in seen:
                continue
            seen.add(key)
            found.append(key)
    except sqlite3.Error:
        pass
    return found


# ---------------------------------------------------------------------------
# Cost stubs
# ---------------------------------------------------------------------------

def _estimate_tokens(logical_text_bytes: int, token_estimate: int) -> int:
    if token_estimate > 0:
        return token_estimate
    return int(logical_text_bytes / _CHARS_PER_TOKEN)


def stub_cost_fields(
    *,
    chunk_count: int,
    logical_text_bytes: int,
    token_estimate: int,
    indexed_file_count: int,
) -> dict[str, Any]:
    """Planning-only cost stubs. Never derived from storage GB."""
    tokens = _estimate_tokens(logical_text_bytes, token_estimate)
    embed_sgd = round(
        (tokens / 1_000_000.0) * _EMBED_USD_PER_MILLION_TOKENS * _SGD_PER_USD, 4
    )
    # Local extract is $0; generation stub is embed-scale only.
    initial = embed_sgd
    refresh = round(initial * _REFRESH_FRACTION, 4)
    return {
        "currency": "SGD",
        "label": "estimated stub",
        "disclaimer": COST_DISCLAIMER,
        "initial_generation": {
            "amount_sgd": initial,
            "label": "estimated stub",
            "basis": (
                f"~{tokens} tokens × ${_EMBED_USD_PER_MILLION_TOKENS}/1M embed "
                f"× {_SGD_PER_USD} SGD/USD; local extract S$0. "
                "Not invoiced. Not derived from storage GB."
            ),
        },
        "refresh": {
            "amount_sgd": refresh,
            "label": "estimated stub",
            "basis": (
                f"{int(_REFRESH_FRACTION * 100)}% of initial generation "
                "(assumes incremental nightly refresh). Not invoiced."
            ),
        },
        "per_query": {
            "amount_sgd": _PER_QUERY_SGD,
            "label": "estimated stub",
            "basis": (
                "Evidence-pack compose (~8–15 chunks). "
                "Independent of storage GB. Not invoiced."
            ),
        },
        "indexed_files_in_estimate": indexed_file_count,
        "chunks_in_estimate": chunk_count,
    }


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure_operating_storage(
    conn: sqlite3.Connection,
    work_dir: Path,
    *,
    previous: dict[str, Any] | None = None,
    configured_projects: Iterable[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Build a storage snapshot.

    Walks work_dir only (TIGA operating files). Source archive sizes come from
    the files catalog (size_bytes), not a live NAS scan.
    """
    taken_at = now or _utc_now()
    work_dir = Path(work_dir)
    files = walk_work_dir(work_dir)

    categories: dict[str, dict[str, Any]] = {
        name: {
            "bytes": 0,
            "file_count": 0,
            "attribution": "measured",
            "persistent": name in _PERSISTENT_CATEGORIES,
        }
        for name in (
            "sqlite_index",
            "vector_store",
            "atlas",
            "logs",
            "caches",
            "temporary",
            "backups",
            "other",
        )
    }
    atlas_by_slug: dict[str, int] = {}
    work_dir_bytes = 0
    for path, size, category in files:
        work_dir_bytes += size
        categories[category]["bytes"] += size
        categories[category]["file_count"] += 1
        if category == "atlas":
            slug = _atlas_project_slug(path)
            if slug:
                atlas_by_slug[slug] = atlas_by_slug.get(slug, 0) + size

    catalog = _catalog_projects(conn)
    extra_ids = _configured_project_ids(conn, configured_projects)
    for pid in extra_ids:
        catalog.setdefault(pid, {
            "project_id": pid,
            "file_count": 0,
            "indexed_file_count": 0,
            "processed_file_count": 0,
            "source_bytes": 0,
            "duplicate_file_count": 0,
            "chunk_count": 0,
            "logical_text_bytes": 0,
            "embedded_chunk_count": 0,
            "token_estimate": 0,
        })

    # Ensure unknown appears when any unlabelled rows exist, or always if
    # the catalog is empty but work_dir has shared stores.
    if UNKNOWN_PROJECT not in catalog:
        catalog[UNKNOWN_PROJECT] = {
            "project_id": UNKNOWN_PROJECT,
            "file_count": 0,
            "indexed_file_count": 0,
            "processed_file_count": 0,
            "source_bytes": 0,
            "duplicate_file_count": 0,
            "chunk_count": 0,
            "logical_text_bytes": 0,
            "embedded_chunk_count": 0,
            "token_estimate": 0,
        }

    sqlite_bytes = int(categories["sqlite_index"]["bytes"])
    vector_bytes = int(categories["vector_store"]["bytes"])
    atlas_bytes = int(categories["atlas"]["bytes"])

    sqlite_weights = {
        pid: max(row["file_count"] + row["chunk_count"], 0)
        for pid, row in catalog.items()
    }
    vector_weights = {
        pid: max(row["embedded_chunk_count"] or row["chunk_count"], 0)
        for pid, row in catalog.items()
    }
    sqlite_weight_total = sum(sqlite_weights.values())
    vector_weight_total = sum(vector_weights.values())

    def _share(total: int, weight: int, weight_total: int) -> int:
        if total <= 0 or weight_total <= 0 or weight <= 0:
            return 0
        return int(total * weight / weight_total)

    # Allocate shared stores. Remainder (rounding + zero-weight) is unattributed.
    sqlite_allocated = 0
    vector_allocated = 0
    sqlite_share: dict[str, int] = {}
    vector_share: dict[str, int] = {}
    # Deterministic order so rounding remainder is stable.
    project_ids = sorted(
        catalog.keys(),
        key=lambda p: (p == UNKNOWN_PROJECT, p.lower()),
    )
    for pid in project_ids:
        s = _share(sqlite_bytes, sqlite_weights[pid], sqlite_weight_total)
        v = _share(vector_bytes, vector_weights[pid], vector_weight_total)
        sqlite_share[pid] = s
        vector_share[pid] = v
        sqlite_allocated += s
        vector_allocated += v

    sqlite_remainder = max(sqlite_bytes - sqlite_allocated, 0)
    vector_remainder = max(vector_bytes - vector_allocated, 0)

    # Atlas overlays whose slug matches a known project are measured per
    # project; unmatched overlays (orphan files) stay in unattributed.
    used_atlas_slugs: set[str] = set()
    atlas_for: dict[str, int] = {}
    assigned_atlas = 0
    for pid in project_ids:
        slug = _slug(pid)
        size = 0
        if slug not in used_atlas_slugs:
            size = atlas_by_slug.get(slug, 0)
            used_atlas_slugs.add(slug)
        atlas_for[pid] = size
        assigned_atlas += size
    unattributed_atlas = max(atlas_bytes - assigned_atlas, 0)

    prev_projects: dict[str, dict[str, Any]] = {}
    if previous and isinstance(previous.get("projects"), list):
        for p in previous["projects"]:
            if isinstance(p, dict) and p.get("project_id"):
                prev_projects[p["project_id"]] = p
    prev_total = None
    if previous and isinstance(previous.get("totals"), dict):
        prev_total = previous["totals"].get("work_dir_bytes")

    projects_out: list[dict[str, Any]] = []
    allocated_operating = 0
    for pid in project_ids:
        row = catalog[pid]
        atlas_for_project = atlas_for.get(pid, 0)
        operating = sqlite_share[pid] + vector_share[pid] + atlas_for_project
        allocated_operating += operating

        file_count = row["file_count"]
        indexed = row["indexed_file_count"]
        coverage = (indexed / file_count) if file_count else None
        source_bytes = row["source_bytes"]
        ratio = (operating / source_bytes) if source_bytes else None

        if coverage is None:
            projected = None
            projection_assumption = (
                "No catalog files for this project; cannot project a completed index."
            )
        elif coverage == 0:
            projected = None
            projection_assumption = (
                "Zero indexed files so far; cannot project from current operating footprint."
            )
        elif coverage >= 1.0:
            projected = operating
            projection_assumption = (
                "Coverage is 1.0 — projection equals the current operating footprint. "
                "Assumes remaining work is maintenance, not first-time extract/embed."
            )
        else:
            projected = int(round(operating / coverage))
            projection_assumption = (
                "Linear: current operating bytes ÷ indexed-file coverage. "
                "Assumes remaining files have similar extract/embed density. "
                "Does not include the source archive."
            )

        prev = prev_projects.get(pid) or {}
        prev_op = None
        if isinstance(prev.get("operating"), dict):
            prev_op = prev["operating"].get("bytes")
        growth = None if prev_op is None else operating - int(prev_op)

        identity = "unknown" if pid == UNKNOWN_PROJECT else "known"
        # Atlas bytes for this slug are measured; shared stores are estimated.
        if operating == 0:
            attribution = "measured"
        elif sqlite_share[pid] or vector_share[pid]:
            attribution = "estimated" if atlas_for_project == 0 else "mixed"
        else:
            attribution = "measured"

        projects_out.append({
            "project_id": pid,
            "identity": identity,
            "identity_note": (
                "Missing project identity — catalog rows had empty/Unknown "
                "project_code and project_id."
                if identity == "unknown"
                else "Resolved from project_code, else project_id."
            ),
            "file_count": file_count,
            "indexed_file_count": indexed,
            "processed_file_count": row["processed_file_count"],
            "chunk_count": row["chunk_count"],
            "page_reference_count": row["chunk_count"],
            "embedded_chunk_count": row["embedded_chunk_count"],
            "duplicate_file_count": row["duplicate_file_count"],
            "coverage": None if coverage is None else round(coverage, 4),
            "coverage_label": (
                None if coverage is None else f"{coverage * 100:.1f}%"
            ),
            "source_archive_bytes": source_bytes,
            "source_archive_gb": bytes_to_gb(source_bytes),
            "source_archive_human": format_bytes(source_bytes),
            "source_archive_attribution": "catalog",
            "source_archive_note": (
                "Sum of files.size_bytes for non-duplicate catalog rows. "
                "Not a live NAS walk."
            ),
            "logical_text_bytes": row["logical_text_bytes"],
            "logical_text_note": (
                "UTF-8 length of extracted chunk text. Not on-disk DB size."
            ),
            "operating": {
                "bytes": operating,
                "gb": bytes_to_gb(operating),
                "human": format_bytes(operating),
                "attribution": attribution,
                "allocation": (
                    "SQLite and vector bytes: estimated share of measured "
                    "shared-store totals (weight = files + chunks for SQLite, "
                    "embedded chunks for vectors). Atlas overlay: measured "
                    "when the filename slug matches the project."
                ),
                "components": {
                    "sqlite": {
                        "bytes": sqlite_share[pid],
                        "attribution": "estimated" if sqlite_share[pid] else "measured",
                    },
                    "vectors": {
                        "bytes": vector_share[pid],
                        "attribution": "estimated" if vector_share[pid] else "measured",
                    },
                    "atlas": {
                        "bytes": atlas_for_project,
                        "attribution": "measured",
                    },
                },
            },
            "index_to_source_ratio": None if ratio is None else round(ratio, 6),
            "growth_bytes": growth,
            "growth_human": format_bytes(growth) if growth is not None else None,
            "projected_completed_bytes": projected,
            "projected_completed_human": format_bytes(projected),
            "projection_assumption": projection_assumption,
            "cost": stub_cost_fields(
                chunk_count=row["chunk_count"],
                logical_text_bytes=row["logical_text_bytes"],
                token_estimate=row["token_estimate"],
                indexed_file_count=indexed,
            ),
        })

    unattributed_bytes = (
        int(categories["logs"]["bytes"])
        + int(categories["caches"]["bytes"])
        + int(categories["temporary"]["bytes"])
        + int(categories["backups"]["bytes"])
        + int(categories["other"]["bytes"])
        + sqlite_remainder
        + vector_remainder
        + unattributed_atlas
    )
    persistent_bytes = sum(
        int(categories[c]["bytes"]) for c in _PERSISTENT_CATEGORIES
    )
    temporary_bytes = sum(
        int(categories[c]["bytes"]) for c in _TEMPORARY_CATEGORIES
    )

    source_total = sum(p["source_archive_bytes"] for p in projects_out)
    index_to_source = (
        (work_dir_bytes / source_total) if source_total else None
    )
    growth_total = None if prev_total is None else work_dir_bytes - int(prev_total)

    recon_delta = (allocated_operating + unattributed_bytes) - work_dir_bytes
    snapshot: dict[str, Any] = {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "snapshot_at": _iso(taken_at),
        "work_dir": str(work_dir.resolve()) if work_dir.exists() else str(work_dir),
        "units": {
            "bytes": "bytes on disk (1 B)",
            "gb": "gibibytes (bytes / 1024^3)",
            "human": "B / KiB / MiB / GiB (1024)",
            "note": (
                "Logical extracted-text length is reported separately and is "
                "not on-disk database size."
            ),
        },
        "totals": {
            "work_dir_bytes": work_dir_bytes,
            "work_dir_gb": bytes_to_gb(work_dir_bytes),
            "work_dir_human": format_bytes(work_dir_bytes),
            "source_archive_bytes": source_total,
            "source_archive_gb": bytes_to_gb(source_total),
            "source_archive_human": format_bytes(source_total),
            "source_archive_attribution": "catalog",
            "persistent_bytes": persistent_bytes,
            "persistent_human": format_bytes(persistent_bytes),
            "temporary_and_backup_bytes": temporary_bytes,
            "temporary_and_backup_human": format_bytes(temporary_bytes),
            "index_to_source_ratio": (
                None if index_to_source is None else round(index_to_source, 6)
            ),
            "file_count": sum(p["file_count"] for p in projects_out),
            "chunk_count": sum(p["chunk_count"] for p in projects_out),
            "indexed_file_count": sum(p["indexed_file_count"] for p in projects_out),
        },
        "categories": categories,
        "shared": {
            "sqlite_bytes": sqlite_bytes,
            "sqlite_human": format_bytes(sqlite_bytes),
            "sqlite_attribution": "measured",
            "vector_bytes": vector_bytes,
            "vector_human": format_bytes(vector_bytes),
            "vector_attribution": "measured",
            "atlas_bytes": atlas_bytes,
            "atlas_attribution": "measured",
            "sqlite_allocated_bytes": sqlite_allocated,
            "vector_allocated_bytes": vector_allocated,
            "sqlite_unattributed_bytes": sqlite_remainder,
            "vector_unattributed_bytes": vector_remainder,
            "allocation_method": (
                "Shared SQLite / LanceDB totals are measured from work_dir. "
                "Per-project shares are estimated: SQLite weight = file_count + "
                "chunk_count; vector weight = embedded_chunk_count (fallback "
                "chunk_count). Remainder after integer allocation is "
                "unattributed overhead. Logical text length is not used as disk size."
            ),
        },
        "projects": projects_out,
        "unattributed": {
            "bytes": unattributed_bytes,
            "gb": bytes_to_gb(unattributed_bytes),
            "human": format_bytes(unattributed_bytes),
            "attribution": "measured remainder",
            "explanation": (
                "Logs, caches, temporary files, backups, config/other, "
                "unmatched Atlas files, and integer-rounding remainder of "
                "shared SQLite/vector stores. Included so project allocations "
                "reconcile to total physical work_dir usage."
            ),
        },
        "reconciliation": {
            "allocated_bytes": allocated_operating,
            "unattributed_bytes": unattributed_bytes,
            "work_dir_bytes": work_dir_bytes,
            "delta_bytes": recon_delta,
            "ok": recon_delta == 0,
        },
        "previous_snapshot_at": (
            previous.get("snapshot_at") if previous else None
        ),
        "growth": {
            "work_dir_bytes": growth_total,
            "work_dir_human": (
                format_bytes(growth_total) if growth_total is not None else None
            ),
        },
        "cost_disclaimer": COST_DISCLAIMER,
        "measurement": {
            "mode": "fresh",
            "scanned_work_dir": True,
            "scanned_source_archive": False,
            "cache_ttl_seconds": DEFAULT_CACHE_TTL_SECONDS,
        },
    }
    return snapshot


# ---------------------------------------------------------------------------
# Cache / report
# ---------------------------------------------------------------------------

def load_latest_snapshot(work_dir: Path) -> dict[str, Any] | None:
    path = snapshot_path(work_dir)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def save_snapshot(work_dir: Path, snapshot: dict[str, Any]) -> Path:
    path = snapshot_path(work_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _snapshot_age_seconds(snapshot: dict[str, Any], now: datetime) -> float | None:
    raw = snapshot.get("snapshot_at")
    if not raw:
        return None
    try:
        then = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    if then.tzinfo is None:
        then = then.replace(tzinfo=timezone.utc)
    return max((now - then).total_seconds(), 0.0)


def get_storage_report(
    conn: sqlite3.Connection,
    work_dir: Path,
    *,
    refresh: bool = False,
    max_age_seconds: int = DEFAULT_CACHE_TTL_SECONDS,
    configured_projects: Iterable[str] | None = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """
    Return a cached snapshot when fresh enough; otherwise remasure and persist.

    Page loads should call this with refresh=False so we do not walk work_dir
    (and never the NAS archive) on every request.
    """
    taken_at = now or _utc_now()
    cached = load_latest_snapshot(work_dir)
    if cached and not refresh:
        age = _snapshot_age_seconds(cached, taken_at)
        if age is not None and age <= max_age_seconds:
            cached = dict(cached)
            measurement = dict(cached.get("measurement") or {})
            measurement["mode"] = "cached"
            measurement["age_seconds"] = int(age)
            measurement["scanned_work_dir"] = False
            measurement["scanned_source_archive"] = False
            cached["measurement"] = measurement
            return cached

    snapshot = measure_operating_storage(
        conn,
        work_dir,
        previous=cached,
        configured_projects=configured_projects,
        now=taken_at,
    )
    save_snapshot(work_dir, snapshot)
    return snapshot


def format_storage_report(snapshot: dict[str, Any]) -> str:
    """Readable CLI / log report."""
    totals = snapshot.get("totals") or {}
    shared = snapshot.get("shared") or {}
    unattr = snapshot.get("unattributed") or {}
    recon = snapshot.get("reconciliation") or {}
    growth = snapshot.get("growth") or {}
    lines = [
        "TIGA operating storage",
        f"Snapshot: {snapshot.get('snapshot_at', '—')}"
        f"  ({(snapshot.get('measurement') or {}).get('mode', 'fresh')})",
        f"Work dir: {snapshot.get('work_dir', '—')}",
        f"  Total: {totals.get('work_dir_human', format_bytes(totals.get('work_dir_bytes', 0)))}"
        f"  ({totals.get('work_dir_bytes', 0)} B)",
        f"  Persistent: {totals.get('persistent_human', '—')}"
        f"  · temp/backup: {totals.get('temporary_and_backup_human', '—')}",
        f"Source archive (catalog, not live NAS): "
        f"{totals.get('source_archive_human', '—')}"
        f"  ({totals.get('source_archive_bytes', 0)} B)",
        f"Index-to-source ratio: {totals.get('index_to_source_ratio', '—')}",
        f"Catalog: {totals.get('file_count', 0)} files, "
        f"{totals.get('chunk_count', 0)} chunks, "
        f"{totals.get('indexed_file_count', 0)} indexed",
        "",
        "Shared stores (measured exactly)",
        f"  SQLite index: {shared.get('sqlite_human', '—')}"
        f"  ({shared.get('sqlite_bytes', 0)} B)",
        f"  Vector store: {shared.get('vector_human', '—')}"
        f"  ({shared.get('vector_bytes', 0)} B)",
        f"  {shared.get('allocation_method', '')}",
        "",
        "Projects",
    ]
    for p in snapshot.get("projects") or []:
        op = p.get("operating") or {}
        ident = "  [unknown identity]" if p.get("identity") == "unknown" else ""
        lines.append(
            f"  {p.get('project_id')}: {op.get('human', '—')} "
            f"({op.get('bytes', 0)} B, {op.get('attribution', '?')})"
            f"{ident}"
        )
        lines.append(
            f"    files {p.get('file_count', 0)}  indexed {p.get('indexed_file_count', 0)}  "
            f"chunks {p.get('chunk_count', 0)}  "
            f"coverage {p.get('coverage_label') or '—'}  "
            f"idx/src {p.get('index_to_source_ratio', '—')}"
        )
        lines.append(
            f"    source (catalog) {p.get('source_archive_human', '—')}  "
            f"logical text {format_bytes(p.get('logical_text_bytes', 0))} "
            f"(not disk)"
        )
        if p.get("growth_bytes") is not None:
            lines.append(f"    growth since last snapshot: {p.get('growth_human')}")
        lines.append(
            f"    projected completed index: {p.get('projected_completed_human', '—')}"
        )
        lines.append(f"    {p.get('projection_assumption', '')}")
        cost = p.get("cost") or {}
        gen = (cost.get("initial_generation") or {}).get("amount_sgd")
        ref = (cost.get("refresh") or {}).get("amount_sgd")
        pq = (cost.get("per_query") or {}).get("amount_sgd")
        lines.append(
            f"    cost stubs (SGD, estimated): generate {gen}  "
            f"refresh {ref}  per-query {pq}"
        )

    lines += [
        "",
        "Unattributed / shared overhead",
        f"  {unattr.get('human', '—')} ({unattr.get('bytes', 0)} B)  "
        f"[{unattr.get('attribution', '')}]",
        f"  {unattr.get('explanation', '')}",
        "",
        "Reconciliation",
        f"  allocated {recon.get('allocated_bytes', 0)} B + "
        f"unattributed {recon.get('unattributed_bytes', 0)} B = "
        f"work_dir {recon.get('work_dir_bytes', 0)} B  "
        f"delta {recon.get('delta_bytes', 0)} B  "
        f"{'OK' if recon.get('ok') else 'MISMATCH'}",
    ]
    if growth.get("work_dir_bytes") is not None:
        lines.append(f"  growth since {snapshot.get('previous_snapshot_at')}: "
                     f"{growth.get('work_dir_human')}")
    lines += [
        "",
        snapshot.get("cost_disclaimer") or COST_DISCLAIMER,
    ]
    return "\n".join(lines)
