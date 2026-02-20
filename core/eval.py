"""
core/eval.py — Evaluation harness for search quality.

Usage:
  python tiga.py eval                        # load fixture
  python tiga.py eval --queries "hospital"   # ad-hoc

Reads:  tiga_work/fixtures/eval_queries.yaml
Writes: tiga_work/reports/eval_<timestamp>.json
Exit:   0 = pass  |  1 = invalid citations detected (hard gate)

validate_citation() is exported at module level so query.py / app.py
can import and reuse it for runtime citation checking.
"""

from __future__ import annotations

import json
import logging
import sqlite3 as _sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from config import cfg
from core.db import file_id_from_path

logger = logging.getLogger(__name__)

_FIXTURE_FILENAME = "eval_queries.yaml"


# ---------------------------------------------------------------------------
# Citation validator
# ---------------------------------------------------------------------------

def validate_citation(citation: str, db_path: str, root_paths: list[str]) -> bool:
    """
    Validate a citation string against disk + DB.

    Formats:
      "rel/path/to/file.pdf#p12"
      "[rootLabel] rel/path/to/file.pdf#p12"

    Check 1: file exists on disk under at least one of root_paths.
    Check 2: chunk with matching file_id + ref_value exists in DB.
    Returns True only if both pass. Never raises.
    """
    try:
        s = str(citation).strip()

        # Strip optional root label "[rootA] …"
        if s.startswith("["):
            bracket_end = s.index("]")
            s = s[bracket_end + 1:].strip()

        if "#" not in s:
            return False

        rel, ref = s.rsplit("#", 1)
        rel = rel.strip()
        ref = ref.strip()
        if not rel or not ref:
            return False

        # Check 1: file exists on disk under any root
        found_abs: Path | None = None
        for root in root_paths:
            try:
                candidate = Path(root) / rel
                if candidate.exists():
                    found_abs = candidate.resolve()
                    break
            except OSError:
                continue

        if found_abs is None:
            return False

        # Check 2: chunk exists in DB
        file_id = file_id_from_path(found_abs.as_posix())
        conn = _sqlite3.connect(db_path, check_same_thread=False)
        try:
            row = conn.execute(
                "SELECT 1 FROM chunks WHERE file_id=? AND ref_value=? LIMIT 1",
                (file_id, ref),
            ).fetchone()
            return row is not None
        finally:
            conn.close()

    except Exception:
        return False


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

def _load_fixture(fixture_path: Path) -> list[dict[str, Any]]:
    if not fixture_path.exists():
        return []
    with fixture_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, list) else []


def _ad_hoc_fixture(queries: list[str]) -> list[dict[str, Any]]:
    return [{"query": q, "expected_paths": []} for q in queries]


# ---------------------------------------------------------------------------
# Percentile (no numpy — stdlib only)
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = p / 100.0 * (len(sv) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(sv) - 1)
    return sv[lo] + (sv[hi] - sv[lo]) * (idx - lo)


# ---------------------------------------------------------------------------
# Recall helper
# ---------------------------------------------------------------------------

def _has_hit(returned_paths: list[str], expected_paths: list[str]) -> bool:
    if not expected_paths:
        return False
    for ep in expected_paths:
        ep_norm = ep.replace("\\", "/")
        for fp in returned_paths:
            fp_norm = fp.replace("\\", "/")
            if fp_norm.endswith(ep_norm) or ep_norm in fp_norm:
                return True
    return False


# ---------------------------------------------------------------------------
# Eval runner
# ---------------------------------------------------------------------------

def run_eval(
    queries: list[str] | None = None,
    top_k: int | None = None,
    verbose: bool = True,
) -> int:
    """
    Run eval and return exit code (0 = pass, 1 = invalid citations found).

    queries: if None, load from tiga_work/fixtures/eval_queries.yaml.
             if provided, treat as ad-hoc (no expected_paths checked).
    """
    from core.query import search

    cfg.ensure_dirs()
    k = top_k or cfg.top_k
    db_path = str(cfg.get_db_path())
    root_paths = [str(r) for r in cfg.index_roots]

    # Load fixture
    if queries:
        fixture = _ad_hoc_fixture(queries)
    else:
        fixture_path = cfg.work_dir / "fixtures" / _FIXTURE_FILENAME
        fixture = _load_fixture(fixture_path)
        if not fixture:
            print(
                f"No eval fixture found at {fixture_path}\n"
                "Run `python tiga.py init` to generate it, then add your test queries.",
                file=sys.stderr,
            )
            return 0

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    query_reports: list[dict[str, Any]] = []
    latencies_ms: list[float] = []
    hits = 0
    total_citations = 0
    invalid_count = 0

    for entry in fixture:
        query_str = entry.get("query", "")
        expected_paths: list[str] = entry.get("expected_paths", [])

        # Run query — empty result is OK pre-Phase 4
        t0 = time.perf_counter()
        try:
            results = search(query_str, top_k=k)
        except Exception as e:
            logger.warning("Search failed for %r: %s — treating as empty", query_str, e)
            results = []
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        latencies_ms.append(elapsed_ms)

        # Recall
        returned_paths = [r.get("file_path", "") for r in results]
        hit = _has_hit(returned_paths, expected_paths)
        if hit:
            hits += 1

        # Citations — collected from results once query engine returns them
        q_citations: list[str] = []
        for r in results:
            q_citations.extend(r.get("citations", []))
        q_invalid = [c for c in q_citations if not validate_citation(c, db_path, root_paths)]
        total_citations += len(q_citations)
        invalid_count  += len(q_invalid)

        q_report: dict[str, Any] = {
            "query":             query_str,
            "hit":               hit,
            "elapsed_ms":        elapsed_ms,
            "returned_paths":    returned_paths[:k],
            "expected_paths":    expected_paths,
            "citations":         q_citations,
            "invalid_citations": q_invalid,
        }
        query_reports.append(q_report)

        if verbose:
            marker = "PASS" if hit else "MISS"
            print(f"\n[{marker}] {query_str!r}  ({elapsed_ms:.0f} ms)")
            for path in returned_paths[:k]:
                print(f"       {path}")
            if q_invalid:
                print(f"       !! INVALID CITATIONS: {q_invalid}", file=sys.stderr)

    # Aggregate
    n = len(fixture)
    top5_recall = round(hits / n, 4) if n else 0.0
    citation_valid_pct = (
        round((1 - invalid_count / total_citations) * 100, 2)
        if total_citations > 0 else 100.0
    )
    p50 = round(_percentile(latencies_ms, 50), 1)
    p95 = round(_percentile(latencies_ms, 95), 1)
    exit_code = 1 if invalid_count > 0 else 0

    report: dict[str, Any] = {
        "ts":                 ts,
        "total_queries":      n,
        "top5_recall":        top5_recall,
        "citation_valid_pct": citation_valid_pct,
        "invalid_citations":  invalid_count,
        "total_citations":    total_citations,
        "latency_p50_ms":     p50,
        "latency_p95_ms":     p95,
        "exit_code":          exit_code,
        "queries":            query_reports,
    }

    # Save JSON report
    report_dir = cfg.get_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"eval_{ts}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Summary
    gate = "PASS" if exit_code == 0 else "FAIL"
    print(f"\n{'─' * 52}")
    print(f"  Queries:          {n}")
    print(f"  Top-5 recall:     {top5_recall * 100:.0f}%")
    print(f"  Citation valid:   {citation_valid_pct:.0f}%  [{gate}]")
    print(f"  Latency p50/p95:  {p50:.0f} / {p95:.0f} ms")
    print(f"  Report:           {report_path}")
    print(f"{'─' * 52}")

    if exit_code == 1:
        print(
            f"\n  GATE FAIL: {invalid_count} invalid citation(s) detected.",
            file=sys.stderr,
        )

    return exit_code
