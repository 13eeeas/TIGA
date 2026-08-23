"""
core/pipeline_validate.py — Dry-run index pipeline + search-only benchmark.

Use before pointing at real NAS roots:
  python tiga.py validate

Runs discover → extract → embed (mocked or real) → FTS on fixture archive,
then scores Hunt retrieval without any LLM/API compose.
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import yaml

from config import Config, load_config
from core.db import get_connection
from core.eval import _has_hit, _load_fixture, _percentile, validate_citation
from core.index import run_full_pipeline
from core.query import search

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FIXTURE_ARCHIVE = REPO_ROOT / "tests" / "fixtures" / "mini_archive"
DEFAULT_SEARCH_BENCHMARK = REPO_ROOT / "tests" / "fixtures" / "search_benchmark.yaml"
DEFAULT_DUAL_BENCHMARK = REPO_ROOT / "tests" / "fixtures" / "search_benchmark_dual.yaml"

_GATEWAY_TOP5_TARGET = 0.90
_GATEWAY_PARAPHRASE_TARGET = 0.80


@contextmanager
def mock_embed_if_needed(use_mock: bool):
    """Patch Ollama/LanceDB embed path so validate runs without GPU/API."""
    if not use_mock:
        yield
        return

    class _FakeMergeBuilder:
        def when_matched_update_all(self):
            return self

        def when_not_matched_insert_all(self):
            return self

        def execute(self, _rows):
            return None

    class _FakeTable:
        def merge_insert(self, _key):
            return _FakeMergeBuilder()

    def _batched(texts, cfg, batch_size=None):
        return [[0.1] * 768 for _ in texts]

    with patch("core.index._vectors.embed_texts_batched", side_effect=_batched), \
         patch("core.index._vectors._get_chunk_table", return_value=_FakeTable()), \
         patch("core.vectors.embed_texts_batched", side_effect=_batched), \
         patch("core.vectors.query_vector", return_value=[]), \
         patch("lancedb.connect", return_value=MagicMock()):
        yield


def setup_sandbox(
    work_dir: Path,
    fixture_archive: Path | None = None,
) -> tuple[Config, Path]:
    """
    Create an isolated tiga_work sandbox with fixture archive copied in.
    Returns (config, archive_root inside sandbox).
    """
    src = fixture_archive or DEFAULT_FIXTURE_ARCHIVE
    if not src.exists():
        raise FileNotFoundError(f"Fixture archive not found: {src}")

    work_dir = work_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    archive_root = work_dir / "fixture_archive"
    if archive_root.exists():
        shutil.rmtree(archive_root)
    shutil.copytree(src, archive_root)

    cfg_file = work_dir / "config.yaml"
    cfg_data = {
        "index_roots": [str(archive_root)],
        "retrieval": {
            "top_k_default": 10,
            # Hunt Google-close signals ON even without Ollama/vectors
            "reranker_enabled": False,
            "prefer_latest_default": True,
            "path_boost_enabled": True,
            "project_code_boost_enabled": True,
            "phrase_match_enabled": True,
            "domain_expand_enabled": True,
            "max_chunks_per_file": 2,
        },
        "compose": {
            "api_enabled": False,
            "provider": "ollama",
        },
        "pipeline": {
            "extract_workers": 1,
            "fingerprint_strategy": "metadata",
        },
        "scheduler": {
            "day_run_indexing": True,
            "night_run_indexing": True,
        },
    }
    cfg_file.write_text(yaml.dump(cfg_data), encoding="utf-8")
    cfg = load_config(config_file=cfg_file, work_dir=work_dir)
    cfg.ensure_dirs()
    return cfg, archive_root


def run_index_validation(
    cfg: Config,
    conn: sqlite3.Connection,
    *,
    mock_embed: bool = True,
) -> dict[str, Any]:
    """Run full index pipeline; return stats + integrity checks."""
    t0 = time.perf_counter()
    stats = run_full_pipeline(conn, cfg)

    indexed = conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE status='INDEXED'"
    ).fetchone()["n"]
    chunks = conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
    fts_rows = conn.execute("SELECT COUNT(*) AS n FROM chunks_fts").fetchone()["n"]
    extracted = conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE status IN ('EXTRACTED','EMBEDDED','INDEXED')"
    ).fetchone()["n"]

    errors: list[str] = []
    if indexed == 0:
        errors.append("no files reached INDEXED status")
    if chunks == 0:
        errors.append("no chunks in database")
    if fts_rows == 0:
        errors.append("FTS index empty (chunks_fts has 0 rows)")

    return {
        "stats": stats,
        "files_indexed": indexed,
        "files_extracted_or_better": extracted,
        "chunk_count": chunks,
        "fts_row_count": fts_rows,
        "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        "mock_embed": mock_embed,
        "ok": len(errors) == 0,
        "errors": errors,
    }


def _score_query_set(
    entries: list[dict[str, Any]],
    cfg: Config,
    *,
    top_k: int,
    verbose: bool,
    label: str,
) -> dict[str, Any]:
    """Score a list of {query, expected_paths} entries — Hunt only."""
    db_path = str(cfg.get_db_path())
    root_paths = [str(r) for r in cfg.index_roots]
    hits = 0
    invalid_citations = 0
    total_citations = 0
    latencies: list[float] = []
    queries_out: list[dict[str, Any]] = []

    if verbose and entries:
        print(f"\n  [{label}] {len(entries)} queries")

    for entry in entries:
        q = entry.get("query", "")
        expected = entry.get("expected_paths", [])
        t0 = time.perf_counter()
        try:
            results = search(q, top_k=top_k, cfg_obj=cfg)
        except Exception as exc:
            logger.warning("search failed for %r: %s", q, exc)
            results = []
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        latencies.append(elapsed)

        returned_paths = [r.get("file_path", "") for r in results]
        hit = _has_hit(returned_paths, expected) if expected else False
        if hit:
            hits += 1

        q_citations = [r.get("citation", "") for r in results if r.get("citation")]
        bad = [c for c in q_citations if not validate_citation(c, db_path, root_paths)]
        invalid_citations += len(bad)
        total_citations += len(q_citations)

        row = {
            "query": q,
            "hit": hit,
            "elapsed_ms": elapsed,
            "expected_paths": expected,
            "returned_paths": returned_paths[:top_k],
            "top_citation": q_citations[0] if q_citations else "",
            "invalid_citations": bad,
            "bucket": label,
        }
        queries_out.append(row)
        if verbose:
            mark = "PASS" if hit else "MISS"
            print(f"\n[{mark}] ({label}) {q!r}  ({elapsed:.0f} ms)")
            for p in returned_paths[:3]:
                print(f"       {p}")

    n = len(entries)
    recall = round(hits / n, 4) if n else 0.0
    citation_valid_pct = (
        round((1 - invalid_citations / total_citations) * 100, 2)
        if total_citations else 100.0
    )
    return {
        "label": label,
        "total_queries": n,
        "hits": hits,
        "top5_recall": recall,
        "top5_recall_pct": round(recall * 100, 1),
        "citation_valid_pct": citation_valid_pct,
        "invalid_citations": invalid_citations,
        "latency_p50_ms": round(_percentile(latencies, 50), 1),
        "latency_p95_ms": round(_percentile(latencies, 95), 1),
        "queries": queries_out,
    }


def run_search_benchmark(
    cfg: Config,
    fixture_path: Path | None = None,
    *,
    top_k: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Hunt-only retrieval benchmark — no compose, no API.

    Supports:
      - list YAML (legacy search_benchmark.yaml)
      - dual YAML with `literal:` + `paraphrase:` keys (preferred)
    """
    path = fixture_path or DEFAULT_DUAL_BENCHMARK
    if not path.exists() and fixture_path is None:
        path = DEFAULT_SEARCH_BENCHMARK

    raw = yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else None
    if not raw:
        raise FileNotFoundError(f"No benchmark queries at {path}")

    k = top_k or cfg.top_k
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")

    if isinstance(raw, dict) and ("literal" in raw or "paraphrase" in raw):
        literal_entries = raw.get("literal") or []
        paraphrase_entries = raw.get("paraphrase") or []
        lit = _score_query_set(literal_entries, cfg, top_k=k, verbose=verbose, label="literal")
        para = _score_query_set(
            paraphrase_entries, cfg, top_k=k, verbose=verbose, label="paraphrase"
        )
        all_queries = lit["queries"] + para["queries"]
        total = lit["total_queries"] + para["total_queries"]
        hits = lit["hits"] + para["hits"]
        overall = round(hits / total, 4) if total else 0.0
        invalid = lit["invalid_citations"] + para["invalid_citations"]
        gateway_pass = (
            lit["top5_recall"] >= _GATEWAY_TOP5_TARGET
            and para["top5_recall"] >= _GATEWAY_PARAPHRASE_TARGET
            and invalid == 0
        )
        report = {
            "mode": "search_only_dual",
            "ts": ts,
            "total_queries": total,
            "top5_recall": overall,
            "top5_recall_pct": round(overall * 100, 1),
            "literal_recall_pct": lit["top5_recall_pct"],
            "paraphrase_recall_pct": para["top5_recall_pct"],
            "gateway_target_pct": _GATEWAY_TOP5_TARGET * 100,
            "paraphrase_target_pct": _GATEWAY_PARAPHRASE_TARGET * 100,
            "gateway_pass": gateway_pass,
            "citation_valid_pct": min(lit["citation_valid_pct"], para["citation_valid_pct"]),
            "invalid_citations": invalid,
            "latency_p50_ms": lit["latency_p50_ms"],
            "latency_p95_ms": max(lit["latency_p95_ms"], para["latency_p95_ms"]),
            "literal": lit,
            "paraphrase": para,
            "queries": all_queries,
        }
        return report

    # Legacy list format
    fixture = raw if isinstance(raw, list) else _load_fixture(path)
    scored = _score_query_set(fixture, cfg, top_k=k, verbose=verbose, label="legacy")
    report = {
        "mode": "search_only",
        "ts": ts,
        "total_queries": scored["total_queries"],
        "top5_recall": scored["top5_recall"],
        "top5_recall_pct": scored["top5_recall_pct"],
        "gateway_target_pct": _GATEWAY_TOP5_TARGET * 100,
        "gateway_pass": scored["top5_recall"] >= _GATEWAY_TOP5_TARGET
        and scored["invalid_citations"] == 0,
        "citation_valid_pct": scored["citation_valid_pct"],
        "invalid_citations": scored["invalid_citations"],
        "latency_p50_ms": scored["latency_p50_ms"],
        "latency_p95_ms": scored["latency_p95_ms"],
        "queries": scored["queries"],
    }
    return report


def run_validate(
    work_dir: Path | None = None,
    *,
    fixture_archive: Path | None = None,
    benchmark_fixture: Path | None = None,
    mock_embed: bool = True,
    verbose: bool = True,
) -> int:
    """
    Full pre-flight: sandbox index + search benchmark.
    Exit 0 if index OK and benchmark passes gateway citation gate.
    """
    sandbox = work_dir or (REPO_ROOT / "tiga_work_validate")
    cfg, archive_root = setup_sandbox(sandbox, fixture_archive)
    conn = get_connection(cfg.get_db_path())

    try:
        if verbose:
            print(f"\n{'=' * 60}")
            print("  TIGA validate — index pipeline (fixture, no NAS)")
            print(f"  Archive: {archive_root}")
            print(f"  Embed:   {'mocked (no Ollama)' if mock_embed else 'real Ollama'}")
            print(f"{'=' * 60}\n")

        with mock_embed_if_needed(mock_embed):
            index_report = run_index_validation(cfg, conn, mock_embed=mock_embed)

            if verbose:
                print("Index pipeline")
                print(f"  files INDEXED : {index_report['files_indexed']}")
                print(f"  chunks        : {index_report['chunk_count']}")
                print(f"  FTS rows      : {index_report['fts_row_count']}")
                print(f"  elapsed       : {index_report['elapsed_ms']:.0f} ms")
                if index_report["errors"]:
                    for err in index_report["errors"]:
                        print(f"  ERROR: {err}")

            if not index_report["ok"]:
                return 1

            if verbose:
                print(f"\n{'-' * 60}")
                print("  Search benchmark (Hunt only — no API)")
                print(f"{'-' * 60}")

            search_report = run_search_benchmark(
                cfg, benchmark_fixture, verbose=verbose
            )

        report_dir = cfg.get_report_dir()
        report_dir.mkdir(parents=True, exist_ok=True)
        out_path = report_dir / f"validate_{search_report['ts']}.json"
        combined = {"index": index_report, "search": search_report}
        out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")

        if verbose:
            print(f"\n{'-' * 60}")
            if search_report.get("mode") == "search_only_dual":
                print(f"  Literal top-5     : {search_report.get('literal_recall_pct', '—')}%  "
                      f"(target {_GATEWAY_TOP5_TARGET * 100:.0f}%)")
                print(f"  Paraphrase top-5  : {search_report.get('paraphrase_recall_pct', '—')}%  "
                      f"(target {_GATEWAY_PARAPHRASE_TARGET * 100:.0f}%)")
            print(f"  Overall top-5     : {search_report['top5_recall_pct']:.0f}%")
            print(f"  Citation valid   : {search_report['citation_valid_pct']:.0f}%")
            print(f"  Latency p50/p95  : {search_report['latency_p50_ms']:.0f} / "
                  f"{search_report['latency_p95_ms']:.0f} ms")
            print(f"  Report           : {out_path}")
            gate = "PASS" if search_report["gateway_pass"] else "NEEDS WORK"
            print(f"  Search gate      : {gate}")
            print(f"{'-' * 60}\n")

        if search_report["invalid_citations"] > 0:
            return 1
        return 0 if search_report["gateway_pass"] else 2

    finally:
        conn.close()


def run_validate_job(
    work_dir: Path | None = None,
    *,
    fixture_archive: Path | None = None,
    benchmark_fixture: Path | None = None,
    mock_embed: bool = True,
) -> dict[str, Any]:
    """Structured validate result for API/admin."""
    sandbox = work_dir or (REPO_ROOT / "tiga_work_validate")
    cfg, archive_root = setup_sandbox(sandbox, fixture_archive)
    conn = get_connection(cfg.get_db_path())
    try:
        with mock_embed_if_needed(mock_embed):
            index_report = run_index_validation(cfg, conn, mock_embed=mock_embed)
            if not index_report["ok"]:
                return {
                    "exit_code": 1,
                    "archive_root": str(archive_root),
                    "mock_embed": mock_embed,
                    "index": index_report,
                    "search": None,
                    "report_path": None,
                    "gateway_pass": False,
                }
            search_report = run_search_benchmark(
                cfg, benchmark_fixture, verbose=False
            )
        report_dir = cfg.get_report_dir()
        report_dir.mkdir(parents=True, exist_ok=True)
        out_path = report_dir / f"validate_{search_report['ts']}.json"
        combined = {"index": index_report, "search": search_report}
        out_path.write_text(json.dumps(combined, indent=2), encoding="utf-8")
        exit_code = 1 if search_report.get("invalid_citations", 0) > 0 else (
            0 if search_report.get("gateway_pass") else 2
        )
        return {
            "exit_code": exit_code,
            "archive_root": str(archive_root),
            "mock_embed": mock_embed,
            "index": index_report,
            "search": search_report,
            "report_path": str(out_path),
            "gateway_pass": bool(search_report.get("gateway_pass")),
        }
    finally:
        conn.close()


def list_validate_reports(*report_dirs: Path) -> list[dict[str, Any]]:
    """List validate_*.json reports newest first."""
    seen: set[str] = set()
    items: list[dict[str, Any]] = []
    for d in report_dirs:
        if not d or not Path(d).exists():
            continue
        for path in sorted(Path(d).glob("validate_*.json"), reverse=True):
            key = str(path.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                search = data.get("search") or {}
                index = data.get("index") or {}
                items.append({
                    "path": str(path),
                    "name": path.name,
                    "ts": search.get("ts") or path.stem.replace("validate_", ""),
                    "top5_recall_pct": search.get("top5_recall_pct"),
                    "literal_recall_pct": search.get("literal_recall_pct"),
                    "paraphrase_recall_pct": search.get("paraphrase_recall_pct"),
                    "gateway_pass": search.get("gateway_pass"),
                    "files_indexed": index.get("files_indexed"),
                    "mock_embed": index.get("mock_embed"),
                    "report": data,
                })
            except Exception as exc:
                logger.warning("Skipping report %s: %s", path, exc)
    items.sort(key=lambda r: r.get("ts") or "", reverse=True)
    return items
