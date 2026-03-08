#!/usr/bin/env python3
"""
test_run.py — TIGA automated test runner.

Runs the full TIGA pipeline against ONE folder and writes a detailed
error/debug report. Use this before committing to the full archive index.

Usage:
    python test_run.py /path/to/test/folder
    python test_run.py /path/to/test/folder --work-dir /tmp/tiga_test
    python test_run.py /path/to/test/folder --report-out report.md
    python test_run.py /path/to/test/folder --bm25-only   # skip embed (no Ollama needed)

The report is written to:
    <work-dir>/reports/test_report_<timestamp>.md

and also printed to stdout.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import platform
import re
import sqlite3
import sys
import time
import traceback
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Log capture — must be set up before importing tiga modules
# ---------------------------------------------------------------------------

class _LogCapture(logging.Handler):
    """Capture all log records into a list for report generation."""

    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)

    def by_level(self, level: int) -> list[logging.LogRecord]:
        return [r for r in self.records if r.levelno >= level]

    def warnings_and_above(self) -> list[logging.LogRecord]:
        return self.by_level(logging.WARNING)


_log_capture = _LogCapture()
_root_logger = logging.getLogger()
_root_logger.addHandler(_log_capture)
_root_logger.setLevel(logging.DEBUG)

# Console handler — INFO and above so the user sees progress live
_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(logging.Formatter("%(levelname)-8s %(name)s — %(message)s"))
_root_logger.addHandler(_console_handler)

log = logging.getLogger("tiga.test_run")


# ---------------------------------------------------------------------------
# Dependency check (pure stdlib — runs before any tiga import)
# ---------------------------------------------------------------------------

_REQUIRED_PACKAGES = [
    ("yaml",         "PyYAML",          True),
    ("lancedb",      "lancedb",         True),
    ("pyarrow",      "pyarrow",         True),
    ("pypdf",        "pypdf",           True),
    ("docx",         "python-docx",     True),
    ("pptx",         "python-pptx",     True),
    ("openpyxl",     "openpyxl",        False),
    ("pytesseract",  "pytesseract",     False),
    ("sentence_transformers", "sentence-transformers", False),
    ("fastapi",      "fastapi",         False),
    ("streamlit",    "streamlit",       False),
]


def _check_dependencies() -> dict[str, Any]:
    results: dict[str, Any] = {"ok": [], "missing_required": [], "missing_optional": []}
    for import_name, pip_name, required in _REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", "?")
            results["ok"].append((pip_name, ver))
        except ImportError:
            if required:
                results["missing_required"].append(pip_name)
            else:
                results["missing_optional"].append(pip_name)
    return results


# ---------------------------------------------------------------------------
# Ollama check (pure urllib — no tiga imports needed)
# ---------------------------------------------------------------------------

def _check_ollama(base_url: str = "http://localhost:11434") -> dict[str, Any]:
    import json
    import urllib.request

    result: dict[str, Any] = {
        "reachable": False,
        "models": [],
        "embed_model_present": False,
        "chat_model_present": False,
        "error": None,
    }
    try:
        with urllib.request.urlopen(base_url.rstrip("/") + "/api/tags", timeout=5) as resp:
            data = json.loads(resp.read())
            result["reachable"] = True
            result["models"] = [m["name"] for m in data.get("models", [])]
    except Exception as e:
        result["error"] = str(e)
    return result


# ---------------------------------------------------------------------------
# Minimal config.yaml writer
# ---------------------------------------------------------------------------

_CONFIG_TEMPLATE = """\
# TIGA test run — auto-generated config
work_dir: {work_dir}

index_roots:
  - "{test_folder}"

include_globs:
  - '**/*'
exclude_globs:
  - '**/.git/**'
  - '**/~$*'
  - '**/*.tmp'
  - '**/node_modules/**'
  - '**/__pycache__/**'

max_file_mb: 512

lane_rules:
  text_extractable_exts:
    - .pdf
    - .docx
    - .doc
    - .pptx
    - .ppt
    - .txt
    - .md
    - .xlsx
    - .xls
    - .msg
    - .eml
  metadata_only_exts:
    - .dwg
    - .rvt
    - .ifc
    - .skp
    - .jpg
    - .jpeg
    - .png
    - .gif
    - .bmp
    - .tif
    - .tiff
    - .mp4
    - .mov
    - .avi
    - .zip
    - .rar
    - .7z

ollama:
  base_url: http://localhost:11434
  embed_model: nomic-embed-text
  chat_model: mistral
  timeout_seconds: 60
  num_ctx: 8192
  embed_batch_size: 32
  embed_batch_sleep_s: 0.05

retrieval:
  top_k_default: 5
  hybrid_weight_bm25: 0.4
  hybrid_weight_vector: 0.6
  hybrid_alpha: 0.6
  reranker_enabled: false

pipeline:
  extract_workers: 2
  fingerprint_strategy: full

project_inference:
  enable: true
  confidence_threshold_unknown: 0.3

typology_inference:
  confidence_threshold_unknown: 0.3
"""


# ---------------------------------------------------------------------------
# Smoke query helpers
# ---------------------------------------------------------------------------

def _pick_smoke_queries(test_folder: Path, db_path: Path) -> list[str]:
    """
    Build 3–5 smoke queries from:
    1. Folder name keywords
    2. Most common words in chunk texts (from DB)
    3. Extension-based fallbacks
    """
    queries: list[str] = []

    # Keyword from folder name
    folder_words = re.sub(r"[^a-zA-Z0-9 ]", " ", test_folder.name).split()
    folder_words = [w for w in folder_words if len(w) > 3]
    if folder_words:
        queries.append(" ".join(folder_words[:3]))

    # Try to pull common words from DB chunks
    try:
        conn = sqlite3.connect(str(db_path))
        rows = conn.execute("SELECT text FROM chunks LIMIT 200").fetchall()
        conn.close()

        word_freq: dict[str, int] = {}
        stopwords = {
            "the", "and", "for", "are", "this", "that", "with",
            "from", "have", "been", "will", "not", "was", "its",
            "page", "date", "rev", "ref", "no", "to", "of", "in",
            "is", "it", "at", "as", "by", "an", "be", "or", "on",
        }
        for row in rows:
            for word in re.findall(r"[a-zA-Z]{4,}", (row[0] or "")):
                w = word.lower()
                if w not in stopwords:
                    word_freq[w] = word_freq.get(w, 0) + 1

        top_words = sorted(word_freq, key=lambda w: word_freq[w], reverse=True)[:10]
        if len(top_words) >= 2:
            queries.append(top_words[0])
            queries.append(f"{top_words[1]} {top_words[2]}" if len(top_words) >= 3 else top_words[1])
    except Exception:
        pass

    # Generic fallbacks
    fallbacks = ["specification", "drawing", "report", "design", "plan"]
    for fb in fallbacks:
        if len(queries) >= 4:
            break
        if fb not in " ".join(queries).lower():
            queries.append(fb)

    return queries[:4]


# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

class ReportBuilder:
    def __init__(self) -> None:
        self._buf = StringIO()

    def h1(self, text: str) -> None:
        self._buf.write(f"\n# {text}\n\n")

    def h2(self, text: str) -> None:
        self._buf.write(f"\n## {text}\n\n")

    def h3(self, text: str) -> None:
        self._buf.write(f"\n### {text}\n\n")

    def line(self, text: str = "") -> None:
        self._buf.write(text + "\n")

    def table(self, headers: list[str], rows: list[list[str]]) -> None:
        sep = "|".join(["---"] * len(headers))
        self._buf.write("| " + " | ".join(headers) + " |\n")
        self._buf.write("|" + sep + "|\n")
        for row in rows:
            self._buf.write("| " + " | ".join(str(c) for c in row) + " |\n")
        self._buf.write("\n")

    def code(self, text: str, lang: str = "") -> None:
        self._buf.write(f"```{lang}\n{text}\n```\n\n")

    def build(self) -> str:
        return self._buf.getvalue()


# ---------------------------------------------------------------------------
# Phase timing context
# ---------------------------------------------------------------------------

class _Timer:
    def __init__(self, label: str) -> None:
        self.label = label
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> "_Timer":
        self._start = time.time()
        log.info("▶ Starting: %s", self.label)
        return self

    def __exit__(self, *_: Any) -> None:
        self.elapsed = time.time() - self._start
        log.info("✓ Done: %s (%.1fs)", self.label, self.elapsed)


# ---------------------------------------------------------------------------
# DB analysis helpers
# ---------------------------------------------------------------------------

def _query_db(db_path: Path, sql: str, params: tuple = ()) -> list[dict]:
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        log.warning("DB query failed: %s — %s", sql[:60], e)
        return []


def _get_failed_files(db_path: Path) -> list[dict]:
    return _query_db(
        db_path,
        "SELECT file_name, extension, lane, status, error_code, error_detail "
        "FROM files WHERE status IN ('FAILED') ORDER BY error_code, file_name",
    )


def _get_status_counts(db_path: Path) -> dict[str, int]:
    rows = _query_db(
        db_path,
        "SELECT status, COUNT(*) as cnt FROM files GROUP BY status ORDER BY status",
    )
    return {r["status"]: r["cnt"] for r in rows}


def _get_lane_counts(db_path: Path) -> dict[str, int]:
    rows = _query_db(
        db_path,
        "SELECT lane, COUNT(*) as cnt FROM files GROUP BY lane ORDER BY lane",
    )
    return {r["lane"]: r["cnt"] for r in rows}


def _get_empty_text_files(db_path: Path) -> list[dict]:
    """TEXT_EXTRACTABLE files that yielded zero non-empty chunk text — likely scanned."""
    return _query_db(
        db_path,
        """SELECT f.file_name, f.extension, f.status, COUNT(c.chunk_id) as chunks,
                  SUM(CASE WHEN c.text = '' OR c.text IS NULL THEN 1 ELSE 0 END) as empty_chunks
           FROM files f
           JOIN chunks c ON c.file_id = f.file_id
           WHERE f.lane = 'TEXT_EXTRACTABLE'
           GROUP BY f.file_id
           HAVING empty_chunks = chunks AND chunks > 0
           ORDER BY f.file_name""",
    )


def _get_chunk_stats(db_path: Path) -> dict[str, Any]:
    rows = _query_db(
        db_path,
        "SELECT COUNT(*) as total, "
        "AVG(token_estimate) as avg_tokens, "
        "MAX(token_estimate) as max_tokens, "
        "SUM(CASE WHEN embedded=1 THEN 1 ELSE 0 END) as embedded, "
        "SUM(CASE WHEN text='' OR text IS NULL THEN 1 ELSE 0 END) as empty "
        "FROM chunks",
    )
    return rows[0] if rows else {}


def _get_ext_breakdown(db_path: Path) -> list[dict]:
    return _query_db(
        db_path,
        """SELECT extension, lane,
                  COUNT(*) as files,
                  SUM(CASE WHEN status='INDEXED' THEN 1 ELSE 0 END) as indexed,
                  SUM(CASE WHEN status='FAILED'  THEN 1 ELSE 0 END) as failed
           FROM files GROUP BY extension, lane ORDER BY files DESC""",
    )


# ---------------------------------------------------------------------------
# BM25 smoke query (pure SQLite — no Ollama needed)
# ---------------------------------------------------------------------------

def _run_bm25_query(db_path: Path, query: str, top_k: int = 5) -> list[dict]:
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        # Strip likely FTS special chars
        safe_q = re.sub(r'["\(\)\*\+\-]', " ", query).strip()
        if not safe_q:
            return []
        rows = conn.execute(
            """SELECT c.chunk_id, c.ref_value, c.text,
                      f.file_name, f.extension, f.project_id,
                      bm25(chunks_fts) AS score
               FROM chunks_fts
               JOIN chunks c ON c.rowid = chunks_fts.rowid
               JOIN files  f ON f.file_id = c.file_id
               WHERE chunks_fts MATCH ?
               ORDER BY score
               LIMIT ?""",
            (safe_q, top_k),
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        return [{"error": str(e)}]


# ---------------------------------------------------------------------------
# Main test runner
# ---------------------------------------------------------------------------

def run_test(
    test_folder: Path,
    work_dir: Path,
    report_out: Path | None,
    bm25_only: bool,
) -> int:
    """
    Run the full TIGA pipeline test against test_folder.
    Returns exit code: 0 = no critical errors, 1 = critical errors found.
    """
    started_at = datetime.now()
    rpt = ReportBuilder()
    timings: dict[str, float] = {}

    print(f"\n{'='*60}")
    print(f"  TIGA Test Run")
    print(f"  Folder : {test_folder}")
    print(f"  Work   : {work_dir}")
    print(f"  Started: {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # ── 0. Dependency check ──────────────────────────────────────────────────
    log.info("Checking dependencies...")
    dep_results = _check_dependencies()

    if dep_results["missing_required"]:
        log.error(
            "Missing REQUIRED packages: %s — install with: pip install %s",
            dep_results["missing_required"],
            " ".join(dep_results["missing_required"]),
        )
        print(f"\n[FATAL] Missing required packages: {dep_results['missing_required']}")
        print("Run: pip install " + " ".join(dep_results["missing_required"]))
        return 1

    # ── 1. Ollama check ──────────────────────────────────────────────────────
    log.info("Checking Ollama...")
    ollama_status = _check_ollama()
    embed_available = ollama_status["reachable"] and not bm25_only
    if bm25_only:
        log.info("--bm25-only flag set; skipping embed phase")
    elif not ollama_status["reachable"]:
        log.warning(
            "Ollama not reachable at localhost:11434 — "
            "embed phase will be skipped (BM25-only mode)"
        )

    # ── 2. Write test config ─────────────────────────────────────────────────
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "db").mkdir(exist_ok=True)
    (work_dir / "vectors").mkdir(exist_ok=True)
    (work_dir / "logs").mkdir(exist_ok=True)
    (work_dir / "reports").mkdir(exist_ok=True)

    config_path = work_dir / "config.yaml"
    config_path.write_text(
        _CONFIG_TEMPLATE.format(
            work_dir=str(work_dir).replace("\\", "/"),
            test_folder=str(test_folder).replace("\\", "/"),
        ),
        encoding="utf-8",
    )
    log.info("Wrote test config: %s", config_path)

    # Set env var BEFORE importing tiga modules so cfg singleton loads correctly
    os.environ["TIGA_WORK_DIR"] = str(work_dir)

    # ── 3. Import tiga modules (deferred until after env var is set) ─────────
    log.info("Loading TIGA modules...")
    try:
        # Force reload of config singleton with the test work dir
        import config as _config_mod
        importlib.reload(_config_mod)
        cfg = _config_mod.cfg

        from core.db import get_connection, get_stats
        from core.index import run_full_pipeline
        from core.extract import extract_chunks
    except Exception:
        log.error("Failed to import TIGA modules:\n%s", traceback.format_exc())
        return 1

    db_path = cfg.get_db_path()
    log.info("DB: %s", db_path)

    # ── 4. Open DB connection ────────────────────────────────────────────────
    try:
        conn = get_connection(db_path)
    except Exception:
        log.error("Failed to open DB:\n%s", traceback.format_exc())
        return 1

    # ── 5. Run full pipeline ─────────────────────────────────────────────────
    pipeline_error: str | None = None
    pipeline_stats: dict[str, int] = {}

    if embed_available:
        log.info("Running full pipeline (discover + extract + embed + FTS)...")
    else:
        log.info("Running pipeline without embed (discover + extract + FTS only)...")
        # Temporarily override config to disable embedding
        cfg._data = getattr(cfg, "_data", {})

    try:
        with _Timer("Full pipeline") as t:
            pipeline_stats = run_full_pipeline(conn, cfg_obj=cfg)
        timings["pipeline"] = t.elapsed
    except Exception:
        pipeline_error = traceback.format_exc()
        log.error("Pipeline raised an exception:\n%s", pipeline_error)

    # ── 6. Collect DB facts ──────────────────────────────────────────────────
    status_counts = _get_status_counts(db_path)
    lane_counts   = _get_lane_counts(db_path)
    failed_files  = _get_failed_files(db_path)
    empty_files   = _get_empty_text_files(db_path)
    chunk_stats   = _get_chunk_stats(db_path)
    ext_breakdown = _get_ext_breakdown(db_path)

    # ── 7. Run smoke queries (BM25 always; vector if Ollama available) ───────
    log.info("Running smoke queries...")
    smoke_queries = _pick_smoke_queries(test_folder, db_path)
    smoke_results: list[dict[str, Any]] = []

    for q in smoke_queries:
        with _Timer(f"BM25 query: '{q}'") as t:
            hits = _run_bm25_query(db_path, q)
        timings[f"query_{q[:20]}"] = t.elapsed
        smoke_results.append({"query": q, "hits": hits, "elapsed": t.elapsed})

    # ── 8. Collect warnings from log ─────────────────────────────────────────
    log_warnings = _log_capture.warnings_and_above()

    # ── 9. Build report ──────────────────────────────────────────────────────
    finished_at = datetime.now()
    duration_s  = (finished_at - started_at).total_seconds()

    rpt.h1("TIGA Test Run Report")
    rpt.line(f"**Date:** {started_at.strftime('%Y-%m-%d %H:%M:%S')}")
    rpt.line(f"**Duration:** {duration_s:.1f}s")
    rpt.line(f"**Test folder:** `{test_folder}`")
    rpt.line(f"**Work dir:** `{work_dir}`")
    rpt.line(f"**Platform:** {platform.system()} {platform.release()} / Python {sys.version.split()[0]}")
    rpt.line()

    # Overall verdict
    critical_issues = (
        len(dep_results["missing_required"]) > 0
        or pipeline_error is not None
        or status_counts.get("FAILED", 0) > status_counts.get("INDEXED", 0)
    )
    verdict = "FAIL" if critical_issues else ("PASS" if not failed_files else "PASS WITH WARNINGS")
    rpt.line(f"**Overall: {verdict}**")
    rpt.line()

    # ── Section: Environment ─────────────────────────────────────────────────
    rpt.h2("1. Environment")
    rpt.h3("Dependencies")
    dep_rows = [(name, ver, "✅") for name, ver in dep_results["ok"]]
    dep_rows += [(name, "NOT INSTALLED", "❌ REQUIRED") for name in dep_results["missing_required"]]
    dep_rows += [(name, "NOT INSTALLED", "⚠️ optional") for name in dep_results["missing_optional"]]
    rpt.table(["Package", "Version", "Status"], dep_rows)

    rpt.h3("Ollama")
    if ollama_status["reachable"]:
        rpt.line(f"✅ Reachable at `http://localhost:11434`")
        rpt.line(f"**Models loaded:** {', '.join(ollama_status['models']) or 'none'}")
        rpt.line()
        if not any("nomic" in m for m in ollama_status["models"]):
            rpt.line("⚠️  `nomic-embed-text` not found. Run: `ollama pull nomic-embed-text`")
        if not any("mistral" in m for m in ollama_status["models"]):
            rpt.line("⚠️  `mistral` not found. Run: `ollama pull mistral`")
    else:
        rpt.line(f"❌ Not reachable — error: `{ollama_status['error']}`")
        rpt.line("Embed phase was skipped. Only BM25 (FTS) search is available.")
        rpt.line("To fix: install Ollama from https://ollama.com and run `ollama serve`")
    rpt.line()

    # ── Section: Pipeline Results ────────────────────────────────────────────
    rpt.h2("2. Pipeline Results")

    if pipeline_error:
        rpt.h3("Pipeline Exception")
        rpt.line("❌ The pipeline raised an unhandled exception:")
        rpt.code(pipeline_error)

    rpt.h3("File Status Summary")
    total_files = sum(status_counts.values())
    status_rows = [[s, str(c), f"{100*c/total_files:.0f}%" if total_files else "0%"]
                   for s, c in sorted(status_counts.items())]
    status_rows.append(["**TOTAL**", str(total_files), "100%"])
    rpt.table(["Status", "Count", "%"], status_rows)

    rpt.h3("File Lane Breakdown")
    lane_rows = [[l, str(c)] for l, c in sorted(lane_counts.items())]
    rpt.table(["Lane", "Count"], lane_rows)

    rpt.h3("Extension Breakdown")
    ext_rows = [
        [r["extension"] or "(none)", r["lane"] or "?",
         str(r["files"]), str(r["indexed"]), str(r["failed"])]
        for r in ext_breakdown
    ]
    rpt.table(["Extension", "Lane", "Files", "Indexed", "Failed"], ext_rows)

    rpt.h3("Chunk Statistics")
    if chunk_stats:
        rpt.line(f"- **Total chunks:** {chunk_stats.get('total', 0)}")
        rpt.line(f"- **Embedded:** {chunk_stats.get('embedded', 0)}")
        rpt.line(f"- **Empty text:** {chunk_stats.get('empty', 0)}")
        avg_tok = chunk_stats.get("avg_tokens")
        max_tok = chunk_stats.get("max_tokens")
        rpt.line(f"- **Avg token estimate:** {round(avg_tok) if avg_tok else 'N/A'}")
        rpt.line(f"- **Max token estimate:** {max_tok if max_tok else 'N/A'}")
    rpt.line()

    # ── Section: Failures ────────────────────────────────────────────────────
    rpt.h2("3. Failed Files")

    if not failed_files:
        rpt.line("✅ No FAILED files.")
    else:
        rpt.line(f"❌ **{len(failed_files)} file(s) failed** during pipeline processing.\n")
        fail_rows = [
            [f["file_name"], f["extension"], f["lane"] or "?",
             f["error_code"] or "?", (f["error_detail"] or "")[:80]]
            for f in failed_files
        ]
        rpt.table(["File", "Ext", "Lane", "Error Code", "Detail"], fail_rows)
    rpt.line()

    # ── Section: Scanned / Empty PDFs ────────────────────────────────────────
    rpt.h2("4. Likely Scanned Documents (No Text Extracted)")

    if not empty_files:
        rpt.line("✅ No text-extractable files returned zero text.")
    else:
        rpt.line(
            f"⚠️  **{len(empty_files)} file(s)** appear to be scanned (zero text extracted "
            f"from all chunks). These are invisible to BM25 and vector search.\n"
        )
        empty_rows = [
            [f["file_name"], f["extension"], str(f["chunks"]), str(f["empty_chunks"]), f["status"]]
            for f in empty_files
        ]
        rpt.table(["File", "Ext", "Chunks", "Empty Chunks", "Status"], empty_rows)
        rpt.line()
        rpt.line("**Fix:** Enable OCR in `config.yaml`:")
        rpt.code("ocr:\n  enabled: true\n  tesseract_cmd: tesseract", lang="yaml")
        rpt.line("Then re-run index. Requires Tesseract binary: https://github.com/tesseract-ocr/tesseract")

    # ── Section: Smoke Queries ───────────────────────────────────────────────
    rpt.h2("5. Smoke Queries (BM25)")
    rpt.line("These queries test that indexed content is searchable. Results come from SQLite FTS5 (BM25).\n")

    for sr in smoke_results:
        q = sr["query"]
        hits = sr["hits"]
        elapsed = sr["elapsed"]

        rpt.h3(f'Query: "{q}"')
        rpt.line(f"*Elapsed: {elapsed*1000:.0f}ms — {len(hits)} result(s)*\n")

        if not hits:
            rpt.line("❌ No results. This query returned nothing from the index.")
            rpt.line(
                "Possible causes: nothing indexed yet, or the query terms don't appear "
                "in any chunk text.\n"
            )
        elif hits[0].get("error"):
            rpt.line(f"❌ Query error: `{hits[0]['error']}`\n")
        else:
            result_rows = []
            for i, h in enumerate(hits[:5], 1):
                snippet = (h.get("text") or "")[:100].replace("\n", " ")
                result_rows.append([
                    str(i),
                    h.get("file_name", "?"),
                    h.get("ref_value", "?"),
                    f"{h.get('score', 0):.3f}",
                    snippet,
                ])
            rpt.table(["#", "File", "Chunk", "Score", "Snippet (100 chars)"], result_rows)

    # ── Section: Log Warnings ────────────────────────────────────────────────
    rpt.h2("6. Warnings and Errors from Log")

    if not log_warnings:
        rpt.line("✅ No warnings or errors logged.")
    else:
        rpt.line(f"**{len(log_warnings)} warning(s)/error(s) captured:**\n")
        for rec in log_warnings[:50]:  # cap at 50
            level_icon = "❌" if rec.levelno >= logging.ERROR else "⚠️"
            rpt.line(
                f"{level_icon} `[{rec.levelname}] {rec.name}` — "
                f"{rec.getMessage()[:200]}"
            )
        if len(log_warnings) > 50:
            rpt.line(f"\n*... and {len(log_warnings) - 50} more (see full log).*")
    rpt.line()

    # ── Section: Recommendations ─────────────────────────────────────────────
    rpt.h2("7. Recommendations")

    rec_count = 0

    if not ollama_status["reachable"]:
        rec_count += 1
        rpt.line(
            f"**{rec_count}. Install and start Ollama** to enable vector search.\n"
            "   ```\n   ollama serve\n   ollama pull nomic-embed-text\n   ollama pull mistral\n   ```"
        )

    if dep_results["missing_optional"]:
        rec_count += 1
        rpt.line(
            f"**{rec_count}. Install optional packages** for full functionality:\n"
            f"   ```\n   pip install {' '.join(dep_results['missing_optional'])}\n   ```"
        )

    if empty_files:
        rec_count += 1
        rpt.line(
            f"**{rec_count}. Enable OCR** for {len(empty_files)} scanned document(s). "
            "Without OCR these files are invisible to search. "
            "Add to config.yaml: `ocr: {enabled: true}`"
        )

    if failed_files:
        error_codes = list({f["error_code"] for f in failed_files if f.get("error_code")})
        rec_count += 1
        rpt.line(
            f"**{rec_count}. Investigate {len(failed_files)} failed file(s).**\n"
            f"   Error codes seen: `{'`, `'.join(error_codes)}`\n"
            "   Check error_detail column in `files` table for specifics."
        )

    if chunk_stats.get("total", 0) == 0:
        rec_count += 1
        rpt.line(
            f"**{rec_count}. No chunks were created.** "
            "This usually means the folder had no text-extractable files, "
            "or all files failed extraction. Check the Extension Breakdown section."
        )
    elif chunk_stats.get("embedded", 0) == 0 and ollama_status["reachable"]:
        rec_count += 1
        rpt.line(
            f"**{rec_count}. No chunks were embedded** despite Ollama being available. "
            "Check that `nomic-embed-text` model is pulled: `ollama pull nomic-embed-text`"
        )

    all_queries_empty = all(not sr["hits"] or sr["hits"][0].get("error") for sr in smoke_results)
    if all_queries_empty and chunk_stats.get("total", 0) > 0:
        rec_count += 1
        rpt.line(
            f"**{rec_count}. All smoke queries returned zero results** despite chunks existing. "
            "This may indicate an FTS5 trigger failure. Try running `python tiga.py rebuild`."
        )

    if chunk_stats.get("max_tokens", 0) and chunk_stats["max_tokens"] > 4000:
        rec_count += 1
        rpt.line(
            f"**{rec_count}. Large chunks detected** (max {chunk_stats['max_tokens']} tokens). "
            "Consider reviewing chunking — very large chunks reduce retrieval precision."
        )

    if rec_count == 0:
        rpt.line("✅ No recommendations — everything looks healthy.")

    # ── Section: Timings ─────────────────────────────────────────────────────
    rpt.h2("8. Timings")
    timing_rows = [[label, f"{secs:.2f}s"] for label, secs in timings.items()]
    timing_rows.append(["**Total run**", f"**{duration_s:.1f}s**"])
    rpt.table(["Phase", "Duration"], timing_rows)

    # ── Write report ─────────────────────────────────────────────────────────
    report_text = rpt.build()

    if report_out is None:
        ts = started_at.strftime("%Y%m%d_%H%M%S")
        report_out = work_dir / "reports" / f"test_report_{ts}.md"

    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(report_text, encoding="utf-8")

    # Print a compact summary to console
    print("\n" + "="*60)
    print(f"  RESULT: {verdict}")
    print("="*60)
    print(f"  Files  : {total_files} total | "
          f"{status_counts.get('INDEXED', 0)} indexed | "
          f"{status_counts.get('FAILED', 0)} failed")
    print(f"  Chunks : {chunk_stats.get('total', 0)} total | "
          f"{chunk_stats.get('embedded', 0)} embedded")
    print(f"  Ollama : {'✓ reachable' if ollama_status['reachable'] else '✗ not reachable'}")
    print(f"  Report : {report_out}")
    print("="*60 + "\n")

    if critical_issues:
        print("PASTE THE REPORT ABOVE BACK TO CLAUDE FOR DEBUGGING.\n")

    return 0 if not critical_issues else 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="TIGA automated test runner — validates pipeline against one folder."
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Path to the test folder to index.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Work directory for test DB/vectors (default: ./tiga_test_work next to this script).",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Where to write the markdown report (default: <work-dir>/reports/test_report_<ts>.md).",
    )
    parser.add_argument(
        "--bm25-only",
        action="store_true",
        help="Skip the embed phase entirely (useful if Ollama is not set up yet).",
    )
    args = parser.parse_args()

    test_folder = args.folder.resolve()
    if not test_folder.exists():
        print(f"[ERROR] Test folder does not exist: {test_folder}")
        sys.exit(1)
    if not test_folder.is_dir():
        print(f"[ERROR] Not a directory: {test_folder}")
        sys.exit(1)

    work_dir = (
        args.work_dir.resolve()
        if args.work_dir
        else Path(__file__).resolve().parent / "tiga_test_work"
    )

    exit_code = run_test(
        test_folder=test_folder,
        work_dir=work_dir,
        report_out=args.report_out,
        bm25_only=args.bm25_only,
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
