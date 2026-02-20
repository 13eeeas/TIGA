"""
run_tianmu_index.py — Standalone Tianmu indexing + RAG test script.

Runs the full 4-phase pipeline on Z:/186 - Tianmu only,
then tests 6 retrieval queries and reports pass/fail.

Usage:
    python run_tianmu_index.py [--skip-index]
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── path setup ──────────────────────────────────────────────────────────────
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

WORK_DIR = HERE / "tiga_work_tianmu"
WORK_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TIGA_WORK_DIR"] = str(WORK_DIR)

# ── logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(WORK_DIR / "index.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run_tianmu")

# ── imports (after path + env set) ──────────────────────────────────────────
from config import load_config
from core.db import get_connection
from core.index import run_full_pipeline
from core.query import search


# ── RAG test queries ─────────────────────────────────────────────────────────

TEST_QUERIES = [
    {
        "id":       "typology",
        "question": "What is the project type or typology for Project 186 Tianmu?",
        "keywords": ["residential", "housing", "commercial", "mixed", "tianmu", "186"],
        "desc":     "project typology / building type",
    },
    {
        "id":       "brief",
        "question": "What was the project brief for Tianmu? What were the client requirements?",
        "keywords": ["brief", "requirement", "client", "program", "scope", "design"],
        "desc":     "project brief / client requirements",
    },
    {
        "id":       "tender",
        "question": "Where are the tender set drawings or contract documentation for Tianmu?",
        "keywords": ["tender", "contract", "documentation", "drawing", "set", "CD"],
        "desc":     "tender set / contract drawings location",
    },
    {
        "id":       "presentation",
        "question": "Where are the client presentation slides for Tianmu project?",
        "keywords": ["presentation", "client", "slide", "ppt", "pptx"],
        "desc":     "client presentation slides",
    },
    {
        "id":       "renders",
        "question": "Where are the render images or visualisations for Tianmu?",
        "keywords": ["render", "visuali", "image", "3d", "perspective", "view"],
        "desc":     "renders / visualisations",
    },
    {
        "id":       "proposals",
        "question": "Where are the early design proposals or schematic design documents for Tianmu?",
        "keywords": ["schematic", "SD", "early", "proposal", "concept", "design"],
        "desc":     "early design proposals",
    },
]


def run_index_phase(cfg, conn) -> dict:
    logger.info("=" * 60)
    logger.info("PHASE: Full pipeline — discover -> extract -> embed -> FTS")
    logger.info("=" * 60)
    t0 = time.perf_counter()
    stats = run_full_pipeline(conn, cfg_obj=cfg)
    elapsed = round(time.perf_counter() - t0, 1)
    print(f"\n[INDEX] Done in {elapsed}s")
    print(f"  discovered:      {stats.get('discovered', 0)} new files")
    print(f"  extracted:       {stats.get('files_extracted', 0)} files  ({stats.get('chunks_new', 0)} chunks)")
    print(f"  extract_failed:  {stats.get('files_extract_failed', 0)} files")
    print(f"  embedded:        {stats.get('files_embedded', 0)} files  ({stats.get('chunks_embedded', 0)} chunks, {stats.get('chunks_skipped', 0)} skipped)")
    print(f"  indexed (INDEXED): {stats.get('files_indexed', 0)} files")
    return stats


def run_retrieval_tests(cfg, conn) -> tuple[int, int]:
    print("\n" + "=" * 60)
    print("RAG RETRIEVAL TESTS")
    print("=" * 60)

    passed = 0
    total  = len(TEST_QUERIES)

    for q in TEST_QUERIES:
        print(f"\n[{q['id'].upper()}] {q['desc']}")
        print(f"  Q: {q['question']}")

        t0 = time.perf_counter()
        results = search(q["question"], top_k=5, conn=conn, cfg_obj=cfg)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        if not results:
            print(f"  FAIL — no results returned  ({elapsed_ms} ms)")
            continue

        # Check if any keyword appears in the top-5 results
        hit = False
        for kw in q["keywords"]:
            for r in results:
                text = (
                    r.get("rel_path", "") + " " +
                    r.get("snippet", "") + " " +
                    r.get("file_name", "") + " " +
                    r.get("project_id", "")
                ).lower()
                if kw.lower() in text:
                    hit = True
                    break
            if hit:
                break

        status = "PASS" if hit else "FAIL"
        if hit:
            passed += 1

        print(f"  {status}  ({elapsed_ms} ms)  — top results:")
        for i, r in enumerate(results[:3], 1):
            print(f"    [{i}] {r['citation']}  score={r['final_score']:.3f}")
            if r.get("snippet"):
                print(f"         {r['snippet'][:120]}")

    print(f"\n{'=' * 60}")
    print(f"RESULT: {passed}/{total} queries passed")
    pct = round(100 * passed / total)
    print(f"ACCURACY: {pct}%  ({'PASS' if pct >= 95 else 'NEEDS WORK'})")
    return passed, total


def main():
    parser = argparse.ArgumentParser(description="Tianmu index + RAG test")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip indexing, just run retrieval tests")
    args = parser.parse_args()

    cfg = load_config(work_dir=WORK_DIR)
    cfg.ensure_dirs()
    conn = get_connection(cfg.get_db_path())

    try:
        if not args.skip_index:
            run_index_phase(cfg, conn)

        passed, total = run_retrieval_tests(cfg, conn)
        sys.exit(0 if passed >= int(total * 0.95) else 1)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
