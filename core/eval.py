"""
core/eval.py — Simple evaluation harness for search quality.

Run a set of test queries and report top-k results.
Not part of the main pipeline — used for offline testing.

Usage:
    python -m core.eval
"""

from __future__ import annotations

import json
import time
from typing import Any

from config import cfg
from core.query import search


# ---------------------------------------------------------------------------
# Default test queries (architecture-domain examples)
# ---------------------------------------------------------------------------

DEFAULT_QUERIES = [
    "site analysis report",
    "structural drawings",
    "client brief competition",
    "sustainability specifications",
    "floor plan submissions",
]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_eval(
    queries: list[str] | None = None,
    top_k: int | None = None,
    verbose: bool = True,
) -> list[dict[str, Any]]:
    """
    Run eval queries and return results with timing.
    """
    qs = queries or DEFAULT_QUERIES
    k = top_k or cfg.top_k
    report: list[dict[str, Any]] = []

    for query in qs:
        t0 = time.perf_counter()
        results = search(query, top_k=k)
        elapsed = round(time.perf_counter() - t0, 3)

        entry = {
            "query": query,
            "elapsed_s": elapsed,
            "results": [
                {
                    "rank": i + 1,
                    "file_name": r["file_name"],
                    "project": r["project"],
                    "typology": r["typology"],
                    "score": r["combined_score"],
                }
                for i, r in enumerate(results)
            ],
        }
        report.append(entry)

        if verbose:
            print(f"\nQuery: {query!r}  ({elapsed}s)")
            for r in entry["results"]:
                print(
                    f"  [{r['rank']}] {r['file_name']}  "
                    f"project={r['project']}  score={r['score']:.3f}"
                )

    return report


if __name__ == "__main__":
    results = run_eval()
    print("\n\nFull report:")
    print(json.dumps(results, indent=2))
