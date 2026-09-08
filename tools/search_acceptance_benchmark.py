"""Run a retrieval-only acceptance benchmark against a live TIGA portal.

Each YAML entry has ``query`` and either ``expected_paths`` (all must be
present) or ``expected_any_paths`` (one relevant result is enough).  The latter
is important for natural-language queries that do not specify a revision.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import yaml


def _query(endpoint: str, query: str, top_k: int) -> dict:
    body = json.dumps({"query": query, "compose": False, "top_k": top_k}).encode()
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/api/query",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        default="tests/fixtures/architect_query_benchmark_nus_biz3.yaml",
    )
    parser.add_argument("--endpoint", default="http://127.0.0.1:7862")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--minimum", type=float, default=0.90)
    args = parser.parse_args()

    entries = yaml.safe_load(Path(args.fixture).read_text(encoding="utf-8")) or []
    outcomes: list[dict] = []
    for entry in entries:
        started = time.perf_counter()
        response = _query(args.endpoint, entry["query"], args.top_k)
        results = response.get("results", [])
        haystack = "\n".join(
            f"{result.get('title', '')}\n{result.get('rel_path', '')}".lower()
            for result in results
        )
        all_targets = entry.get("expected_paths")
        any_targets = entry.get("expected_any_paths")
        hit = (
            all(target.lower() in haystack for target in all_targets)
            if all_targets else any(target.lower() in haystack for target in any_targets or [])
        )
        outcomes.append({
            "query": entry["query"],
            "hit": hit,
            "mode": response.get("mode"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 1),
            "top_titles": [result.get("title", "") for result in results[:3]],
        })

    passed = sum(1 for outcome in outcomes if outcome["hit"])
    rate = passed / len(outcomes) if outcomes else 0.0
    print(json.dumps({"passed": passed, "total": len(outcomes), "rate": rate,
                      "minimum": args.minimum, "outcomes": outcomes}, indent=2))
    return 0 if rate >= args.minimum else 1


if __name__ == "__main__":
    raise SystemExit(main())
