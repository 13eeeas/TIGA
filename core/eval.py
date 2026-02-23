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
_ROUTING_EVAL_FILENAME = "eval_questions.json"


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
    print(f"\n{'-' * 52}")
    print(f"  Queries:          {n}")
    print(f"  Top-5 recall:     {top5_recall * 100:.0f}%")
    print(f"  Citation valid:   {citation_valid_pct:.0f}%  [{gate}]")
    print(f"  Latency p50/p95:  {p50:.0f} / {p95:.0f} ms")
    print(f"  Report:           {report_path}")
    print(f"{'-' * 52}")

    if exit_code == 1:
        print(
            f"\n  GATE FAIL: {invalid_count} invalid citation(s) detected.",
            file=sys.stderr,
        )

    return exit_code


# ---------------------------------------------------------------------------
# Routing eval (Chunk 7) — fast, no Ollama required
# ---------------------------------------------------------------------------

def _load_routing_fixture() -> list[dict[str, Any]]:
    """Load eval_questions.json from tests/ directory."""
    # Try tests/ directory first
    candidates = [
        Path(__file__).parent.parent / "tests" / _ROUTING_EVAL_FILENAME,
        cfg.work_dir / "fixtures" / _ROUTING_EVAL_FILENAME,
    ]
    for p in candidates:
        if p.exists():
            try:
                import re as _re
                text = p.read_text(encoding="utf-8")
                # Strip JS-style // comments before JSON parsing
                text = _re.sub(r'//[^\n]*', '', text)
                return json.loads(text)
            except Exception as e:
                logger.warning("Failed to load routing fixture %s: %s", p, e)
    return []


def run_routing_eval(verbose: bool = True) -> int:
    """
    Run routing-only eval (no search, no Ollama).
    Tests that classify() returns the expected mode for each question.

    Returns:
        exit code: 0 = >=85% accuracy, 1 = below threshold
    """
    from core.router import QueryRouter

    router = QueryRouter()
    fixture = _load_routing_fixture()

    if not fixture:
        print(
            "No routing eval fixture found.\n"
            "Expected: tests/eval_questions.json",
            file=sys.stderr,
        )
        return 0

    total = len(fixture)
    correct = 0
    failures: list[dict[str, Any]] = []

    for item in fixture:
        q = item.get("q", "")
        expected_mode = item.get("mode", "")
        if not q or not expected_mode:
            continue

        result = router.classify(q)
        actual_mode = result.mode
        passed = actual_mode == expected_mode

        if passed:
            correct += 1
        else:
            failures.append({
                "q":        q,
                "expected": expected_mode,
                "actual":   actual_mode,
                "tags":     result.expanded_query.concept_tags if result.expanded_query else [],
                "conf":     round(result.confidence, 2),
                "reason":   result.reason,
            })

        if verbose:
            marker = "PASS" if passed else "FAIL"
            tags_str = (
                ",".join(result.expanded_query.concept_tags)
                if result.expanded_query and result.expanded_query.concept_tags
                else "none"
            )
            print(f"[{marker}] {q!r}")
            if not passed:
                print(f"       Expected: {expected_mode}  Got: {actual_mode}  Tags: [{tags_str}]")

    accuracy = correct / total if total else 0.0
    threshold = 0.85
    exit_code = 0 if accuracy >= threshold else 1
    gate = "PASS" if exit_code == 0 else "FAIL"

    print(f"\n{'-' * 52}")
    print(f"  Questions:   {total}")
    print(f"  Correct:     {correct}")
    print(f"  Accuracy:    {accuracy * 100:.1f}%  [{gate}]  (threshold: {threshold * 100:.0f}%)")
    print(f"  Failures:    {len(failures)}")
    print(f"{'-' * 52}")

    if failures:
        print("\nFailed questions:")
        for f in failures:
            print(f"  [{f['expected']} -> {f['actual']}] {f['q']!r}")
            if f['tags']:
                print(f"    Concept tags: {f['tags']}")

    # Save report
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_dir = cfg.get_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"routing_eval_{ts}.json"
    report_path.write_text(
        json.dumps({
            "ts": ts, "total": total, "correct": correct,
            "accuracy": round(accuracy, 4), "threshold": threshold,
            "failures": failures,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"  Report: {report_path}")

    return exit_code


# ---------------------------------------------------------------------------
# Stress eval — run all N questions end-to-end, export HTML report
# ---------------------------------------------------------------------------

_STRESS_FIXTURE = "stress_test_186_tianmu.json"

_CATEGORY_LABELS = {
    "A_identity":    "A — Project Identity",
    "B_metrics":     "B — Scale & Metrics",
    "C_stakeholders":"C — Stakeholders",
    "D_file_location":"D — File Location",
    "E_design_intent":"E — Design Intent",
    "F_technical":   "F — Technical Specs",
    "G_compliance":  "G — Compliance & Waivers",
    "H_meetings":    "H — Meetings & History",
    "I_cross_project":"I — Cross-Project",
    "Z_edge_cases":  "Z — Edge Cases",
}

_MODE_COLORS = {
    "structured":   "#3b82f6",   # blue
    "file_locator": "#10b981",   # green
    "semantic":     "#8b5cf6",   # purple
    "cross_project":"#f59e0b",   # amber
}


def _load_stress_fixture() -> list[dict[str, Any]]:
    """Load stress_test_186_tianmu.json from tests/ directory."""
    candidates = [
        Path(__file__).parent.parent / "tests" / _STRESS_FIXTURE,
        cfg.work_dir / "fixtures" / _STRESS_FIXTURE,
    ]
    for p in candidates:
        if p.exists():
            try:
                import re as _re
                text = p.read_text(encoding="utf-8")
                text = _re.sub(r'//[^\n]*', '', text)  # strip JS comments
                return json.loads(text)
            except Exception as e:
                logger.warning("Failed to load stress fixture %s: %s", p, e)
    return []


def _run_one_question(item: dict[str, Any], router, conn, cfg_obj) -> dict[str, Any]:
    """
    Run a single stress-test question through the full pipeline.
    Returns a result dict with routing info + raw results from the executor.
    Never raises.
    """
    from core.query import (
        execute_structured_query, execute_file_locator_query,
        execute_cross_project_query, search,
    )

    q = item.get("q", "")
    expected_mode = item.get("expected_mode", "")
    t0 = time.perf_counter()

    try:
        route = router.classify(q)
        actual_mode = route.mode
        routing_ok = actual_mode == expected_mode

        concept_tags = (
            route.expanded_query.concept_tags if route.expanded_query else []
        )
        filters_used = route.filters or {}

        # Run the appropriate executor
        if actual_mode == "structured":
            exec_result = execute_structured_query(route, conn=conn, cfg_obj=cfg_obj)
            answer_text = exec_result.get("answer_text", "")
            files_returned: list[dict] = []
            snippets: list[dict] = []

        elif actual_mode == "file_locator":
            exec_result = execute_file_locator_query(route, conn=conn, cfg_obj=cfg_obj)
            answer_text = exec_result.get("answer_text", "")
            files_returned = exec_result.get("files", [])
            snippets = []

        elif actual_mode == "cross_project":
            exec_result = execute_cross_project_query(route, conn=conn, cfg_obj=cfg_obj)
            answer_text = exec_result.get("answer_text", "")
            files_returned = []
            snippets = []

        else:  # semantic
            expanded_terms = (
                route.expanded_query.expanded_terms if route.expanded_query else []
            )
            sr = search(q, top_k=8, filters=filters_used or None,
                        cfg_obj=cfg_obj, conn=conn, expanded_terms=expanded_terms)
            answer_text = f"{len(sr)} chunk(s) retrieved"
            files_returned = []
            snippets = [
                {
                    "rel_path":     r.get("rel_path", ""),
                    "snippet":      r.get("snippet", "")[:200],
                    "final_score":  r.get("final_score", 0.0),
                    "bm25_score":   r.get("bm25_score", 0.0),
                    "vector_score": r.get("vector_score", 0.0),
                    "citation":     r.get("citation", ""),
                }
                for r in sr
            ]

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

        return {
            "id":            item.get("id", 0),
            "q":             q,
            "category":      item.get("category", ""),
            "expected_mode": expected_mode,
            "actual_mode":   actual_mode,
            "routing_ok":    routing_ok,
            "confidence":    round(route.confidence, 2),
            "concept_tags":  concept_tags,
            "filters":       filters_used,
            "project_code":  route.project_code,
            "answer_text":   answer_text,
            "files":         files_returned[:20],  # cap for report size
            "snippets":      snippets[:8],
            "expected_hint": item.get("expected_hint", ""),
            "known_failure_risk": item.get("known_failure_risk", ""),
            "elapsed_ms":    elapsed_ms,
            "error":         None,
        }

    except Exception as e:
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)
        logger.warning("Stress eval error for q=%r: %s", q, e)
        return {
            "id":            item.get("id", 0),
            "q":             q,
            "category":      item.get("category", ""),
            "expected_mode": expected_mode,
            "actual_mode":   "error",
            "routing_ok":    False,
            "confidence":    0.0,
            "concept_tags":  [],
            "filters":       {},
            "project_code":  None,
            "answer_text":   "",
            "files":         [],
            "snippets":      [],
            "expected_hint": item.get("expected_hint", ""),
            "known_failure_risk": item.get("known_failure_risk", ""),
            "elapsed_ms":    elapsed_ms,
            "error":         str(e),
        }


def _generate_html_report(results: list[dict], ts: str) -> str:
    """Build a self-contained HTML report from stress eval results."""
    routing_pass = sum(1 for r in results if r["routing_ok"])
    total = len(results)
    routing_pct = routing_pass / total * 100 if total else 0

    # Per-category stats
    cat_stats: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        s = cat_stats.setdefault(cat, {"total": 0, "routing_ok": 0, "has_results": 0})
        s["total"] += 1
        if r["routing_ok"]:
            s["routing_ok"] += 1
        if r["files"] or r["snippets"]:
            s["has_results"] += 1

    # Build category summary rows
    cat_rows = ""
    for cat, s in cat_stats.items():
        label = _CATEGORY_LABELS.get(cat, cat)
        pct = s["routing_ok"] / s["total"] * 100 if s["total"] else 0
        bar_color = "#10b981" if pct >= 80 else "#f59e0b" if pct >= 60 else "#ef4444"
        cat_rows += f"""
        <tr>
          <td>{label}</td>
          <td>{s["routing_ok"]}/{s["total"]}</td>
          <td>
            <div style="background:#e5e7eb;border-radius:4px;height:16px;width:100%;min-width:80px">
              <div style="background:{bar_color};border-radius:4px;height:16px;width:{pct:.0f}%"></div>
            </div>
          </td>
          <td>{pct:.0f}%</td>
          <td>{s["has_results"]}/{s["total"]} returned results</td>
        </tr>"""

    # Build per-question rows
    q_rows = ""
    prev_cat = None
    for r in results:
        cat = r["category"]
        if cat != prev_cat:
            label = _CATEGORY_LABELS.get(cat, cat)
            q_rows += f"""
        <tr style="background:#1e293b">
          <td colspan="99" style="padding:10px 8px;font-weight:700;font-size:13px;
              color:#94a3b8;letter-spacing:.05em;text-transform:uppercase">{label}</td>
        </tr>"""
            prev_cat = cat

        route_bg  = "#052e16" if r["routing_ok"] else "#3f0000"
        route_txt = "#4ade80" if r["routing_ok"] else "#f87171"
        route_label = "OK" if r["routing_ok"] else "WRONG"
        exp_color = _MODE_COLORS.get(r["expected_mode"], "#94a3b8")
        act_color = _MODE_COLORS.get(r["actual_mode"],   "#94a3b8")
        risk_badge = ""
        if r.get("known_failure_risk"):
            risk_badge = f' <span style="background:#7c3aed;color:#fff;font-size:10px;padding:1px 5px;border-radius:3px">{r["known_failure_risk"]}</span>'

        tags_html = ""
        if r.get("concept_tags"):
            tags_html = "  ".join(
                f'<span style="background:#374151;color:#d1d5db;font-size:10px;'
                f'padding:1px 5px;border-radius:3px">{t}</span>'
                for t in r["concept_tags"]
            )

        filters_str = ""
        if r.get("filters"):
            filters_str = " | ".join(f"{k}={v}" for k, v in r["filters"].items())

        # Files table
        files_html = ""
        if r["files"]:
            file_rows = "".join(
                f'<tr><td style="color:#93c5fd;word-break:break-all">{f.get("file_path","")}</td>'
                f'<td>{f.get("content_type","")}</td>'
                f'<td>{f.get("folder_stage","")}</td>'
                f'<td>{f.get("file_date","")}</td></tr>'
                for f in r["files"][:15]
            )
            files_html = f"""
            <table style="width:100%;border-collapse:collapse;font-size:11px;margin-top:6px">
              <thead><tr style="color:#64748b">
                <th style="text-align:left;padding:2px 6px">File path</th>
                <th>Type</th><th>Stage</th><th>Date</th>
              </tr></thead>
              <tbody>{file_rows}</tbody>
            </table>"""
            if len(r["files"]) == 20:
                files_html += '<p style="color:#64748b;font-size:10px;margin:4px 0 0">…(capped at 20 files)</p>'

        # Snippets
        snippets_html = ""
        if r["snippets"]:
            snip_items = ""
            for s in r["snippets"]:
                snip_items += f"""
                <div style="border-left:2px solid #4c1d95;padding:4px 8px;margin-bottom:6px">
                  <div style="color:#a78bfa;font-size:10px">{s["rel_path"]}
                    — score: {s["final_score"]:.3f}
                    (bm25={s["bm25_score"]:.2f} vec={s["vector_score"]:.2f})
                  </div>
                  <div style="color:#d1d5db;font-size:11px;margin-top:2px">{s["snippet"] or "(no snippet)"}</div>
                </div>"""
            snippets_html = f'<div style="margin-top:6px">{snip_items}</div>'

        answer_html = ""
        if r["answer_text"] and r["actual_mode"] in ("structured", "cross_project"):
            escaped = r["answer_text"].replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br>")
            answer_html = f'<div style="background:#1e293b;border-radius:4px;padding:8px;font-size:11px;color:#94a3b8;margin-top:6px">{escaped}</div>'

        error_html = ""
        if r.get("error"):
            error_html = f'<div style="color:#f87171;font-size:11px;margin-top:4px">ERROR: {r["error"]}</div>'

        results_count = len(r["files"]) + len(r["snippets"])
        results_badge_bg = "#052e16" if results_count > 0 else "#1e293b"
        results_badge_txt = "#4ade80" if results_count > 0 else "#64748b"

        q_rows += f"""
        <tr style="border-top:1px solid #1e293b">
          <td style="color:#64748b;text-align:center;vertical-align:top;padding:10px 6px;
                     font-size:11px;white-space:nowrap">{r["id"]}</td>
          <td style="vertical-align:top;padding:10px 8px">
            <div style="color:#f1f5f9;font-size:13px">{r["q"]}{risk_badge}</div>
            <div style="margin-top:4px">{tags_html}</div>
            {f'<div style="color:#64748b;font-size:10px;margin-top:3px">filters: {filters_str}</div>' if filters_str else ""}
            <div style="color:#475569;font-size:10px;margin-top:4px">
              expected: <em style="color:#94a3b8">{r["expected_hint"]}</em>
            </div>
            {answer_html}
            {files_html}
            {snippets_html}
            {error_html}
          </td>
          <td style="vertical-align:top;padding:10px 8px;white-space:nowrap;text-align:center">
            <span style="background:{route_bg};color:{route_txt};font-size:11px;
                         padding:2px 7px;border-radius:4px;font-weight:700">{route_label}</span>
          </td>
          <td style="vertical-align:top;padding:10px 8px;white-space:nowrap">
            <span style="background:{exp_color}22;color:{exp_color};font-size:10px;
                         padding:1px 5px;border-radius:3px">{r["expected_mode"]}</span>
            <br>
            <span style="background:{act_color}22;color:{act_color};font-size:10px;
                         padding:1px 5px;border-radius:3px;margin-top:3px;display:inline-block">{r["actual_mode"]}</span>
          </td>
          <td style="vertical-align:top;padding:10px 8px;white-space:nowrap;
                     text-align:center;color:#64748b;font-size:11px">
            <span style="background:{results_badge_bg};color:{results_badge_txt};
                         padding:2px 6px;border-radius:3px">{results_count}</span>
          </td>
          <td style="vertical-align:top;padding:10px 8px;white-space:nowrap;
                     text-align:right;color:#475569;font-size:11px">{r["elapsed_ms"]:.0f} ms</td>
        </tr>"""

    gate_color = "#4ade80" if routing_pct >= 85 else "#f87171"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>TIGA Stress Eval — {ts}</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: #0f172a; color: #e2e8f0; font-family: ui-monospace, 'Cascadia Code', 'Fira Code', monospace; }}
  .container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
  h1 {{ font-size: 20px; color: #f8fafc; margin-bottom: 4px; }}
  .subtitle {{ color: #64748b; font-size: 12px; margin-bottom: 24px; }}
  .summary {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 16px 20px; flex: 1; min-width: 140px; }}
  .card-val {{ font-size: 28px; font-weight: 700; color: {gate_color}; }}
  .card-lbl {{ font-size: 11px; color: #64748b; margin-top: 4px; }}
  table {{ width: 100%; border-collapse: collapse; }}
  th {{ text-align: left; padding: 8px; color: #64748b; font-size: 11px; text-transform: uppercase; letter-spacing: .05em; border-bottom: 1px solid #1e293b; }}
  td {{ padding: 8px; vertical-align: top; }}
  .cat-table {{ background: #1e293b; border-radius: 8px; overflow: hidden; margin-bottom: 24px; }}
  .main-table {{ background: #111827; border-radius: 8px; overflow: hidden; }}
  tr:hover td {{ background: rgba(255,255,255,.02); }}
</style>
</head>
<body>
<div class="container">
  <h1>TIGA Hunt — Stress Eval Report</h1>
  <div class="subtitle">Project 186 Tianmu · {total} questions · Generated {ts}</div>

  <div class="summary">
    <div class="card">
      <div class="card-val">{routing_pct:.0f}%</div>
      <div class="card-lbl">Routing accuracy<br>({routing_pass}/{total} correct)</div>
    </div>
    <div class="card">
      <div class="card-val" style="color:#94a3b8">{sum(1 for r in results if r["files"] or r["snippets"])}</div>
      <div class="card-lbl">Questions with results returned</div>
    </div>
    <div class="card">
      <div class="card-val" style="color:#94a3b8">{round(sum(r["elapsed_ms"] for r in results)/total)}</div>
      <div class="card-lbl">Avg latency (ms)</div>
    </div>
    <div class="card">
      <div class="card-val" style="color:#f87171">{sum(1 for r in results if r.get("error"))}</div>
      <div class="card-lbl">Errors</div>
    </div>
  </div>

  <div class="cat-table">
    <table>
      <thead><tr>
        <th>Category</th><th>Routing</th><th style="width:140px"></th><th>%</th><th>Results</th>
      </tr></thead>
      <tbody>{cat_rows}</tbody>
    </table>
  </div>

  <div class="main-table">
    <table>
      <thead><tr>
        <th style="width:36px">#</th>
        <th>Question &amp; Results</th>
        <th style="width:72px;text-align:center">Route</th>
        <th style="width:100px">Mode<br><span style="font-size:9px">exp / actual</span></th>
        <th style="width:52px;text-align:center">Results</th>
        <th style="width:60px;text-align:right">Time</th>
      </tr></thead>
      <tbody>{q_rows}</tbody>
    </table>
  </div>

  <div style="margin-top:16px;color:#334155;font-size:11px;text-align:center">
    TIGA Hunt stress eval · {ts} · Questions without results may need indexing or Ollama running
  </div>
</div>
</body>
</html>"""


def run_stress_eval(project: str | None = None) -> int:
    """
    Run all stress-test questions end-to-end through router + executor.
    Exports a self-contained HTML report to tiga_work/reports/.

    Does NOT require Ollama for routing + file_locator + structured queries.
    Semantic/vector results will show 0 if Ollama is unavailable (BM25 only).

    Returns exit code: 0 always (stress eval is not a hard gate).
    """
    from core.router import QueryRouter
    from core.db import get_connection

    cfg.ensure_dirs()

    fixture = _load_stress_fixture()
    if not fixture:
        print(
            f"No stress test fixture found.\n"
            f"Expected: tests/{_STRESS_FIXTURE}",
            file=sys.stderr,
        )
        return 1

    # Filter to a specific project category if requested
    if project:
        fixture = [
            item for item in fixture
            if project.lower() in item.get("category", "").lower()
               or project.lower() in item.get("q", "").lower()
        ]
        if not fixture:
            print(f"No questions found matching project filter '{project}'.")
            return 1

    total = len(fixture)
    print(f"Running {total} stress-test questions…")
    print("(Ollama optional — file_locator + structured work without it)\n")

    router = QueryRouter()
    conn = get_connection(cfg.get_db_path())

    # Load known project codes so the router can detect them
    try:
        router.load_project_codes(conn)
    except Exception:
        pass

    results: list[dict[str, Any]] = []
    routing_pass = 0

    for i, item in enumerate(fixture, 1):
        r = _run_one_question(item, router, conn, cfg)
        results.append(r)

        ok_str = "OK  " if r["routing_ok"] else "MISS"
        results_n = len(r["files"]) + len(r["snippets"])
        print(
            f"  [{i:3d}/{total}] [{ok_str}] "
            f"mode={r['actual_mode']:<14} results={results_n:<3} "
            f"{r['q'][:60]!r}"
        )
        if r.get("error"):
            print(f"           ERROR: {r['error']}")
        if r["routing_ok"]:
            routing_pass += 1

    conn.close()

    routing_pct = routing_pass / total * 100 if total else 0

    # Save JSON
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    report_dir = cfg.get_report_dir()
    report_dir.mkdir(parents=True, exist_ok=True)

    json_path = report_dir / f"stress_eval_{ts}.json"
    json_path.write_text(
        json.dumps(results, indent=2, default=str),
        encoding="utf-8",
    )

    # Save HTML
    html_path = report_dir / f"stress_eval_{ts}.html"
    html_path.write_text(
        _generate_html_report(results, ts),
        encoding="utf-8",
    )

    # Summary
    print(f"\n{'=' * 60}")
    print(f"  Stress eval complete — {total} questions")
    print(f"  Routing accuracy  : {routing_pct:.0f}%  ({routing_pass}/{total})")
    print(f"  With results      : {sum(1 for r in results if r['files'] or r['snippets'])}/{total}")
    print(f"  Avg latency       : {round(sum(r['elapsed_ms'] for r in results)/total)} ms")

    # Category breakdown
    print(f"\n  Category breakdown:")
    cat_pass: dict[str, list[bool]] = {}
    for r in results:
        cat_pass.setdefault(r["category"], []).append(r["routing_ok"])
    for cat, vals in cat_pass.items():
        pct = sum(vals) / len(vals) * 100
        bar = "#" * int(pct / 5)
        label = _CATEGORY_LABELS.get(cat, cat)
        print(f"  {label:<30} {sum(vals)}/{len(vals)}  {bar}")

    print(f"\n  HTML report: {html_path}")
    print(f"  JSON data  : {json_path}")
    print(f"{'=' * 60}")

    return 0
