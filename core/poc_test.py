"""
core/poc_test.py — One-click POC test: pick projects → index → stress retrieval → export.

Terminology note for users:
  This is **corpus-adaptive eval** (auto-generate tests from what you indexed),
  not machine-learning reinforcement learning. The feedback loop is:
    office test → export dump → dev refines Hunt → redeploy → re-test
"""

from __future__ import annotations

import json
import logging
import shutil
import sqlite3
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from config import Config, load_config
from core.db import get_connection
from core.eval import _has_hit, _percentile, validate_citation
from core.field_collector import export_bundle as export_field_bundle
from core.index import run_full_pipeline
from core.query import search
from core.query_generator import generate_queries_for_projects
from core.router import QueryRouter

logger = logging.getLogger(__name__)

_SESSION_FILE = "session.json"
_PLAYBOOK_NAME = "REFINEMENT_PLAYBOOK.md"


def poc_dir(cfg: Config | None = None) -> Path:
    from config import cfg as default
    c = cfg or default
    d = c.work_dir / "poc_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


def exports_dir(cfg: Config | None = None) -> Path:
    d = poc_dir(cfg) / "exports"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ts_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


def save_session(data: dict[str, Any], cfg: Config | None = None) -> Path:
    path = poc_dir(cfg) / _SESSION_FILE
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_session(cfg: Config | None = None) -> dict[str, Any]:
    path = poc_dir(cfg) / _SESSION_FILE
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def list_projects_from_roots(
    index_roots: list[Path],
    *,
    max_depth: int = 1,
) -> list[dict[str, Any]]:
    """Discover project folders under index roots (filesystem scan)."""
    projects: list[dict[str, Any]] = []
    seen: set[str] = set()
    for root in index_roots:
        if not root.exists():
            continue
        if max_depth == 1:
            candidates = [p for p in root.iterdir() if p.is_dir()]
        else:
            candidates = [
                p for p in root.rglob("*")
                if p.is_dir() and len(p.relative_to(root).parts) == max_depth
            ]
        for cand in sorted(candidates):
            key = str(cand.resolve())
            if key in seen:
                continue
            seen.add(key)
            indexable = sum(
                1 for fp in cand.rglob("*")
                if fp.is_file() and fp.suffix.lower() in {
                    ".pdf", ".docx", ".pptx", ".txt", ".md", ".doc", ".xlsx"
                }
            )
            projects.append({
                "path": str(cand),
                "name": cand.name,
                "indexable_files": indexable,
                "root_parent": str(root),
            })
    projects.sort(key=lambda p: p["indexable_files"])
    return projects


def list_projects_from_db(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Projects already in the index."""
    rows = conn.execute(
        """
        SELECT project_id,
               COUNT(*) AS file_count,
               MIN(file_path) AS sample_path
        FROM files
        WHERE status = 'INDEXED' AND project_id IS NOT NULL AND project_id != 'Unknown'
        GROUP BY project_id
        ORDER BY file_count DESC
        """
    ).fetchall()
    return [
        {
            "project_id": r["project_id"],
            "path": str(Path(r["sample_path"]).parent),
            "name": r["project_id"],
            "indexed_files": r["file_count"],
            "sample_path": r["sample_path"],
        }
        for r in rows
    ]


def discover_projects(
    conn: sqlite3.Connection,
    cfg: Config,
    *,
    prefer_db: bool = True,
) -> list[dict[str, Any]]:
    db_projects = list_projects_from_db(conn) if prefer_db else []
    if len(db_projects) >= 2:
        return db_projects
    fs_projects = list_projects_from_roots(cfg.index_roots)
    if db_projects:
        db_paths = {p.get("project_id") for p in db_projects}
        for fp in fs_projects:
            if fp["name"] not in db_paths:
                db_projects.append({
                    "project_id": fp["name"],
                    "path": fp["path"],
                    "name": fp["name"],
                    "indexed_files": 0,
                    "indexable_files": fp["indexable_files"],
                })
        return db_projects
    return [
        {
            "project_id": p["name"],
            "path": p["path"],
            "name": p["name"],
            "indexed_files": 0,
            "indexable_files": p["indexable_files"],
        }
        for p in fs_projects
    ]


def _cfg_with_roots(base: Config, selected_paths: list[str]) -> Config:
    data = yaml.safe_load(base.config_file.read_text(encoding="utf-8"))
    data["index_roots"] = selected_paths
    overlay = base.work_dir / "poc_test" / "config_overlay.yaml"
    overlay.parent.mkdir(parents=True, exist_ok=True)
    overlay.write_text(yaml.dump(data), encoding="utf-8")
    return load_config(config_file=overlay, work_dir=base.work_dir)


def index_selected_projects(
    selected: list[dict[str, Any]],
    cfg: Config,
    conn: sqlite3.Connection,
) -> dict[str, Any]:
    paths = [s["path"] for s in selected if s.get("path")]
    test_cfg = _cfg_with_roots(cfg, paths)
    t0 = time.perf_counter()
    stats = run_full_pipeline(conn, test_cfg)
    elapsed = round(time.perf_counter() - t0, 1)
    return {"stats": stats, "elapsed_s": elapsed, "roots": paths}


def run_retrieval_stress(
    queries: list[dict[str, Any]],
    conn: sqlite3.Connection,
    cfg: Config,
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    """Hunt-only stress test with router synonym expansion."""
    router = QueryRouter()
    db_path = str(cfg.get_db_path())
    root_paths = [str(r) for r in cfg.index_roots]

    hits_lit = hits_para = 0
    n_lit = n_para = 0
    latencies: list[float] = []
    results_out: list[dict[str, Any]] = []

    for entry in queries:
        q = entry["query"]
        expected = entry.get("expected_paths") or []
        bucket = entry.get("bucket", "literal")
        if bucket == "literal":
            n_lit += 1
        else:
            n_para += 1

        expanded = router.expand_query(q).expanded_terms
        t0 = time.perf_counter()
        try:
            res = search(q, top_k=top_k, cfg_obj=cfg, conn=conn, expanded_terms=expanded)
        except Exception as exc:
            logger.warning("stress search failed %r: %s", q, exc)
            res = []
        elapsed = round((time.perf_counter() - t0) * 1000, 1)
        latencies.append(elapsed)

        paths = [r.get("file_path", "") for r in res]
        hit = _has_hit(paths, expected) if expected else False
        if hit:
            if bucket == "literal":
                hits_lit += 1
            else:
                hits_para += 1

        cites = [r.get("citation", "") for r in res if r.get("citation")]
        bad = [c for c in cites if not validate_citation(c, db_path, root_paths)]

        results_out.append({
            **entry,
            "hit": hit,
            "elapsed_ms": elapsed,
            "returned_paths": paths[:top_k],
            "invalid_citations": bad,
        })

    total = len(queries)
    lit_recall = round(hits_lit / n_lit, 4) if n_lit else 0.0
    para_recall = round(hits_para / n_para, 4) if n_para else 0.0
    overall = round((hits_lit + hits_para) / total, 4) if total else 0.0

    by_project: dict[str, dict[str, int]] = {}
    for r in results_out:
        pid = r.get("project_id") or "unknown"
        by_project.setdefault(pid, {"hits": 0, "total": 0})
        by_project[pid]["total"] += 1
        if r.get("hit"):
            by_project[pid]["hits"] += 1

    return {
        "mode": "hunt_stress",
        "ts": _ts_slug(),
        "total_queries": total,
        "literal_recall_pct": round(lit_recall * 100, 1),
        "paraphrase_recall_pct": round(para_recall * 100, 1),
        "overall_recall_pct": round(overall * 100, 1),
        "literal_hits": hits_lit,
        "literal_total": n_lit,
        "paraphrase_hits": hits_para,
        "paraphrase_total": n_para,
        "latency_p50_ms": round(_percentile(latencies, 50), 1),
        "latency_p95_ms": round(_percentile(latencies, 95), 1),
        "by_project": {
            k: {
                "recall_pct": round(v["hits"] / v["total"] * 100, 1) if v["total"] else 0,
                **v,
            }
            for k, v in by_project.items()
        },
        "queries": results_out,
        "misses": [r for r in results_out if not r.get("hit")],
    }


def _build_playbook(
    session: dict[str, Any],
    generation: dict[str, Any],
    stress: dict[str, Any],
    export_name: str,
) -> str:
    misses = stress.get("misses") or []
    miss_lines = "\n".join(
        f"- `{m.get('query')}` → expected `{m.get('expected_paths')}` "
        f"got `{ (m.get('returned_paths') or ['NONE'])[0] }`"
        for m in misses[:15]
    ) or "- (none — great job!)"

    return f"""# TIGA POC Test — Refinement Playbook

Generated: {_utc_now()}
Export: `{export_name}`

## What this test was

Corpus-adaptive retrieval stress test (Hunt only — **no LLM/API cost**).
Queries were **auto-generated from your indexed projects**, not hand-written.

| Metric | Result |
|--------|--------|
| Projects tested | {len(session.get('selected', []))} |
| Queries generated | {generation.get('total', 0)} |
| Literal recall | {stress.get('literal_recall_pct')}% |
| Paraphrase recall | {stress.get('paraphrase_recall_pct')}% |
| Overall top-5 | {stress.get('overall_recall_pct')}% |

**Targets:** literal ≥90%, paraphrase ≥80%

---

## Bring this dump back to dev (Cursor agent)

Paste this block:

```
Import my office POC test export: poc_test/exports/{export_name}
Review misses in stress_results.json and generated_queries.yaml.
Tune Hunt only (no API):
  1. core/retrieval_boost.py _DOMAIN_EXPAND for paraphrase misses
  2. tiga_work/query_synonyms.yml for concept routing misses
  3. tests/fixtures/search_benchmark_dual.yaml — add top 5 misses as permanent tests
Re-run: python tiga.py validate
```

---

## Run a variant test in the office (after changing config)

```bat
:: Re-stress without re-indexing (fast — config/reranker/synonym changes)
python tiga.py poc-test stress

:: Full re-run: pick projects → index → stress → export
python tiga.py poc-test run

:: Or one-click
poc-test.bat
```

### Useful config knobs to A/B test

```yaml
retrieval:
  reranker_enabled: true|false
  domain_expand_enabled: true|false
  path_boost_enabled: true|false
  hybrid_weight_vector: 0.6   # needs Ollama embed running
```

---

## Top misses to fix first

{miss_lines}

---

## Add gold labels when staff know the answer

```bat
python tiga.py collect label --query "your query" --expected path/to/correct/file.pdf
python tiga.py collect export
```

---

## Not reinforcement learning

This loop is **eval-driven refinement**: auto-generated tests from your corpus,
export misses, human/agent tunes retrieval rules, re-test. True RL would train
weights from feedback — overkill for Gateway 1. This gets you Google-close Hunt
before spending on Einstein API.
"""


def export_poc_test_bundle(
    session: dict[str, Any],
    generation: dict[str, Any],
    stress: dict[str, Any],
    index_report: dict[str, Any] | None,
    cfg: Config,
    conn: sqlite3.Connection | None = None,
    *,
    include_field_collect: bool = True,
) -> Path:
    """Zip everything needed for dev refinement."""
    name = f"poc_test_export_{_ts_slug()}.zip"
    zip_path = exports_dir(cfg) / name

    playbook = _build_playbook(session, generation, stress, name)

    eval_yaml = [
        {
            "query": q["query"],
            "expected_paths": q.get("expected_paths") or [],
            "bucket": q.get("bucket"),
            "project_id": q.get("project_id"),
        }
        for q in (generation.get("queries") or [])
        if q.get("expected_paths")
    ]

    manifest = {
        "export_version": 1,
        "type": "poc_test",
        "exported_at": _utc_now(),
        "session": session,
        "generation_summary": {
            "total": generation.get("total"),
            "literal": generation.get("literal"),
            "paraphrase": generation.get("paraphrase"),
        },
        "stress_summary": {
            "literal_recall_pct": stress.get("literal_recall_pct"),
            "paraphrase_recall_pct": stress.get("paraphrase_recall_pct"),
            "overall_recall_pct": stress.get("overall_recall_pct"),
            "miss_count": len(stress.get("misses") or []),
        },
        "index_report": index_report,
    }

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))
        zf.writestr("generated_queries.yaml", yaml.dump(generation, allow_unicode=True))
        zf.writestr("stress_results.json", json.dumps(stress, indent=2))
        zf.writestr("office_eval.yaml", yaml.dump(eval_yaml, allow_unicode=True))
        zf.writestr(_PLAYBOOK_NAME, playbook)
        zf.writestr("session.json", json.dumps(session, indent=2))
        if index_report:
            zf.writestr("index_report.json", json.dumps(index_report, indent=2))

        if include_field_collect and conn:
            try:
                fc_path = export_field_bundle(conn, cfg_obj=cfg)
                zf.write(fc_path, arcname=f"field_collect/{fc_path.name}")
            except Exception as exc:
                logger.warning("field collect in poc export skipped: %s", exc)

    save_session({**session, "last_export": str(zip_path), "last_stress": stress}, cfg)
    return zip_path


def import_poc_test_bundle(zip_path: Path, cfg: Config | None = None) -> dict[str, Any]:
    """Import office POC test export on dev machine."""
    from core.field_collector import import_bundle as import_field
    c = cfg or load_config()
    dest = poc_dir(c) / Path(zip_path).stem
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    fixtures = c.work_dir / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in (
        ("office_eval.yaml", "office_eval_from_poc.yaml"),
        ("generated_queries.yaml", "generated_queries_from_poc.yaml"),
    ):
        src = dest / src_name
        if src.exists():
            shutil.copy2(src, fixtures / dst_name)

    merged = fixtures / "office_eval_merged.yaml"
    new_eval = yaml.safe_load((dest / "office_eval.yaml").read_text(encoding="utf-8"))
    existing: list = []
    if merged.exists():
        existing = yaml.safe_load(merged.read_text(encoding="utf-8")) or []
    if not isinstance(existing, list):
        existing = []
    seen = {e.get("query", "").lower() for e in existing if isinstance(e, dict)}
    if isinstance(new_eval, list):
        for e in new_eval:
            if isinstance(e, dict) and e.get("query", "").lower() not in seen:
                existing.append(e)
                seen.add(e.get("query", "").lower())
    merged.write_text(yaml.dump(existing, allow_unicode=True), encoding="utf-8")

    fc_zips = list(dest.glob("field_collect/*.zip"))
    field_result = None
    if fc_zips:
        field_result = import_field(fc_zips[0], cfg_obj=c)

    return {
        "imported_to": str(dest),
        "playbook": str(dest / _PLAYBOOK_NAME),
        "stress_results": str(dest / "stress_results.json"),
        "office_eval_merged": str(merged),
        "field_import": field_result,
    }


def run_poc_test(
    selected: list[dict[str, Any]],
    cfg: Config,
    *,
    skip_index: bool = False,
    top_k: int = 5,
    verbose: bool = True,
) -> dict[str, Any]:
    """Full pipeline: index → generate → stress → export."""
    conn = get_connection(cfg.get_db_path())
    try:
        session = {
            "started_at": _utc_now(),
            "selected": selected,
            "skip_index": skip_index,
        }
        save_session(session, cfg)

        index_report = None
        if not skip_index:
            if verbose:
                print(f"\n{'=' * 60}")
                print("  Step 2/4 — Indexing selected projects")
                print(f"{'=' * 60}\n")
            index_report = index_selected_projects(selected, cfg, conn)
            if verbose:
                s = index_report["stats"]
                print(f"  Indexed in {index_report['elapsed_s']}s — "
                      f"files: {s.get('files_indexed', 0)}")

        project_keys = []
        for sel in selected:
            key = sel.get("project_id") or sel.get("name") or Path(sel.get("path", "")).name
            if key:
                project_keys.append(key)

        if verbose:
            print(f"\n{'=' * 60}")
            print("  Step 3/4 — Generating corpus-adaptive queries")
            print(f"{'=' * 60}\n")

        gen = generate_queries_for_projects(conn, project_keys, cfg.index_roots)
        gen_dict = gen.to_dict()
        if verbose:
            print(f"  Generated {gen_dict['total']} queries "
                  f"({gen_dict['literal']} literal, {gen_dict['paraphrase']} paraphrase)")

        if verbose:
            print(f"\n{'=' * 60}")
            print("  Step 4/4 — Retrieval stress test (Hunt only)")
            print(f"{'=' * 60}\n")

        stress = run_retrieval_stress(
            gen_dict["queries"], conn, cfg, top_k=top_k
        )
        if verbose:
            print(f"  Literal recall:    {stress['literal_recall_pct']}%")
            print(f"  Paraphrase recall: {stress['paraphrase_recall_pct']}%")
            print(f"  Overall top-5:     {stress['overall_recall_pct']}%")
            print(f"  Misses:            {len(stress.get('misses') or [])}")

        if verbose:
            print(f"\n{'=' * 60}")
            print("  Exporting test dump")
            print(f"{'=' * 60}\n")

        zip_path = export_poc_test_bundle(
            session, gen_dict, stress, index_report, cfg, conn
        )
        if verbose:
            print(f"  Export: {zip_path}")
            print(f"\n  Open REFINEMENT_PLAYBOOK.md inside the zip for next steps.")
            print(f"  Copy zip to dev → python tiga.py poc-test import <zip>\n")

        return {
            "session": session,
            "generation": gen_dict,
            "stress": stress,
            "index_report": index_report,
            "export_path": str(zip_path),
            "exit_code": 0 if stress["overall_recall_pct"] >= 80 else 2,
        }
    finally:
        conn.close()
