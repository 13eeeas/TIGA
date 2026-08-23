"""
core/query_generator.py — Corpus-adaptive query generation for architecture firms.

Generates literal + paraphrase Hunt test queries from what is actually indexed.
Auto-labels expected_paths from folder semantics (Brief, Tender, etc.) — this is
**corpus-adaptive eval**, not ML reinforcement learning, but it closes the loop:
  index → generate tests from corpus → stress retrieval → export misses → refine Hunt

No API cost. Queries reflect real project structure on the NAS.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Architecture-firm query templates: (bucket, query template, match hints)
# match: dict with any of category, folder_stage, path_contains (list of substrings)
_TEMPLATE_SPECS: list[dict[str, Any]] = [
    {
        "id": "brief_literal",
        "bucket": "literal",
        "template": "{code} project brief client requirements",
        "match": {"path_contains": ["brief"], "categories": ["brief", "design"]},
    },
    {
        "id": "brief_paraphrase",
        "bucket": "paraphrase",
        "template": "what did the client ask for on {name}",
        "match": {"path_contains": ["brief"], "categories": ["brief", "design"]},
    },
    {
        "id": "tender_literal",
        "bucket": "literal",
        "template": "{code} tender documentation Rev submission",
        "match": {"path_contains": ["tender"], "categories": ["tender", "submission"]},
    },
    {
        "id": "tender_paraphrase",
        "bucket": "paraphrase",
        "template": "contractor pricing drawings package for {code}",
        "match": {"path_contains": ["tender"], "categories": ["tender"]},
    },
    {
        "id": "submission_literal",
        "bucket": "literal",
        "template": "{code} authority submission planning",
        "match": {"path_contains": ["submission", "authority"], "categories": ["submission"]},
    },
    {
        "id": "renders_literal",
        "bucket": "literal",
        "template": "{code} 3D renders presentation images",
        "match": {"path_contains": ["render", "viz", "presentation", "image"], "categories": ["renders_3d", "renders", "presentations"]},
    },
    {
        "id": "meetings_literal",
        "bucket": "literal",
        "template": "{code} meeting minutes design review",
        "match": {"path_contains": ["minute", "meeting"], "categories": ["meetings", "minutes"]},
    },
    {
        "id": "spec_literal",
        "bucket": "literal",
        "template": "{code} specifications technical spec",
        "match": {"path_contains": ["spec"], "categories": ["specifications", "spec"]},
    },
    {
        "id": "ifc_literal",
        "bucket": "literal",
        "template": "{code} IFC issued for construction set",
        "match": {"path_contains": ["ifc", "construction"], "folder_stages": ["IFC"]},
    },
    {
        "id": "competition_paraphrase",
        "bucket": "paraphrase",
        "template": "competition entry submission for {name}",
        "match": {"path_contains": ["competition", "submission"], "categories": ["submission"]},
    },
    {
        "id": "typology_paraphrase",
        "bucket": "paraphrase",
        "template": "{typology} building design concept {code}",
        "match": {"path_contains": ["brief", "concept", "design"], "categories": ["brief", "design"]},
    },
    {
        "id": "site_paraphrase",
        "bucket": "paraphrase",
        "template": "site analysis context for {name}",
        "match": {"path_contains": ["site"], "categories": ["site", "design"]},
    },
]


@dataclass
class GeneratedQuery:
    query: str
    expected_paths: list[str]
    bucket: str
    template_id: str
    project_id: str
    project_name: str
    anchor_file: str
    category: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "expected_paths": self.expected_paths,
            "bucket": self.bucket,
            "template_id": self.template_id,
            "project_id": self.project_id,
            "project_name": self.project_name,
            "anchor_file": self.anchor_file,
            "category": self.category,
        }


@dataclass
class QueryGenerationReport:
    queries: list[GeneratedQuery] = field(default_factory=list)
    skipped: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": len(self.queries),
            "literal": sum(1 for q in self.queries if q.bucket == "literal"),
            "paraphrase": sum(1 for q in self.queries if q.bucket == "paraphrase"),
            "queries": [q.to_dict() for q in self.queries],
            "skipped": self.skipped,
        }


def _rel_suffix(file_path: str, index_roots: list[Path]) -> str:
    p = Path(file_path)
    for root in index_roots:
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            continue
    parts = p.parts
    if len(parts) >= 2:
        return "/".join(parts[-3:])
    return p.name


def _project_display_name(project_id: str, sample_path: str) -> str:
    """Human-ish name from project_id or folder."""
    if project_id and project_id != "Unknown":
        bits = re.split(r"[_\-\s]", project_id)
        if len(bits) > 1:
            return " ".join(b for b in bits[1:] if b and not b.isdigit()) or project_id
        return project_id
    return Path(sample_path).parent.name


def _file_matches(
    row: dict[str, Any],
    match: dict[str, Any],
) -> bool:
    path_l = (row.get("file_path") or "").lower()
    name_l = (row.get("file_name") or "").lower()
    cat = (row.get("canonical_category") or "").lower()
    stage = (row.get("folder_stage") or "").lower()

    for hint in match.get("path_contains") or []:
        if hint.lower() in path_l or hint.lower() in name_l:
            return True
    for c in match.get("categories") or []:
        if c.lower() in cat:
            return True
    for s in match.get("folder_stages") or []:
        if s.lower() == stage:
            return True
    return False


def _project_file_clause(key: str) -> tuple[str, list[str]]:
    """Match project_id or path containing folder key (handles Unknown project_id)."""
    return (
        "(project_id = ? OR file_path LIKE ?)",
        [key, f"%{key}%"],
    )


def _find_anchor_file(
    conn: sqlite3.Connection,
    project_key: str,
    match: dict[str, Any],
) -> dict[str, Any] | None:
    clause, params = _project_file_clause(project_key)
    rows = conn.execute(
        f"""
        SELECT file_path, file_name, canonical_category, folder_stage, typology,
               COALESCE(is_latest, 0) AS is_latest
        FROM files
        WHERE status = 'INDEXED' AND {clause}
        ORDER BY is_latest DESC, file_path ASC
        """,
        params,
    ).fetchall()
    candidates = [dict(r) for r in rows if _file_matches(dict(r), match)]
    if not candidates:
        return None
    candidates.sort(key=lambda r: (-int(r.get("is_latest") or 0), r["file_path"]))
    return candidates[0]


def generate_queries_for_projects(
    conn: sqlite3.Connection,
    project_keys: list[str],
    index_roots: list[Path],
    *,
    max_per_project: int = 12,
) -> QueryGenerationReport:
    """
    Build literal + paraphrase queries with auto expected_paths from indexed files.

    project_keys: project_id or folder name substring (e.g. '261_tianmu').
    """
    report = QueryGenerationReport()
    seen_queries: set[str] = set()

    for key in project_keys:
        count = 0
        clause, params = _project_file_clause(key)
        sample = conn.execute(
            f"SELECT file_path, typology, project_id FROM files "
            f"WHERE status='INDEXED' AND {clause} LIMIT 1",
            params,
        ).fetchone()
        if not sample:
            report.skipped.append({"project_key": key, "reason": "no indexed files"})
            continue

        pid = sample["project_id"] if sample["project_id"] != "Unknown" else key
        name = _project_display_name(pid, sample["file_path"])
        code = key.split("_")[0] if "_" in key else key
        if not (code.isdigit() or re.match(r"^\d{3,4}", code)):
            m = re.search(r"\b(\d{3,4})\b", key)
            code = m.group(1) if m else key.split("_")[0]

        typology = (sample["typology"] or "project").lower()
        if typology == "unknown":
            typology = "building"

        for spec in _TEMPLATE_SPECS:
            if count >= max_per_project:
                break
            anchor = _find_anchor_file(conn, key, spec["match"])
            if not anchor:
                continue

            rel = _rel_suffix(anchor["file_path"], index_roots)
            q = spec["template"].format(
                code=code,
                name=name,
                typology=typology,
                project_id=pid,
            )
            q_key = q.strip().lower()
            if q_key in seen_queries:
                continue
            seen_queries.add(q_key)

            gq = GeneratedQuery(
                query=q,
                expected_paths=[rel],
                bucket=spec["bucket"],
                template_id=spec["id"],
                project_id=pid,
                project_name=name,
                anchor_file=rel,
                category=anchor.get("canonical_category") or spec["id"],
            )
            report.queries.append(gq)
            count += 1

    return report
