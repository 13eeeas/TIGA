"""
core/atlas_model.py — Atlas Ticket A: first-class structured rows.

Principle: the database (overlay rows) is the source of truth. Project pages
assemble from rows + citations. A blank free-form wiki dump is not SoT.

Row kinds
---------
  identity     — name / typology / client / stage / location
  lifecycle    — stage (+ optional note)
  strategies   — design-strategy tags
  team         — roles / people / orgs
  documents    — authoritative | superseded | candidate (Hunt-seeded)
  facts        — evidence-native claims with cite_paths
  decisions    — dated decisions with citations
  precedents   — links to related project codes

Legacy v1 overlays (pins / facts / project / summary) normalize in-memory to
schema_version 2 without rewriting until the next save.
"""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

SCHEMA_VERSION = 2
DOC_AUTHORITIES = ("authoritative", "superseded", "candidate")
DECISION_STATUSES = ("proposed", "accepted", "rejected", "superseded")
PRECEDENT_RELATIONS = ("related", "precedent", "successor", "variant")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_id(prefix: str = "r") -> str:
    return f"{prefix}_{uuid4().hex[:10]}"


def _path_id(path: str) -> str:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:10]
    return f"doc_{digest}"


def empty_overlay(code: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "product": "tiga-atlas-overlay",
        "project_code": code,
        "updated_at": _utc(),
        "summary": None,
        "project": {},
        "lifecycle": {},
        "strategies": [],
        "team": [],
        "documents": [],
        "facts": [],
        "decisions": [],
        "precedents": [],
        "pins": [],
        "hidden_paths": [],
        "open_questions": [],
    }


def normalize_overlay(code: str, overlay: dict[str, Any] | None) -> dict[str, Any]:
    """Upgrade any overlay dict to schema v2 shape (non-destructive)."""
    base = empty_overlay(code)
    if not overlay:
        return base
    out = {**base, **{k: v for k, v in overlay.items() if v is not None}}
    out["schema_version"] = SCHEMA_VERSION
    out["product"] = "tiga-atlas-overlay"
    out["project_code"] = code
    for key in (
        "strategies",
        "team",
        "documents",
        "facts",
        "decisions",
        "precedents",
        "pins",
        "hidden_paths",
        "open_questions",
    ):
        if not isinstance(out.get(key), list):
            out[key] = []
    if not isinstance(out.get("project"), dict):
        out["project"] = {}
    if not isinstance(out.get("lifecycle"), dict):
        out["lifecycle"] = {}

    # Sync lifecycle.stage from curated project.stage when missing.
    stage = out["project"].get("stage")
    if stage and not out["lifecycle"].get("stage"):
        out["lifecycle"]["stage"] = stage

    # Lift legacy pins into document rows (authoritative) when path not present.
    docs_by_path = {
        str(d.get("path")): d for d in out["documents"] if d.get("path")
    }
    for pin in out["pins"]:
        path = pin.get("path")
        if not path:
            continue
        if path in docs_by_path:
            doc = docs_by_path[path]
            if doc.get("authority") == "candidate":
                doc["authority"] = "authoritative"
            if pin.get("role") and not doc.get("role"):
                doc["role"] = pin.get("role")
            continue
        doc = {
            "id": _path_id(path),
            "path": path,
            "title": pin.get("title") or path.rsplit("/", 1)[-1],
            "authority": "authoritative",
            "role": pin.get("role") or "other",
            "note": pin.get("note") or "",
            "cite_paths": [path],
            "supersedes": [],
            "superseded_by": [],
            "source": pin.get("source") or "wiki",
            "updated_at": pin.get("pinned_at") or _utc(),
        }
        out["documents"].append(doc)
        docs_by_path[path] = doc

    return out


def _upsert_by_id(rows: list[dict[str, Any]], row: dict[str, Any], id_key: str = "id") -> dict[str, Any]:
    rid = row.get(id_key)
    if rid:
        for i, existing in enumerate(rows):
            if existing.get(id_key) == rid:
                merged = {**existing, **row}
                rows[i] = merged
                return merged
    if not rid:
        row[id_key] = _row_id(id_key[:3] if id_key else "r")
    rows.append(row)
    return row


def upsert_document(
    overlay: dict[str, Any],
    *,
    path: str,
    title: str = "",
    authority: str = "candidate",
    role: str = "other",
    note: str = "",
    supersedes: list[str] | None = None,
    superseded_by: list[str] | None = None,
    source: str = "wiki",
    doc_id: str | None = None,
) -> dict[str, Any]:
    if authority not in DOC_AUTHORITIES:
        authority = "candidate"
    docs = overlay.setdefault("documents", [])
    existing = None
    for d in docs:
        if path and d.get("path") == path:
            existing = d
            break
        if doc_id and d.get("id") == doc_id:
            existing = d
            break
    row = {
        "id": (existing or {}).get("id") or doc_id or _path_id(path),
        "path": path,
        "title": title or (existing or {}).get("title") or path.rsplit("/", 1)[-1],
        "authority": authority,
        "role": role or (existing or {}).get("role") or "other",
        "note": note if note != "" else (existing or {}).get("note") or "",
        "cite_paths": [path] if path else list((existing or {}).get("cite_paths") or []),
        "supersedes": supersedes if supersedes is not None else list((existing or {}).get("supersedes") or []),
        "superseded_by": superseded_by if superseded_by is not None else list((existing or {}).get("superseded_by") or []),
        "source": source or (existing or {}).get("source") or "wiki",
        "updated_at": _utc(),
    }
    if existing:
        for i, d in enumerate(docs):
            if d is existing or d.get("id") == existing.get("id"):
                docs[i] = row
                break
    else:
        docs.append(row)
    return row


def upsert_strategy(
    overlay: dict[str, Any],
    *,
    tag: str,
    label: str = "",
    note: str = "",
    cite_paths: list[str] | None = None,
    status: str = "active",
    strategy_id: str | None = None,
) -> dict[str, Any]:
    rows = overlay.setdefault("strategies", [])
    row = {
        "id": strategy_id,
        "tag": tag.strip(),
        "label": (label or tag).strip(),
        "note": note,
        "cite_paths": cite_paths or [],
        "status": status,
        "updated_at": _utc(),
    }
    if strategy_id:
        return _upsert_by_id(rows, row)
    for existing in rows:
        if existing.get("tag") == row["tag"]:
            row["id"] = existing.get("id")
            return _upsert_by_id(rows, row)
    row["id"] = _row_id("str")
    rows.append(row)
    return row


def upsert_team_member(
    overlay: dict[str, Any],
    *,
    role: str,
    name: str = "",
    org: str = "",
    note: str = "",
    member_id: str | None = None,
) -> dict[str, Any]:
    rows = overlay.setdefault("team", [])
    row = {
        "id": member_id,
        "role": role.strip(),
        "name": name.strip(),
        "org": org.strip(),
        "note": note,
        "updated_at": _utc(),
    }
    return _upsert_by_id(rows, row, id_key="id")


def upsert_decision(
    overlay: dict[str, Any],
    *,
    title: str,
    status: str = "proposed",
    rationale: str = "",
    cite_paths: list[str] | None = None,
    decided_at: str | None = None,
    related_strategy_ids: list[str] | None = None,
    decision_id: str | None = None,
) -> dict[str, Any]:
    if status not in DECISION_STATUSES:
        status = "proposed"
    rows = overlay.setdefault("decisions", [])
    row = {
        "id": decision_id,
        "title": title.strip(),
        "status": status,
        "rationale": rationale,
        "cite_paths": cite_paths or [],
        "decided_at": decided_at or _utc(),
        "related_strategy_ids": related_strategy_ids or [],
        "updated_at": _utc(),
    }
    return _upsert_by_id(rows, row)


def upsert_precedent(
    overlay: dict[str, Any],
    *,
    project_code: str,
    relation: str = "related",
    note: str = "",
    cite_paths: list[str] | None = None,
    precedent_id: str | None = None,
) -> dict[str, Any]:
    if relation not in PRECEDENT_RELATIONS:
        relation = "related"
    rows = overlay.setdefault("precedents", [])
    row = {
        "id": precedent_id,
        "project_code": project_code.strip(),
        "relation": relation,
        "note": note,
        "cite_paths": cite_paths or [],
        "updated_at": _utc(),
    }
    if precedent_id:
        return _upsert_by_id(rows, row)
    for existing in rows:
        if existing.get("project_code") == row["project_code"] and existing.get("relation") == relation:
            row["id"] = existing.get("id")
            return _upsert_by_id(rows, row)
    row["id"] = _row_id("pre")
    rows.append(row)
    return row


def set_lifecycle(overlay: dict[str, Any], stage: str, note: str = "") -> dict[str, Any]:
    stage = (stage or "").strip()
    lifecycle = overlay.setdefault("lifecycle", {})
    if stage:
        lifecycle["stage"] = stage
        proj = overlay.setdefault("project", {})
        proj["stage"] = stage
    if note:
        lifecycle["note"] = note
    lifecycle["updated_at"] = _utc()
    return lifecycle


def hunt_document_proposals(
    conn: sqlite3.Connection,
    code: str,
    *,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """Seed candidate document rows from Hunt authority signals (not silent truth)."""
    try:
        rows = conn.execute(
            "SELECT file_path, file_name, "
            "COALESCE(is_latest, 0) AS is_latest, "
            "COALESCE(is_superseded, 0) AS is_superseded, "
            "COALESCE(revision, '') AS revision, "
            "COALESCE(canonical_category, '') AS category "
            "FROM files WHERE COALESCE(project_id, 'Unknown') = ? "
            "AND (COALESCE(is_latest, 0) = 1 OR COALESCE(is_superseded, 0) = 1) "
            "ORDER BY is_latest DESC, is_superseded DESC, file_path "
            "LIMIT ?",
            (code, limit),
        ).fetchall()
    except sqlite3.Error:
        return []

    out: list[dict[str, Any]] = []
    for r in rows:
        path = r["file_path"]
        if r["is_superseded"]:
            authority = "superseded"
        elif r["is_latest"]:
            authority = "authoritative"
        else:
            authority = "candidate"
        out.append(
            {
                "id": _path_id(path),
                "path": path,
                "title": r["file_name"] or path.rsplit("/", 1)[-1],
                "authority": authority,
                "role": (r["category"] or "other").lower() or "other",
                "note": f"Hunt signal: revision={r['revision'] or '—'} "
                f"latest={int(r['is_latest'])} superseded={int(r['is_superseded'])}",
                "cite_paths": [path],
                "supersedes": [],
                "superseded_by": [],
                "source": "hunt-proposal",
                "status": "proposal",  # Ticket B: human must approve before silent truth
                "updated_at": _utc(),
            }
        )
    return out


def merge_document_proposals(
    overlay: dict[str, Any],
    proposals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Attach Hunt proposals without overwriting wiki-confirmed document rows."""
    docs = overlay.setdefault("documents", [])
    by_path = {str(d.get("path")): d for d in docs if d.get("path")}
    added: list[dict[str, Any]] = []
    for prop in proposals:
        path = prop.get("path")
        if not path:
            continue
        existing = by_path.get(path)
        if existing and existing.get("source") != "hunt-proposal":
            # Wiki / confirmed row wins; keep Hunt note as hint only.
            continue
        if existing and existing.get("source") == "hunt-proposal":
            for i, d in enumerate(docs):
                if d.get("path") == path:
                    docs[i] = {**existing, **prop, "id": existing.get("id") or prop.get("id")}
                    break
            continue
        docs.append(prop)
        by_path[path] = prop
        added.append(prop)
    return added


def documents_by_authority(overlay: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    buckets = {k: [] for k in DOC_AUTHORITIES}
    for doc in overlay.get("documents") or []:
        auth = doc.get("authority") if doc.get("authority") in DOC_AUTHORITIES else "candidate"
        # Proposals stay candidates until Ticket B approval, even if Hunt said authoritative.
        if doc.get("status") == "proposal" or doc.get("source") == "hunt-proposal":
            if auth == "authoritative":
                # Still surface under authoritative *proposals* list via flag; bucket as candidate for SoT.
                buckets["candidate"].append({**doc, "proposed_authority": auth})
                continue
        buckets[auth].append(doc)
    return buckets


def assemble_from_rows(
    *,
    code: str,
    overlay: dict[str, Any],
    project: dict[str, Any],
    blurb: str,
    facts: list[dict[str, Any]],
    hunt_proposals: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Regenerate a project page solely from structured rows + citations.

    This is Ticket A's acceptance surface: blank wiki prose is not required.
    """
    ov = normalize_overlay(code, overlay)
    if hunt_proposals:
        merge_document_proposals(ov, hunt_proposals)
    docs = documents_by_authority(ov)
    lifecycle = dict(ov.get("lifecycle") or {})
    if not lifecycle.get("stage") and project.get("stage"):
        lifecycle["stage"] = project.get("stage")

    timeline = []
    for dec in ov.get("decisions") or []:
        timeline.append(
            {
                "kind": "decision",
                "at": dec.get("decided_at") or dec.get("updated_at"),
                "title": dec.get("title"),
                "status": dec.get("status"),
                "cite_paths": dec.get("cite_paths") or [],
            }
        )
    timeline.sort(key=lambda x: x.get("at") or "", reverse=True)

    cited_facts = [f for f in facts if f.get("cite_paths")]
    return {
        "schema_version": SCHEMA_VERSION,
        "product": "tiga-atlas-model",
        "source_of_truth": "structured_rows",
        "identity": {
            "code": project.get("code") or code,
            "name": project.get("name") or code,
            "typology": project.get("typology"),
            "client": project.get("client"),
            "stage": project.get("stage") or lifecycle.get("stage"),
            "location": project.get("location"),
            "status": project.get("status"),
        },
        "overview": {
            "blurb": blurb,
            "summary": ov.get("summary") or "",
        },
        "lifecycle": lifecycle,
        "strategies": list(ov.get("strategies") or []),
        "data": {
            "facts": facts,
            "cited_facts": cited_facts,
            "uncited_facts": [f for f in facts if not f.get("cite_paths")],
        },
        "team": list(ov.get("team") or []),
        "timeline": timeline,
        "decisions": list(ov.get("decisions") or []),
        "documents": {
            "authoritative": docs["authoritative"],
            "superseded": docs["superseded"],
            "candidates": docs["candidate"],
        },
        "related": list(ov.get("precedents") or []),
        "open_questions": list(ov.get("open_questions") or []),
    }


def model_answers_project_questions(assembled: dict[str, Any]) -> dict[str, Any]:
    """Cheap readiness checks for Tickets E / board narrative."""
    identity = assembled.get("identity") or {}
    docs = assembled.get("documents") or {}
    what = bool(
        identity.get("name")
        and (identity.get("typology") or identity.get("client") or assembled.get("overview", {}).get("blurb"))
    )
    superseded = bool(docs.get("superseded") or docs.get("authoritative"))
    return {
        "what_is_this_project": what,
        "what_superseded_what": bool(docs.get("superseded")),
        "has_authority_docs": bool(docs.get("authoritative")),
        "has_any_doc_rows": superseded,
    }
