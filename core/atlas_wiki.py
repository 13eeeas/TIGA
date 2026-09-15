"""
core/atlas_wiki.py — Cited project wiki inside TIGA Hunt.

North star: Notion-like browse + Wikipedia-like contribution + Grokopedia-like
auto draft — every durable claim needs a citation path.

Layers
------
  auto   : drafted from project_card + Hunt search candidates (machine)
  wiki   : human pins / facts / hidden paths in tiga_work/atlas/*.overlay.json
  gate   : health grade — draft until pins + cited facts pass

Public API
----------
  list_wiki_projects(conn) -> list
  get_wiki_page(code, conn, cfg) -> dict   # auto + wiki merge
  wiki_pin / wiki_hide / wiki_fact / wiki_unhide
  wiki_compose(code, question, conn, cfg) -> dict  # Einstein-lite stub/API
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from copy import copy
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import cfg as _module_cfg, Config

logger = logging.getLogger(__name__)

SEED_QUERIES = [
    ("overview", "project brief development brief design brief tender brief"),
    ("authority", "invitation letter tender EOI submission approval authority"),
    ("drawings", "submission plan section elevation drawing"),
    ("model", "master model Rhino Revit BIM"),
    ("correspondence", "consultant response letter correspondence meeting minutes"),
]

_SECTION_LIMIT = 8
_BUCKET_LIMIT = 4
_LOW_VALUE_EXTS = {".udsmesh", ".log", ".tmp", ".bak", ".autosave", ".db"}
_GENERIC_IMAGE_RE = re.compile(r"^(?:img|image|render|screenshot|copy)?[\s_-]*\d{1,4}$", re.I)
_QUALITY_TERMS = {
    "overview": ("brief", "overview", "tender", "development", "design", "eoi"),
    "authority": ("invitation", "submission", "approval", "authority", "tender", "eoi", "gebiz"),
    "drawings": ("plan", "section", "elevation", "drawing", "submission", "ga"),
    "model": ("revit", "rhino", "bim", "model", "master", "ifc"),
    "correspondence": ("letter", "response", "minutes", "correspondence", "consultant", "email"),
}

PIN_ROLES = (
    "overview_deck",
    "design_report",
    "authoritative_ga",
    "authority_pack",
    "model_master",
    "other",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(code: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", code.lower()).strip("-") or "project"


def _atlas_dir(cfg_obj: Config | None = None) -> Path:
    cfg = cfg_obj or _module_cfg
    d = cfg.work_dir / "atlas"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _overlay_path(code: str, cfg_obj: Config | None = None) -> Path:
    return _atlas_dir(cfg_obj) / f"{_slug(code)}.overlay.json"


def load_overlay(code: str, cfg_obj: Config | None = None) -> dict[str, Any]:
    path = _overlay_path(code, cfg_obj)
    if not path.exists():
        return {
            "schema_version": 1,
            "product": "tiga-atlas-overlay",
            "project_code": code,
            "updated_at": _utc(),
            "pins": [],
            "facts": [],
            "hidden_paths": [],
            "summary": None,
            "project": {},
        }
    return json.loads(path.read_text(encoding="utf-8"))


def save_overlay(code: str, overlay: dict[str, Any], cfg_obj: Config | None = None) -> Path:
    overlay["project_code"] = code
    overlay["updated_at"] = _utc()
    path = _overlay_path(code, cfg_obj)
    path.write_text(json.dumps(overlay, indent=2), encoding="utf-8")
    return path


def list_wiki_projects(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT project_id, COUNT(*) AS file_count, "
        "SUM(CASE WHEN status = 'INDEXED' THEN 1 ELSE 0 END) AS indexed_count "
        "FROM files WHERE project_id IS NOT NULL AND project_id != '' "
        "AND project_id != 'Unknown' GROUP BY project_id ORDER BY file_count DESC"
    ).fetchall()
    out = []
    for r in rows:
        code = r["project_id"]
        ov = load_overlay(code)
        pin_n = len(ov.get("pins") or [])
        fact_n = len(ov.get("facts") or [])
        out.append(
            {
                "project_id": code,
                "file_count": r["file_count"],
                "indexed_count": r["indexed_count"],
                "wiki_pins": pin_n,
                "wiki_facts": fact_n,
                "status": "curated" if pin_n >= 1 else "auto-draft",
            }
        )
    return out


def _classify(path: str, title: str, bucket: str) -> str:
    blob = f"{path} {title}".lower()
    if bucket == "drawings" or any(k in blob for k in ("dwg", "plan", "section", "elevation", "ga ")):
        return "drawings_models"
    if bucket == "model" or any(k in blob for k in ("rvt", "3dm", "ifc", "skp", "nwd")):
        return "drawings_models"
    if bucket in ("correspondence", "authority") or any(
        k in blob for k in ("email", "msg", "letter", "ura", "bca", "waiver", "submission")
    ):
        return "correspondence"
    return "key_documents"


def _suggest_role(path: str, title: str, bucket: str) -> str:
    blob = f"{path} {title}".lower()
    if bucket == "overview" or "presentation" in blob or "brief" in blob:
        return "overview_deck"
    if "report" in blob:
        return "design_report"
    if bucket == "drawings" or "ga" in blob:
        return "authoritative_ga"
    if bucket == "authority" or any(k in blob for k in ("ura", "bca", "submission")):
        return "authority_pack"
    if bucket == "model":
        return "model_master"
    return "other"


def _auto_candidates(code: str, cfg_obj: Config, conn: sqlite3.Connection) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for bucket, _query in SEED_QUERIES:
        try:
            terms = _QUALITY_TERMS[bucket]
            clauses = " OR ".join(
                "LOWER(f.file_name) LIKE ?" for _ in terms
            )
            if bucket == "model":
                clauses += " OR LOWER(f.extension) IN ('.rvt', '.rfa', '.ifc', '.3dm', '.skp', '.nwd')"
            hits = conn.execute(
                f"""
                SELECT f.file_path, f.file_name, f.extension, f.content_type,
                       f.is_latest, f.is_superseded, f.file_date,
                       COALESCE((SELECT c.text FROM chunks c
                                 WHERE c.file_id = f.file_id
                                 ORDER BY c.created_at, c.ref_value LIMIT 1), '') AS snippet
                FROM files f
                WHERE f.project_id = ? AND f.status = 'INDEXED'
                  AND ({clauses})
                ORDER BY COALESCE(f.is_superseded, 0) ASC,
                         COALESCE(f.is_latest, 0) DESC,
                         COALESCE(f.file_date, '') DESC,
                         f.file_name ASC
                LIMIT 80
                """,
                (code, *(f"%{term}%" for term in terms)),
            ).fetchall()
        except Exception as exc:  # noqa: BLE001
            logger.warning("atlas seed search failed %s: %s", bucket, exc)
            continue
        for h in hits:
            h = dict(h)
            path = h.get("file_path") or ""
            if not path or path in seen_paths:
                continue
            title = h.get("file_name") or Path(path).name
            ext = Path(title).suffix.lower()
            stem = Path(title).stem.strip()
            title_key = re.sub(r"[^a-z0-9]+", " ", stem.lower()).strip()
            if (
                ext in _LOW_VALUE_EXTS
                or not title_key
                or (ext in {".jpg", ".jpeg", ".png", ".webp"} and _GENERIC_IMAGE_RE.match(stem))
            ):
                continue
            blob = title.lower()
            score = 0.35
            score += 0.035 * sum(term in blob for term in _QUALITY_TERMS[bucket])
            if ext in {".pdf", ".doc", ".docx", ".ppt", ".pptx", ".msg", ".eml"}:
                score += 0.04
            if bucket == "model" and ext in {".rvt", ".rfa", ".ifc", ".3dm", ".skp", ".nwd"}:
                score += 0.12
            if bucket == "overview" and "brief" in blob:
                score += 0.18
            if bucket == "overview" and any(term in blob for term in ("price schedule", "attachment", "form of tender")):
                score -= 0.16
            if bucket == "authority" and any(term in blob for term in ("invitation letter", "response to", "approval")):
                score += 0.1
            if bucket == "drawings" and "submission" in blob:
                score += 0.08
            if h.get("is_latest"):
                score += 0.06
            if h.get("is_superseded"):
                score -= 0.2
            if any(term in blob for term in ("gis export", "landuse", "texture", "material library", "policy")):
                score -= 0.12
            seen_paths.add(path)
            items.append(
                {
                    "title": title,
                    "path": path,
                    "snippet": re.sub(r"\s+", " ", h.get("snippet") or "")[:300],
                    "score": round(score, 4),
                    "via": f"auto:{bucket}",
                    "section": _classify(path, title, bucket),
                    "suggested_role": _suggest_role(path, title, bucket),
                    "kind": "candidate",
                    "pinned": False,
                    "title_key": title_key,
                    "seed_bucket": bucket,
                }
            )
    best_by_title: dict[str, dict[str, Any]] = {}
    for item in items:
        key = item.pop("title_key")
        existing = best_by_title.get(key)
        if existing is None or (item.get("score") or 0.0) > (existing.get("score") or 0.0):
            best_by_title[key] = item
    ranked = sorted(best_by_title.values(), key=lambda item: item.get("score") or 0.0, reverse=True)
    section_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    selected: list[dict[str, Any]] = []
    for item in ranked:
        section = item["section"]
        bucket = item.pop("seed_bucket")
        if (
            section_counts.get(section, 0) >= _SECTION_LIMIT
            or bucket_counts.get(bucket, 0) >= _BUCKET_LIMIT
        ):
            continue
        section_counts[section] = section_counts.get(section, 0) + 1
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        selected.append(item)
    return selected


def _project_profile(code: str, conn: sqlite3.Connection) -> dict[str, Any]:
    """Build a compact, factual project snapshot from indexed metadata."""
    totals = conn.execute(
        "SELECT COUNT(*) AS total, "
        "SUM(CASE WHEN status = 'INDEXED' THEN 1 ELSE 0 END) AS indexed, "
        "SUM(CASE WHEN content_type = 'PDF' THEN 1 ELSE 0 END) AS pdfs, "
        "SUM(CASE WHEN content_type IN ('CAD','BIM','Rhino','Render Scene') THEN 1 ELSE 0 END) AS models, "
        "SUM(CASE WHEN content_type = 'Image' THEN 1 ELSE 0 END) AS images "
        "FROM files WHERE project_id = ?",
        (code,),
    ).fetchone()
    types = conn.execute(
        "SELECT COALESCE(NULLIF(content_type, ''), 'Other') AS label, COUNT(*) AS count "
        "FROM files WHERE project_id = ? GROUP BY label ORDER BY count DESC LIMIT 5",
        (code,),
    ).fetchall()
    stages = conn.execute(
        "SELECT DISTINCT folder_stage FROM files WHERE project_id = ? "
        "AND folder_stage IS NOT NULL AND folder_stage != '' "
        "AND LOWER(folder_stage) != 'unknown' ORDER BY folder_stage",
        (code,),
    ).fetchall()
    return {
        "total_files": int(totals["total"] or 0),
        "indexed_files": int(totals["indexed"] or 0),
        "pdf_count": int(totals["pdfs"] or 0),
        "model_count": int(totals["models"] or 0),
        "image_count": int(totals["images"] or 0),
        "top_types": [{"label": r["label"], "count": r["count"]} for r in types],
        "stages": [r["folder_stage"] for r in stages[:6]],
    }


def _auto_summary(project: dict[str, Any], profile: dict[str, Any]) -> str:
    name = project.get("name") or project.get("code") or "This project"
    total = profile.get("total_files", 0)
    indexed = profile.get("indexed_files", 0)
    parts = [
        f"{name} is represented in Hunt by {total:,} file records, with {indexed:,} currently indexed."
    ]
    coverage = []
    if profile.get("pdf_count"):
        coverage.append(f"{profile['pdf_count']:,} PDFs")
    if profile.get("model_count"):
        coverage.append(f"{profile['model_count']:,} CAD and model files")
    if profile.get("image_count"):
        coverage.append(f"{profile['image_count']:,} images")
    if coverage:
        parts.append("The archive includes " + ", ".join(coverage) + ".")
    parts.append("This is an index snapshot; pin authoritative sources and cite facts before treating the page as published knowledge.")
    return " ".join(parts)


def draft_wiki_article(
    code: str,
    *,
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    """Generate a cited narrative draft without promoting candidates to truth."""
    from core.llm_providers import chat_completion

    cfg = cfg_obj or _module_cfg
    page = get_wiki_page(code, conn=conn, cfg_obj=cfg)
    evidence: list[dict[str, str]] = []
    for section in page.get("sections", {}).values():
        for item in section.get("items", []):
            if item.get("kind") != "candidate" or not item.get("snippet"):
                continue
            evidence.append(
                {
                    "title": item.get("title", ""),
                    "path": item.get("path", ""),
                    "excerpt": item.get("snippet", "")[:700],
                }
            )
    evidence = evidence[:16]
    if len(evidence) < 2:
        raise ValueError("Atlas needs at least two text-bearing candidates to draft an article")

    numbered = [{"source": i + 1, **item} for i, item in enumerate(evidence)]
    messages = [
        {
            "role": "system",
            "content": (
                "You are the editorial engine for an architecture project wiki. "
                "Write only claims supported by the supplied excerpts. Never infer client, location, "
                "dates, status, scope, or design intent when absent. Cite every paragraph with one or "
                "more source numbers like [1] or [2][4]. Return strict JSON only with keys overview "
                "(string), sections (array of objects with heading and body), and open_questions "
                "(array of strings). Use 3-5 useful sections and concise professional prose."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"project": page.get("project", {}), "evidence": numbered},
                ensure_ascii=False,
            ),
        },
    ]
    draft_cfg = copy(cfg)
    draft_cfg.chat_model = os.environ.get("TIGA_ATLAS_MODEL", "qwen3:8b")
    draft_cfg.ollama_timeout = max(cfg.ollama_timeout, 90)
    def parse_article(raw_text: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.I)
        first_brace, last_brace = cleaned.find("{"), cleaned.rfind("}")
        if first_brace >= 0 and last_brace > first_brace:
            cleaned = cleaned[first_brace:last_brace + 1]
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ValueError(f"LLM did not return JSON: {cleaned[:180]!r}") from exc
        if isinstance(parsed.get("sections"), dict):
            parsed["sections"] = [
                {"heading": heading, "body": body}
                for heading, body in parsed["sections"].items()
            ]
        if not isinstance(parsed.get("overview"), str) or not isinstance(parsed.get("sections"), list):
            raise ValueError("LLM returned an invalid Atlas article")
        return parsed

    raw, provider = chat_completion(messages, draft_cfg, purpose="atlas-wiki-draft")
    article = parse_article(raw)
    section_text = " ".join(
        str(s.get("body", "")) for s in article.get("sections", []) if isinstance(s, dict)
    )
    if len(article.get("sections", [])) < 2 or not re.search(r"\[\d+\]", article["overview"] + section_text):
        messages.extend(
            [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "Revise this now. Return JSON only. Include at least three non-empty section "
                        "objects and put numbered evidence citations such as [1] at the end of every "
                        "overview and section paragraph. Do not omit the sections."
                    ),
                },
            ]
        )
        raw, provider = chat_completion(messages, draft_cfg, purpose="atlas-wiki-draft-repair")
        article = parse_article(raw)
    article["sections"] = [
        {"heading": str(s.get("heading", "Section")), "body": str(s.get("body", ""))}
        for s in article["sections"][:6]
        if isinstance(s, dict) and s.get("body")
    ]
    article_text = article["overview"] + " " + " ".join(s["body"] for s in article["sections"])
    if len(article["sections"]) < 2 or not re.search(r"\[\d+\]", article_text):
        raise ValueError("LLM draft lacked cited narrative sections; no draft was saved")
    article["open_questions"] = [str(q) for q in (article.get("open_questions") or [])[:8]]
    article.update(
        {
            "provider": provider,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "sources": numbered,
            "status": "ai-draft",
        }
    )
    overlay = load_overlay(code, cfg)
    overlay["article_draft"] = article
    save_overlay(code, overlay, cfg)
    return article


def _auto_facts_from_card(card: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not card:
        return []
    facts = []
    mapping = [
        ("typology", "Typology", card.get("typology_primary")),
        ("location", "Location", card.get("location")),
        ("client", "Client", card.get("client")),
        ("stage", "Stage", card.get("stage")),
        ("gfa", "GFA (sqm)", card.get("gfa_sqm")),
    ]
    for key, label, value in mapping:
        if value in (None, "", []):
            continue
        facts.append(
            {
                "key": key,
                "label": label,
                "value": str(value),
                "cite_paths": [],  # auto facts start uncited → wiki must confirm
                "status": "draft",
                "source": "auto",
            }
        )
    return facts


def compute_health(page: dict[str, Any]) -> dict[str, Any]:
    facts = page.get("facts") or []
    pins = page.get("pins") or []
    uncited = [f.get("key") for f in facts if not (f.get("cite_paths") or [])]
    fact_n = len(facts)
    cited_n = fact_n - len(uncited)
    ratio = (cited_n / fact_n) if fact_n else 1.0
    checks = [
        {"id": "has_identity", "ok": bool(page.get("project", {}).get("code")), "detail": "Project identity"},
        {"id": "has_pin", "ok": len(pins) >= 1, "detail": f"{len(pins)} pin(s) — need ≥1 for Published"},
        {
            "id": "facts_cited",
            "ok": fact_n == 0 or ratio >= 0.8,
            "detail": f"{cited_n}/{fact_n} facts cited (target ≥80%)",
        },
        {"id": "has_summary", "ok": bool((page.get("summary") or "").strip()), "detail": "Summary present"},
    ]
    ok_n = sum(1 for c in checks if c["ok"])
    score = int(round(100 * ok_n / len(checks)))
    grade = "A" if score >= 90 else "B" if score >= 75 else "C" if score >= 60 else "D" if score >= 40 else "F"
    published = len(pins) >= 1 and len(uncited) == 0
    return {
        "score": score,
        "grade": grade,
        "checks": checks,
        "pin_count": len(pins),
        "uncited_facts": len(uncited),
        "fact_cited_ratio": round(ratio, 2),
        "published": published,
        "state": "Published" if published else "Auto-draft / needs wiki",
    }


def get_wiki_page(
    code: str,
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    cfg = cfg_obj or _module_cfg
    from core.project_card import get_project_card

    card = get_project_card(code, conn=conn)
    overlay = load_overlay(code, cfg)
    candidates = _auto_candidates(code, cfg, conn)
    profile = _project_profile(code, conn)

    project = {
        "code": code,
        "name": (card or {}).get("name") or code,
        "aliases": (card or {}).get("alt_names") or [],
        "typology": (card or {}).get("typology_primary") or "",
        "stage": (card or {}).get("stage") or "",
        "location": (card or {}).get("location") or "",
        "client": (card or {}).get("client") or "",
        "status": "auto-draft",
    }
    # wiki project patch
    for k, v in (overlay.get("project") or {}).items():
        if v not in (None, "", []):
            project[k] = v

    summary = overlay.get("summary") or _auto_summary(project, profile)

    pins = list(overlay.get("pins") or [])
    pin_paths = {p.get("path") for p in pins if p.get("path")}
    hidden = set(overlay.get("hidden_paths") or [])

    sections = {
        "key_documents": {"title": "Key documents", "items": []},
        "drawings_models": {"title": "Drawings & models", "items": []},
        "correspondence": {"title": "Correspondence & authority", "items": []},
    }
    for item in candidates:
        if item["path"] in hidden:
            continue
        if item["path"] in pin_paths:
            item = {**item, "pinned": True, "kind": "pinned"}
            for p in pins:
                if p.get("path") == item["path"]:
                    item["role"] = p.get("role")
        sections[item["section"]]["items"].append(item)

    # Auto facts then overlay facts win by key
    facts_by_key = {f["key"]: f for f in _auto_facts_from_card(card) if f.get("key")}
    for f in overlay.get("facts") or []:
        if f.get("key"):
            facts_by_key[f["key"]] = f
    facts = list(facts_by_key.values())

    # Keep archive counts separate from the project introduction. Only cited,
    # non-disputed facts can contribute claims; excerpts remain labelled drafts.
    overview = []
    for fact in facts:
        if fact.get("value") and fact.get("cite_paths") and fact.get("status") != "disputed":
            overview.append({"text": f"{fact.get('label') or fact.get('key')}: {fact['value']}",
                             "cite_paths": fact["cite_paths"], "kind": "cited fact"})
    if not overview:
        for item in sections["key_documents"]["items"]:
            if item.get("snippet") and item.get("path"):
                overview.append({"text": item["snippet"], "cite_paths": [item["path"]],
                                 "kind": "Source excerpt · needs review"})
                break

    page = {
        "schema_version": 1,
        "product": "tiga-atlas-wiki",
        "project": project,
        "summary": summary,
        "overview": overview[:6],
        "pins": pins,
        "facts": facts,
        "sections": sections,
        "open_questions": overlay.get("open_questions")
        or [
            "Pin the overview deck / GA / authority pack",
            "Add cite_paths to every auto fact you keep",
            "Hide junk candidates (Copy of…, caches)",
        ],
        "sources": {
            "mode": "hunt+wiki",
            "built_at": _utc(),
            "overlay": str(_overlay_path(code, cfg)),
            "card_found": card is not None,
            "candidate_count": len(candidates),
        },
        "profile": profile,
    }
    page["health"] = compute_health(page)
    if page["health"]["published"]:
        page["project"]["status"] = "published"
    return page


def wiki_pin(
    code: str,
    *,
    role: str,
    path: str,
    title: str = "",
    note: str = "",
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    if role not in PIN_ROLES:
        role = "other"
    ov = load_overlay(code, cfg_obj)
    pins = ov.setdefault("pins", [])
    pin = {
        "role": role,
        "path": path,
        "title": title or Path(path).name,
        "note": note or "Pinned by wiki contributor",
        "source": "wiki",
        "pinned_at": _utc(),
    }
    for i, existing in enumerate(pins):
        if existing.get("role") == role:
            pins[i] = pin
            break
    else:
        pins.append(pin)
    save_overlay(code, ov, cfg_obj)
    return pin


def wiki_hide(code: str, path: str, cfg_obj: Config | None = None) -> None:
    ov = load_overlay(code, cfg_obj)
    hidden = ov.setdefault("hidden_paths", [])
    if path not in hidden:
        hidden.append(path)
    save_overlay(code, ov, cfg_obj)


def wiki_unhide(code: str, path: str, cfg_obj: Config | None = None) -> None:
    ov = load_overlay(code, cfg_obj)
    ov["hidden_paths"] = [p for p in (ov.get("hidden_paths") or []) if p != path]
    save_overlay(code, ov, cfg_obj)


def wiki_fact(
    code: str,
    *,
    key: str,
    label: str,
    value: str,
    cite_paths: list[str] | None = None,
    status: str = "draft",
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    if status not in ("draft", "verified", "disputed"):
        status = "draft"
    fact = {
        "key": key,
        "label": label or key,
        "value": value,
        "cite_paths": cite_paths or [],
        "status": status,
        "source": "wiki",
    }
    ov = load_overlay(code, cfg_obj)
    facts = ov.setdefault("facts", [])
    for i, existing in enumerate(facts):
        if existing.get("key") == key:
            facts[i] = fact
            break
    else:
        facts.append(fact)
    save_overlay(code, ov, cfg_obj)
    return fact


def evidence_pack(page: dict[str, Any], max_n: int = 12) -> list[dict[str, Any]]:
    """Build an Ask context from human pins and explicitly cited facts only.

    Hunt candidates remain useful for page discovery, but they are proposals,
    not durable evidence, and must never cross the Einstein/Ask trust boundary
    until a contributor pins them.
    """
    pack: list[dict[str, Any]] = []
    for pin in page.get("pins") or []:
        pack.append({"kind": "pin", **{k: pin.get(k) for k in ("role", "title", "path", "note")}})
    for fact in page.get("facts") or []:
        if not fact.get("cite_paths"):
            continue
        pack.append(
            {
                "kind": "fact",
                "key": fact.get("key"),
                "label": fact.get("label"),
                "value": fact.get("value"),
                "cite_paths": fact.get("cite_paths"),
                "status": fact.get("status"),
            }
        )
    return pack[:max_n]


def wiki_compose(page: dict[str, Any], question: str) -> dict[str, Any]:
    """Einstein-lite: pins + cited facts only. API if keyed; else local stub."""
    if not (page.get("pins") or []):
        return {
            "ok": False,
            "provider": "none",
            "answer": "Ask is locked until someone pins at least one authoritative document on this project page.",
            "pack_size": 0,
        }
    pack = evidence_pack(page)
    api_key = os.environ.get("TIGA_LLM_API_KEY") or os.environ.get("TIGA_OPENAI_API_KEY")
    if not api_key:
        lines = [
            "(local Ask stub — set TIGA_LLM_API_KEY for enterprise compose)",
            f"Q: {question}",
            "",
            "Cited evidence only:",
        ]
        for item in pack:
            if item["kind"] == "fact":
                lines.append(f"- FACT {item.get('label')}: {item.get('value')} ← {item.get('cite_paths')}")
            elif item["kind"] == "pin":
                lines.append(f"- PIN {item.get('role')}: {item.get('path')}")
            else:
                lines.append(f"- CANDIDATE {item.get('title')}: {item.get('path')}")
        lines.append("")
        lines.append("Answer from the evidence above only. Do not invent.")
        return {"ok": True, "provider": "local-stub", "answer": "\n".join(lines), "pack_size": len(pack)}

    base = os.environ.get("TIGA_LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.environ.get("TIGA_LLM_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "temperature": 0.1,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are TIGA Einstein. Answer ONLY from the evidence pack. "
                    "Cite paths. If unknown, say you don't know."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {"question": question, "evidence_pack": pack},
                    ensure_ascii=False,
                ),
            },
        ],
    }
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        answer = data["choices"][0]["message"]["content"]
        return {"ok": True, "provider": "api", "model": model, "answer": answer, "pack_size": len(pack)}
    except (urllib.error.URLError, TimeoutError, KeyError, IndexError, json.JSONDecodeError) as exc:
        return {"ok": False, "provider": "api", "answer": f"Ask failed: {exc}", "pack_size": len(pack)}
