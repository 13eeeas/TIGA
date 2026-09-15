"""
core/atlas_wiki.py — Cited project wiki inside TIGA Hunt.

North star: Notion-like browse + Wikipedia-like contribution + Grokopedia-like
auto draft — every durable claim needs a citation path.

Layers
------
  auto   : drafted from project_card + Hunt search candidates (machine)
  wiki   : human pins / facts / hidden paths in tiga_work/atlas/*.overlay.json
  gate   : Published only when curated identity fields + ≥1 pin + cited facts exist

Public API
----------
  list_wiki_projects(conn) -> list
  get_wiki_page(code, conn, cfg) -> dict   # auto + wiki merge
  wiki_pin / wiki_hide / wiki_fact / wiki_unhide / wiki_overview
  wiki_compose(code, question, conn, cfg) -> dict  # Einstein-lite stub/API
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import cfg as _module_cfg, Config

logger = logging.getLogger(__name__)

SEED_QUERIES = [
    ("overview", "overview brief presentation"),
    ("authority", "URA BCA submission approval waiver"),
    ("drawings", "GA plan section elevation drawing"),
    ("model", "Rhino Revit BIM model"),
    ("correspondence", "email letter consultant client"),
]

PIN_ROLES = (
    "overview_deck",
    "design_report",
    "authoritative_ga",
    "authority_pack",
    "model_master",
    "other",
)

# Identity fields for the <5s wiki blurb. All four must be filled (not
# "Needs curation") before a page can show Published / a health score.
CURATED_FIELDS = ("typology", "client", "stage", "location")
OVERVIEW_FIELDS = ("name", "typology", "client", "stage", "location")
_EMPTY_TOKENS = {
    "",
    "needs curation",
    "unknown",
    "n/a",
    "na",
    "none",
    "—",
    "-",
    "tbd",
}
_AUTO_SUMMARY_MARKERS = (
    "auto-draft grokopedia",
    "hunt proposed candidates",
    "confirm pins and cite facts",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(code: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", code.lower()).strip("-") or "project"


def field_is_filled(value: Any) -> bool:
    """True when a curated identity field is a real value, not a placeholder."""
    if value is None:
        return False
    text = str(value).strip()
    if not text:
        return False
    return text.lower() not in _EMPTY_TOKENS


def is_human_summary(summary: str | None) -> bool:
    """True when summary is staff-written, not the Grokopedia auto placeholder."""
    text = (summary or "").strip()
    if not text:
        return False
    lowered = text.lower()
    return not any(marker in lowered for marker in _AUTO_SUMMARY_MARKERS)


def missing_curated_fields(project: dict[str, Any] | None) -> list[str]:
    proj = project or {}
    return [key for key in CURATED_FIELDS if not field_is_filled(proj.get(key))]


def compose_blurb(project: dict[str, Any] | None, summary: str | None = None) -> str:
    """One short wiki sentence: what / client or typology / stage / location."""
    if is_human_summary(summary):
        return str(summary).strip()

    proj = project or {}
    name = str(proj.get("name") or proj.get("code") or "This project").strip()
    typology = str(proj.get("typology") or "").strip() if field_is_filled(proj.get("typology")) else ""
    client = str(proj.get("client") or "").strip() if field_is_filled(proj.get("client")) else ""
    stage = str(proj.get("stage") or "").strip() if field_is_filled(proj.get("stage")) else ""
    location = str(proj.get("location") or "").strip() if field_is_filled(proj.get("location")) else ""

    if not any((typology, client, stage, location)):
        return (
            f"{name} — Needs curation. Add what this job is, client or typology, "
            "stage, and location if known."
        )

    if typology and client:
        lead = f"{name} is a {typology} project for {client}"
    elif typology:
        lead = f"{name} is a {typology} project"
    elif client:
        lead = f"{name} is a project for {client}"
    else:
        lead = name

    extras: list[str] = []
    if stage:
        extras.append(f"currently at {stage}")
    if location:
        extras.append(f"in {location}")
    if extras:
        return f"{lead}, {', '.join(extras)}."
    return f"{lead}."


def _project_from_card_and_overlay(
    code: str,
    card: dict[str, Any] | None,
    overlay: dict[str, Any],
) -> dict[str, Any]:
    project = {
        "code": code,
        "name": (card or {}).get("name") or code,
        "aliases": (card or {}).get("alt_names") or [],
        "typology": (card or {}).get("typology_primary") or "",
        "stage": (card or {}).get("stage") or "",
        "location": (card or {}).get("location") or "",
        "client": (card or {}).get("client") or "",
        "status": "needs-curation",
    }
    for key, value in (overlay.get("project") or {}).items():
        if value not in (None, "", []):
            project[key] = value
    return project


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


def list_wiki_projects(
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,
) -> list[dict[str, Any]]:
    from core.project_card import get_project_card

    rows = conn.execute(
        "SELECT COALESCE(project_id, 'Unknown') AS project_id, COUNT(*) AS file_count "
        "FROM files GROUP BY project_id ORDER BY file_count DESC"
    ).fetchall()
    out = []
    for r in rows:
        code = r["project_id"]
        ov = load_overlay(code, cfg_obj)
        card = get_project_card(code, conn=conn)
        project = _project_from_card_and_overlay(code, card, ov)
        facts_by_key = {f["key"]: f for f in _auto_facts_from_card(card) if f.get("key")}
        for fact in ov.get("facts") or []:
            if fact.get("key"):
                facts_by_key[fact["key"]] = fact
        page = {
            "project": project,
            "summary": ov.get("summary"),
            "pins": ov.get("pins") or [],
            "facts": list(facts_by_key.values()),
        }
        health = compute_health(page)
        blurb = compose_blurb(project, ov.get("summary"))
        pin_n = len(ov.get("pins") or [])
        fact_n = len(ov.get("facts") or [])
        out.append(
            {
                "project_id": code,
                "name": project.get("name") or code,
                "blurb": blurb,
                "file_count": r["file_count"],
                "wiki_pins": pin_n,
                "wiki_facts": fact_n,
                "missing_fields": health.get("missing_fields") or [],
                "published": health["published"],
                "status": "published" if health["published"] else "needs-curation",
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
    from core.query import search

    items: list[dict[str, Any]] = []
    seen: set[str] = set()
    for bucket, q in SEED_QUERIES:
        try:
            hits = search(
                f"{code} {q}",
                top_k=5,
                filters={"project_id": code},
                cfg_obj=cfg_obj,
                conn=conn,
                use_vector=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("atlas seed search failed %s: %s", bucket, exc)
            continue
        for h in hits:
            path = h.get("file_path") or h.get("rel_path") or ""
            if not path or path in seen:
                continue
            seen.add(path)
            title = h.get("file_name") or Path(path).name
            items.append(
                {
                    "title": title,
                    "path": path,
                    "snippet": (h.get("snippet") or "")[:300],
                    "score": h.get("final_score"),
                    "via": f"auto:{bucket}",
                    "section": _classify(path, title, bucket),
                    "suggested_role": _suggest_role(path, title, bucket),
                    "kind": "candidate",
                    "pinned": False,
                }
            )
    return items


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
    """Published only with curated identity + ≥1 pin + cited facts.

    Empty / Needs-curation cards must never report Published or 100/100.
    File counts are not part of this gate.
    """
    facts = page.get("facts") or []
    pins = page.get("pins") or []
    project = page.get("project") or {}
    missing = missing_curated_fields(project)
    uncited = [f.get("key") for f in facts if not (f.get("cite_paths") or [])]
    cited_n = sum(1 for f in facts if f.get("cite_paths"))
    fact_n = len(facts)
    ratio = (cited_n / fact_n) if fact_n else 0.0
    fields_ok = not missing
    pin_ok = len(pins) >= 1
    facts_ok = cited_n >= 1 and len(uncited) == 0
    blurb_ok = is_human_summary(page.get("summary")) or fields_ok
    checks = [
        {
            "id": "curated_fields",
            "ok": fields_ok,
            "detail": (
                "Identity fields filled"
                if fields_ok
                else "Needs curation: " + ", ".join(missing)
            ),
        },
        {
            "id": "has_pin",
            "ok": pin_ok,
            "detail": f"{len(pins)} pin(s) — need ≥1 for Published",
        },
        {
            "id": "cited_facts",
            "ok": facts_ok,
            "detail": (
                f"{cited_n}/{fact_n} facts cited — need ≥1 cited fact and none uncited"
            ),
        },
        {
            "id": "readable_blurb",
            "ok": blurb_ok,
            "detail": "Wiki blurb readable in a glance",
        },
    ]
    ok_n = sum(1 for c in checks if c["ok"])
    published = fields_ok and pin_ok and facts_ok
    score = int(round(100 * ok_n / len(checks))) if checks else 0
    if published:
        score = 100
    elif score >= 100:
        # Guard: never emit a perfect score on a non-published page.
        score = 75
    grade = None
    if published:
        grade = "A" if score >= 90 else "B" if score >= 75 else "C"
    return {
        "score": score,
        "grade": grade,
        "checks": checks,
        "pin_count": len(pins),
        "uncited_facts": len(uncited),
        "cited_facts": cited_n,
        "fact_cited_ratio": round(ratio, 2),
        "published": published,
        "show_health_score": published,
        "missing_fields": missing,
        "state": "Published" if published else "Needs curation",
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
    project = _project_from_card_and_overlay(code, card, overlay)

    human_summary = overlay.get("summary")
    if not is_human_summary(human_summary):
        human_summary = None
    blurb = compose_blurb(project, human_summary)

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

    file_row = conn.execute(
        "SELECT COUNT(*) AS n FROM files WHERE COALESCE(project_id, 'Unknown') = ?",
        (code,),
    ).fetchone()
    file_count = int(file_row["n"] if file_row is not None else 0)

    page = {
        "schema_version": 1,
        "product": "tiga-atlas-wiki",
        "project": project,
        "summary": human_summary or "",
        "blurb": blurb,
        "pins": pins,
        "facts": facts,
        "sections": sections,
        "open_questions": overlay.get("open_questions")
        or [
            "Fill the overview blurb (what / client or typology / stage / location)",
            "Pin the overview deck / GA / authority pack",
            "Add cite_paths to every auto fact you keep",
            "Hide junk candidates (Copy of…, caches)",
        ],
        "index": {
            "file_count": file_count,
            "candidate_count": len(candidates),
            "pin_count": len(pins),
            "hidden_count": len(hidden),
            "overlay": str(_overlay_path(code, cfg)),
        },
        "sources": {
            "mode": "hunt+wiki",
            "built_at": _utc(),
            "overlay": str(_overlay_path(code, cfg)),
            "card_found": card is not None,
            "candidate_count": len(candidates),
        },
    }
    page["health"] = compute_health(page)
    if page["health"]["published"]:
        page["project"]["status"] = "published"
    else:
        page["project"]["status"] = "needs-curation"
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


def wiki_overview(
    code: str,
    *,
    summary: str | None = None,
    project_fields: dict[str, Any] | None = None,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    """Save the human wiki blurb + identity fields into the overlay."""
    ov = load_overlay(code, cfg_obj)
    if summary is not None:
        text = summary.strip()
        ov["summary"] = text or None
    proj = ov.setdefault("project", {})
    for key, value in (project_fields or {}).items():
        if key not in OVERVIEW_FIELDS:
            continue
        if isinstance(value, str):
            value = value.strip()
        if value in (None, ""):
            proj.pop(key, None)
        else:
            proj[key] = value
    save_overlay(code, ov, cfg_obj)
    return {
        "summary": ov.get("summary"),
        "project": dict(proj),
        "overlay": str(_overlay_path(code, cfg_obj)),
    }


def evidence_pack(page: dict[str, Any], max_n: int = 12) -> list[dict[str, Any]]:
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
    for key in ("key_documents", "drawings_models", "correspondence"):
        for item in (page.get("sections", {}).get(key, {}) or {}).get("items") or []:
            if item.get("pinned"):
                continue
            pack.append(
                {
                    "kind": "candidate",
                    "title": item.get("title"),
                    "path": item.get("path"),
                    "snippet": (item.get("snippet") or "")[:240],
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
