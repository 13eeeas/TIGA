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
        "SELECT COALESCE(project_id, 'Unknown') AS project_id, COUNT(*) AS file_count "
        "FROM files GROUP BY project_id ORDER BY file_count DESC"
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

    summary = overlay.get("summary") or (
        f"Auto-draft Grokopedia page for {project['name']}. "
        "Hunt proposed candidates — confirm pins and cite facts like a wiki."
    )

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

    page = {
        "schema_version": 1,
        "product": "tiga-atlas-wiki",
        "project": project,
        "summary": summary,
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
