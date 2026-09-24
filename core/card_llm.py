"""Rewrite an Atlas card from the design brief with the local chat model.

Staff edits (source human) stay. A missing or empty model field keeps the
rule-filled value.
"""

from __future__ import annotations

import json
import re
import sqlite3
from typing import Any, Callable

from config import Config

_TYPOLOGY = {
    "education",
    "hospitality",
    "residential",
    "healthcare",
    "sports",
    "parks",
    "workplace",
    "civic",
}
_STAGE = {"concept", "schematic", "tender", "construction"}
_SKIP_CLIENT = ("woha", "architect", "unknown", "n/a", "tbd")


def parse_card_json(raw: str) -> dict[str, str]:
    """Pull one JSON object out of a model reply."""
    text = re.sub(r"<think>.*?</think>", "", raw or "", flags=re.DOTALL).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict[str, str] = {}
    for key in ("client", "typology", "stage", "location", "blurb"):
        value = data.get(key)
        if value is None:
            continue
        cleaned = re.sub(r"\s+", " ", str(value)).strip(" .")
        if cleaned and cleaned.casefold() not in {"null", "none", "unknown", "n/a"}:
            out[key] = cleaned
    return out


def accept_card_fields(parsed: dict[str, str]) -> dict[str, str]:
    """Drop values outside the card vocabulary."""
    kept: dict[str, str] = {}
    typology = parsed.get("typology", "").casefold()
    if typology in _TYPOLOGY:
        kept["typology"] = typology
    stage = parsed.get("stage", "").casefold()
    if stage in _STAGE:
        kept["stage"] = stage
    client = parsed.get("client", "")
    if client and not any(skip in client.casefold() for skip in _SKIP_CLIENT) and len(client) <= 80:
        kept["client"] = client
    location = parsed.get("location", "")
    if location and len(location) <= 60:
        kept["location"] = location
    blurb = parsed.get("blurb", "")
    if 12 <= len(blurb) <= 280 and "as an ai" not in blurb.casefold():
        kept["blurb"] = blurb if blurb.endswith(".") else blurb + "."
    return kept


def _ollama_card(messages: list[dict[str, str]], cfg: Config) -> str:
    from core.llm_providers import _http_json

    url = cfg.ollama_base_url.rstrip("/") + "/api/chat"
    data = _http_json(
        url,
        {
            "model": cfg.chat_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"num_ctx": 4096, "temperature": 0.1},
        },
        {},
        cfg.ollama_timeout,
    )
    return str((data.get("message") or {}).get("content") or "")


def improve_project_card(
    code: str,
    brief: str,
    project: dict[str, Any],
    *,
    complete: Callable[[list[dict[str, str]]], str],
) -> dict[str, str]:
    """Ask the model for card fields. Returns only accepted keys."""
    current = {
        "name": project.get("name") or code,
        "typology": project.get("typology") or "",
        "client": project.get("client") or "",
        "stage": project.get("stage") or "",
        "location": project.get("location") or "",
    }
    messages = [
        {
            "role": "system",
            "content": (
                "You fill an architecture project card from the brief. "
                "Use only facts written in the brief. "
                "Reply with JSON keys client, typology, stage, location, blurb. "
                "typology is one of education, hospitality, residential, healthcare, "
                "sports, parks, workplace, civic, or null. "
                "stage is one of concept, schematic, tender, construction, or null. "
                "blurb is one sentence under 30 words naming the job, client, and site. "
                "Use null when the brief does not say."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"project": code, "current": current, "brief": (brief or "")[:6000]},
                ensure_ascii=False,
            ),
        },
    ]
    return accept_card_fields(parse_card_json(complete(messages)))


def improve_atlas_cards(
    conn: sqlite3.Connection,
    cfg_obj: Config,
    *,
    complete: Callable[[list[dict[str, str]]], str] | None = None,
) -> list[str]:
    """Rewrite each non-human card. Returns one line per project that changed."""
    from core.atlas_wiki import load_overlay, save_overlay
    from core.typology_infer import brief_text_for_project

    completer = complete or (lambda messages: _ollama_card(messages, cfg_obj))
    rows = conn.execute(
        "SELECT DISTINCT project_id FROM files "
        "WHERE COALESCE(project_id, 'Unknown') != 'Unknown'"
    ).fetchall()
    changed_lines: list[str] = []
    for row in rows:
        code = row["project_id"] if isinstance(row, sqlite3.Row) else row[0]
        overlay = load_overlay(code, cfg_obj)
        if str(overlay.get("summary_source") or "") == "human":
            continue
        project = overlay.setdefault("project", {})
        brief = brief_text_for_project(conn, code)
        if not brief.strip():
            continue
        fields = improve_project_card(code, brief, project, complete=completer)
        notes: list[str] = []
        for key in ("typology", "client", "stage", "location"):
            if key not in fields:
                continue
            if str(project.get(f"{key}_source") or "") == "human":
                continue
            if project.get(key) == fields[key]:
                continue
            project[key] = fields[key]
            project[f"{key}_source"] = "llm"
            notes.append(f"{key}={fields[key]}")
        if "blurb" in fields:
            overlay["summary"] = fields["blurb"]
            overlay["summary_source"] = "llm"
            notes.append("blurb")
        if notes:
            project["name"] = project.get("name") or code
            project["code"] = code
            save_overlay(code, overlay, cfg_obj)
            changed_lines.append(f"{code}: {', '.join(notes)}")
    return changed_lines
