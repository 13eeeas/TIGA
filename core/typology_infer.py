"""Infer a project typology from indexed brief text.

Used at the end of indexing. A saved typology on the project page is left alone.
"""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from config import Config

# Cue words that show up in architecture briefs. A label needs at least two hits.
_CUES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("education", ("university", "school", "campus", "student", "faculty", "nus", "suss")),
    ("hospitality", ("hotel", "hospitality", "guest room", "guest experience", "resort")),
    ("residential", ("residential", "residence", "apartment", "dwelling", "condominium")),
    ("healthcare", ("hospital", "healthcare", "clinic", "patient ward")),
    ("sports", ("stadium", "sports hall", "sport centre", "sport center", "sportsg")),
    ("parks", ("nature park", "nparks", "parkland", "botanic")),
    ("workplace", ("workplace", "office tower", "headquarters")),
    ("civic", ("civic", "museum", "library", "community centre", "community center")),
)


def infer_typology(text: str, name: str = "") -> str | None:
    """Return a typology label, or None when the brief and name do not say.

    A word in the project name counts double. Brief text is usually slide
    labels, so the folder name is part of the evidence.
    """
    blob = (text or "").replace("\x00", " ").casefold()
    name_l = (name or "").casefold()
    if not blob.strip() and not name_l.strip():
        return None
    scores: dict[str, int] = {}
    for label, words in _CUES:
        score = sum(blob.count(word) for word in words)
        score += sum(2 for word in words if word in name_l)
        scores[label] = score
    ordered = sorted(scores.values(), reverse=True)
    if ordered[0] < 2:
        return None
    if len(ordered) > 1 and ordered[0] == ordered[1]:
        return None
    return max(scores, key=scores.get)


def _brief_rank(name: str, path: str) -> int:
    """Design briefs outrank supplier packages that happen to say 'brief'."""
    folded = f"{name}\n{path}".replace("\\", "/").casefold()
    if any(skip in folded for skip in ("/supplier/", "/artist", "lift brief", "lasvit")):
        return 0
    if "main contract design brief" in folded or "design brief" in folded:
        return 3
    if "competition booklet" in folded or "project brief" in folded:
        return 2
    if "brief" in folded:
        return 1
    return 0


def brief_text_for_project(conn: sqlite3.Connection, project_id: str, *, limit: int = 8) -> str:
    """Pull text from the project design brief, not from a supplier brief."""
    files = conn.execute(
        """
        SELECT file_id, file_name, file_path
        FROM files
        WHERE project_id = ?
          AND (
            lower(file_name) LIKE '%brief%'
            OR lower(file_name) LIKE '%competition booklet%'
            OR lower(replace(file_path, '\\', '/')) LIKE '%/brief/%'
          )
        """,
        (project_id,),
    ).fetchall()
    ranked: list[tuple[int, str]] = []
    for row in files:
        file_id = row["file_id"] if isinstance(row, sqlite3.Row) else row[0]
        name = row["file_name"] if isinstance(row, sqlite3.Row) else row[1]
        path = row["file_path"] if isinstance(row, sqlite3.Row) else row[2]
        rank = _brief_rank(str(name or ""), str(path or ""))
        if rank:
            ranked.append((rank, str(file_id)))
    ranked.sort(key=lambda item: item[0], reverse=True)
    picked = [file_id for _rank, file_id in ranked[:4]]
    if not picked:
        return ""
    marks = ",".join("?" for _ in picked)
    rows = conn.execute(
        f"SELECT text FROM chunks WHERE file_id IN ({marks}) LIMIT 40",
        tuple(picked),
    ).fetchall()
    parts: list[str] = []
    for row in rows:
        text = (row["text"] if isinstance(row, sqlite3.Row) else row[0]) or ""
        text = str(text).replace("\x00", " ").strip()
        if text and "no text extracted" not in text.casefold():
            parts.append(text[:2000])
    parts.sort(
        key=lambda text: any(
            word in text.casefold() for word in ("developer", "client", "employer", "changi", "singapore")
        ),
        reverse=True,
    )
    return "\n".join(parts[:limit])


_CLIENT_LABEL = re.compile(
    r"(?i)\b(developer(?:\s*&\s*sub\s*-?\s*lessee)?|client|employer)\s*[:\-]\s*([^\n|]{3,90})"
)
_SKIP_CLIENT = ("woha", "architect", "contractor", "to be confirmed", "tbc", "n/a")


def _clean_party(value: str) -> str | None:
    text = re.sub(r"\s+", " ", value).strip(" .:-")
    if len(text) < 3 or any(skip in text.casefold() for skip in _SKIP_CLIENT):
        return None
    return text[:80]


def client_from_brief(text: str) -> str | None:
    """Read the developer, client, or employer from the brief title block."""
    for match in _CLIENT_LABEL.finditer(text or ""):
        party = _party_after_care_of(match.group(2)) or _clean_party(match.group(2))
        if party:
            return party
    lines = [re.sub(r"\s+", " ", line).strip(" :") for line in (text or "").splitlines()]
    lines = [line for line in lines if line]
    role = re.compile(r"(?i)^(developer|client|employer)$")
    org = re.compile(r"(?i)\b(pte\.?\s*ltd\.?|limited|trust|group|authority|university)\b")
    for index, line in enumerate(lines):
        if not role.match(line):
            continue
        window = lines[index + 1 : index + 8]
        joined = " ".join(window)
        party = _party_after_care_of(joined)
        if party:
            return party
        for nxt in window:
            if org.search(nxt):
                party = _clean_party(nxt)
                if party:
                    return party
    return None


def _party_after_care_of(value: str) -> str | None:
    parts = re.split(r"(?i)\bc/o\b", value or "", maxsplit=1)
    if len(parts) < 2:
        return None
    return _clean_party(parts[1])


_CLIENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("NUS", ("nus",)),
    ("SUSS", ("suss",)),
    ("NParks", ("nparks", "national parks")),
    ("SportSG", ("sportsg", "sport singapore")),
    ("Keppel", ("keppel",)),
    ("Temasek", ("temasek",)),
)

# Later stages first. A job that has reached tender should not stay labelled concept.
_STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("construction", ("/construction/", "as-built", "as built")),
    ("tender", ("stage 2", "/tender/", "ariba")),
    ("schematic", ("schematic", "stage 1")),
    ("concept", ("/concept/",)),
)


# More specific places first. "Singapore" alone is the fallback.
_PLACES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Changi Airport", ("changi airport", "changi")),
    ("Orchard", ("orchard",)),
    ("Marina Bay", ("marina bay",)),
    ("Rochor", ("rochor",)),
    ("Keppel Bay", ("keppel bay",)),
    ("Sentosa", ("sentosa",)),
    ("Paya Lebar", ("paya lebar",)),
)


def infer_client(name: str, text: str = "") -> str | None:
    """Client from a labelled brief line, otherwise from the project name."""
    labelled = client_from_brief(text or "")
    if labelled:
        return labelled
    name_l = (name or "").casefold()
    for label, words in _CLIENTS:
        if any(word in name_l for word in words):
            return label
    return None


def infer_location(name: str, text: str = "") -> str | None:
    """Place named in the opening of the brief. The project name is not a site."""
    del name
    blob = (text or "").replace("\x00", " ")[:1800].casefold()
    place = None
    for label, words in _PLACES:
        if any(word in blob for word in words):
            place = label
            break
    named_singapore = "singapore" in blob
    if place and "singapore" not in place.casefold():
        return f"{place}, Singapore" if named_singapore or place else place
    if place:
        return place
    if named_singapore:
        return "Singapore"
    return None


def infer_stage_from_paths(paths: list[str]) -> str | None:
    """Latest design stage that actually appears in file paths."""
    folded = [p.replace("\\", "/").casefold() for p in paths]
    for label, needles in _STAGES:
        hits = sum(1 for path in folded if any(needle in path for needle in needles))
        if hits >= 3:
            return label
    return None


def _paths_for_project(conn: sqlite3.Connection, project_id: str, *, limit: int = 8000) -> list[str]:
    rows = conn.execute(
        "SELECT file_path FROM files WHERE project_id = ? LIMIT ?",
        (project_id, limit),
    ).fetchall()
    out: list[str] = []
    for row in rows:
        path = row["file_path"] if isinstance(row, sqlite3.Row) else row[0]
        if path:
            out.append(str(path))
    return out


def _locked(project: dict[str, Any], field: str) -> bool:
    return str(project.get(f"{field}_source") or "") in {"human", "llm"}


def apply_index_profile(
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    """Fill empty project fields from the brief, the name, and the folder paths.

    Staff edits (source human) are left alone. The blurb is rewritten from
    those fields unless someone saved their own sentence.
    """
    from core.atlas_wiki import compose_blurb, load_overlay, save_overlay

    rows = conn.execute(
        "SELECT DISTINCT project_id FROM files "
        "WHERE COALESCE(project_id, 'Unknown') != 'Unknown'"
    ).fetchall()
    filled: list[str] = []
    skipped = 0
    for row in rows:
        code = row["project_id"] if isinstance(row, sqlite3.Row) else row[0]
        overlay = load_overlay(code, cfg_obj)
        project = overlay.setdefault("project", {})
        if all(_locked(project, field) for field in ("typology", "client", "stage", "location")):
            if str(overlay.get("summary_source") or "") == "human":
                skipped += 1
                continue
        text = brief_text_for_project(conn, code)
        changed: list[str] = []

        if not _locked(project, "typology"):
            named = infer_typology("", name=str(code))
            label = named or infer_typology(text, name=str(code))
            if label:
                project["typology"] = label
                project["typology_source"] = "index"
                changed.append(f"typology={label}")
            elif project.get("typology") and project.get("typology_source") != "human":
                project.pop("typology", None)
                project.pop("typology_source", None)

        if not _locked(project, "client"):
            client = infer_client(str(code), text)
            if client:
                project["client"] = client
                project["client_source"] = "index"
                changed.append(f"client={client}")
            elif project.get("client_source") == "index":
                project.pop("client", None)
                project.pop("client_source", None)
                changed.append("client cleared")

        if not _locked(project, "location"):
            location = infer_location(str(code), text[:1800])
            if location:
                project["location"] = location
                project["location_source"] = "index"
                changed.append(f"location={location}")
            elif project.get("location_source") == "index":
                project.pop("location", None)
                project.pop("location_source", None)
                changed.append("location cleared")

        if not _locked(project, "stage"):
            stage = infer_stage_from_paths(_paths_for_project(conn, code))
            if stage:
                project["stage"] = stage
                project["stage_source"] = "index"
                changed.append(f"stage={stage}")

        summary_locked = str(overlay.get("summary_source") or "") in {"human", "llm"}
        if not summary_locked and any(
            project.get(key) for key in ("typology", "client", "stage", "location")
        ):
            project["name"] = project.get("name") or code
            project["code"] = code
            overlay["summary"] = compose_blurb(project, None)
            overlay["summary_source"] = "index"
            changed.append("blurb")

        if changed:
            save_overlay(code, overlay, cfg_obj)
            filled.append(f"{code}: {', '.join(changed)}")
    return {"filled": filled, "skipped_existing": skipped}


def apply_brief_typologies(
    conn: sqlite3.Connection,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    """Back-compat name. Indexing fills the whole project card, not only typology."""
    return apply_index_profile(conn, cfg_obj)
