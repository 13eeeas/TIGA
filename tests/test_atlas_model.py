"""tests/test_atlas_model.py — Ticket A structured rows + page assembly."""

from __future__ import annotations

from pathlib import Path

import yaml

from config import load_config
from core.atlas_model import (
    SCHEMA_VERSION,
    assemble_from_rows,
    hunt_document_proposals,
    merge_document_proposals,
    model_answers_project_questions,
    normalize_overlay,
    set_lifecycle,
    upsert_decision,
    upsert_document,
    upsert_precedent,
    upsert_strategy,
    upsert_team_member,
)
from core.atlas_wiki import get_wiki_page, load_overlay, wiki_overview, wiki_pin
from core.db import get_connection, upsert_file
from unittest.mock import patch


def _cfg(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.dump({"index_roots": [str(tmp_path / "archive")]}),
        encoding="utf-8",
    )
    return load_config(config_file=cfg_file, work_dir=tmp_path)


def test_normalize_lifts_pins_to_authoritative_documents() -> None:
    ov = normalize_overlay(
        "261",
        {
            "schema_version": 1,
            "pins": [
                {
                    "role": "overview_deck",
                    "path": "/archive/261/deck.pdf",
                    "title": "Deck",
                    "source": "wiki",
                }
            ],
            "facts": [],
            "project": {"stage": "design"},
        },
    )
    assert ov["schema_version"] == SCHEMA_VERSION
    assert ov["lifecycle"]["stage"] == "design"
    assert len(ov["documents"]) == 1
    assert ov["documents"][0]["authority"] == "authoritative"
    assert ov["documents"][0]["path"] == "/archive/261/deck.pdf"


def test_assemble_from_rows_is_source_of_truth() -> None:
    ov = normalize_overlay("261", None)
    upsert_document(
        ov,
        path="/archive/261/ga.pdf",
        title="GA",
        authority="authoritative",
        role="authoritative_ga",
    )
    upsert_document(
        ov,
        path="/archive/261/ga_old.pdf",
        title="GA old",
        authority="superseded",
        role="authoritative_ga",
        superseded_by=["/archive/261/ga.pdf"],
    )
    upsert_strategy(ov, tag="biophilic", label="Biophilic façade", cite_paths=["/archive/261/ga.pdf"])
    upsert_team_member(ov, role="PM", name="Ada", org="WOHA")
    upsert_decision(
        ov,
        title="Lock façade material",
        status="accepted",
        cite_paths=["/archive/261/ga.pdf"],
    )
    upsert_precedent(ov, project_code="NUS", relation="precedent", note="Campus typology")
    set_lifecycle(ov, "construction")

    assembled = assemble_from_rows(
        code="261",
        overlay=ov,
        project={
            "code": "261",
            "name": "Tianmu",
            "typology": "residential",
            "client": "Far East",
            "stage": "construction",
            "location": "Taipei",
        },
        blurb="Tianmu is a residential project for Far East in Taipei.",
        facts=[
            {
                "key": "client",
                "label": "Client",
                "value": "Far East",
                "cite_paths": ["/archive/261/brief.pdf"],
            }
        ],
    )

    assert assembled["source_of_truth"] == "structured_rows"
    assert assembled["identity"]["name"] == "Tianmu"
    assert assembled["lifecycle"]["stage"] == "construction"
    assert assembled["strategies"][0]["tag"] == "biophilic"
    assert assembled["team"][0]["name"] == "Ada"
    assert assembled["decisions"][0]["title"].startswith("Lock")
    assert assembled["documents"]["authoritative"][0]["path"].endswith("ga.pdf")
    assert assembled["documents"]["superseded"][0]["path"].endswith("ga_old.pdf")
    assert assembled["related"][0]["project_code"] == "NUS"
    assert assembled["data"]["cited_facts"]

    ready = model_answers_project_questions(assembled)
    assert ready["what_is_this_project"] is True
    assert ready["what_superseded_what"] is True
    assert ready["has_authority_docs"] is True


def test_hunt_proposals_do_not_overwrite_wiki_docs(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    conn = get_connection(cfg.get_db_path())
    upsert_file(
        conn,
        {
            "file_id": "f1",
            "file_path": "/archive/261/ga.pdf",
            "file_name": "ga.pdf",
            "project_id": "261",
        },
    )
    upsert_file(
        conn,
        {
            "file_id": "f2",
            "file_path": "/archive/261/ga_old.pdf",
            "file_name": "ga_old.pdf",
            "project_id": "261",
        },
    )
    conn.execute(
        "UPDATE files SET is_latest = 1, is_superseded = 0 WHERE file_id = 'f1'"
    )
    conn.execute(
        "UPDATE files SET is_latest = 0, is_superseded = 1 WHERE file_id = 'f2'"
    )
    conn.commit()

    proposals = hunt_document_proposals(conn, "261")
    conn.close()
    assert len(proposals) == 2
    assert all(p["source"] == "hunt-proposal" for p in proposals)
    assert {p["authority"] for p in proposals} >= {"authoritative", "superseded"}

    ov = normalize_overlay("261", None)
    upsert_document(
        ov,
        path="/archive/261/ga.pdf",
        authority="authoritative",
        source="wiki",
        note="Human confirmed",
    )
    merge_document_proposals(ov, proposals)
    wiki_doc = next(d for d in ov["documents"] if d["path"].endswith("ga.pdf"))
    assert wiki_doc["source"] == "wiki"
    assert wiki_doc["note"] == "Human confirmed"
    # Superseded path only existed as proposal → kept as proposal row.
    old = next(d for d in ov["documents"] if d["path"].endswith("ga_old.pdf"))
    assert old["source"] == "hunt-proposal"


def test_get_wiki_page_exposes_model(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    conn = get_connection(cfg.get_db_path())
    upsert_file(
        conn,
        {
            "file_id": "f1",
            "file_path": "/archive/261/brief.pdf",
            "file_name": "brief.pdf",
            "project_id": "261",
        },
    )
    conn.execute(
        "UPDATE files SET is_latest = 1, is_superseded = 0 WHERE file_id = 'f1'"
    )
    conn.commit()
    wiki_pin("261", role="overview_deck", path="/archive/261/brief.pdf", cfg_obj=cfg)
    wiki_overview(
        "261",
        summary="Tianmu residential for Far East in Taipei.",
        project_fields={
            "name": "Tianmu",
            "typology": "residential",
            "client": "Far East",
            "stage": "construction",
            "location": "Taipei",
        },
        cfg_obj=cfg,
    )
    with patch("core.atlas_wiki._auto_candidates", return_value=[]):
        page = get_wiki_page("261", conn, cfg_obj=cfg)
    conn.close()

    assert page["schema_version"] == SCHEMA_VERSION
    assert "model" in page
    assert page["model"]["source_of_truth"] == "structured_rows"
    assert page["model"]["documents"]["authoritative"]
    assert page["model"]["lifecycle"]["stage"] == "construction"
    assert page["model_readiness"]["what_is_this_project"] is True

    reloaded = load_overlay("261", cfg)
    assert reloaded["schema_version"] == SCHEMA_VERSION
    assert reloaded["documents"]
