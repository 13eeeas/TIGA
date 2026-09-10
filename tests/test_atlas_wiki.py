"""Thesis-level tests for Atlas's human-curated trust boundary."""

from pathlib import Path
from unittest.mock import patch

import yaml

from config import load_config
from core.atlas_wiki import (
    compute_health,
    evidence_pack,
    get_wiki_page,
    list_wiki_projects,
    load_overlay,
    wiki_compose,
    wiki_fact,
    wiki_hide,
    wiki_pin,
)
from core.db import get_connection


def _config(tmp_path: Path):
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        yaml.safe_dump({"index_roots": [str(tmp_path / "archive")]}),
        encoding="utf-8",
    )
    return load_config(config_file=config_file, work_dir=tmp_path)


def test_candidates_are_not_truth_and_overlays_survive_rebuild(tmp_path: Path) -> None:
    cfg = _config(tmp_path)
    conn = get_connection(tmp_path / "db" / "tiga.db")
    candidates = [
        {
            "title": "Overview.pdf",
            "path": "/NAS/NUS/Overview.pdf",
            "snippet": "Unverified search result",
            "score": 0.91,
            "via": "auto:overview",
            "section": "key_documents",
            "suggested_role": "overview_deck",
            "kind": "candidate",
            "pinned": False,
        },
        {
            "title": "Copy of old.pdf",
            "path": "/NAS/NUS/Copy of old.pdf",
            "snippet": "Junk candidate",
            "score": 0.5,
            "via": "auto:overview",
            "section": "key_documents",
            "suggested_role": "other",
            "kind": "candidate",
            "pinned": False,
        },
    ]
    card = {"name": "NUS BIZ3", "typology_primary": "education"}

    try:
        with (
            patch("core.atlas_wiki._auto_candidates", return_value=candidates),
            patch("core.project_card.get_project_card", return_value=card),
        ):
            draft = get_wiki_page("NUS BIZ3", conn=conn, cfg_obj=cfg)
            assert draft["health"]["published"] is False
            assert draft["sections"]["key_documents"]["items"][0]["kind"] == "candidate"

            wiki_pin(
                "NUS BIZ3",
                role="overview_deck",
                path="/NAS/NUS/Overview.pdf",
                cfg_obj=cfg,
            )
            wiki_hide("NUS BIZ3", "/NAS/NUS/Copy of old.pdf", cfg_obj=cfg)
            wiki_fact(
                "NUS BIZ3",
                key="typology",
                label="Typology",
                value="education",
                cite_paths=["/NAS/NUS/Overview.pdf#p3"],
                status="verified",
                cfg_obj=cfg,
            )

            rebuilt = get_wiki_page("NUS BIZ3", conn=conn, cfg_obj=cfg)

        overlay = load_overlay("NUS BIZ3", cfg)
        assert overlay["pins"][0]["source"] == "wiki"
        assert overlay["facts"][0]["cite_paths"] == ["/NAS/NUS/Overview.pdf#p3"]
        items = rebuilt["sections"]["key_documents"]["items"]
        assert [item["path"] for item in items] == ["/NAS/NUS/Overview.pdf"]
        assert items[0]["kind"] == "pinned"
        assert rebuilt["health"]["published"] is True
    finally:
        conn.close()


def test_ask_uses_pins_and_cited_facts_but_never_candidates() -> None:
    page = {
        "pins": [{"role": "overview_deck", "title": "Overview", "path": "/NAS/Overview.pdf"}],
        "facts": [
            {"key": "gfa", "label": "GFA", "value": "10,000 sqm", "cite_paths": ["/NAS/GFA.pdf#p2"], "status": "verified"},
            {"key": "client", "label": "Client", "value": "Unverified", "cite_paths": [], "status": "draft"},
        ],
        "sections": {
            "key_documents": {"items": [{"kind": "candidate", "title": "Rumour", "path": "/NAS/Rumour.txt", "snippet": "Do not trust"}]},
        },
    }

    pack = evidence_pack(page)
    assert {item["kind"] for item in pack} == {"pin", "fact"}
    assert all("Rumour" not in str(item) for item in pack)
    result = wiki_compose(page, "What is the GFA?")
    assert result["ok"] is True
    assert "/NAS/GFA.pdf#p2" in result["answer"]
    assert "Rumour" not in result["answer"]
    assert "Unverified" not in result["answer"]


def test_ask_locks_without_authoritative_pin() -> None:
    result = wiki_compose({"pins": [], "facts": [], "sections": {}}, "What is true?")
    assert result["ok"] is False
    assert "locked" in result["answer"].lower()


def test_health_rejects_uncited_durable_facts() -> None:
    page = {
        "project": {"code": "NUS BIZ3"},
        "summary": "Project summary",
        "pins": [{"path": "/NAS/Overview.pdf"}],
        "facts": [{"key": "gfa", "cite_paths": []}],
    }
    health = compute_health(page)
    assert health["published"] is False
    assert health["uncited_facts"] == 1


def test_project_list_only_surfaces_named_indexed_corpora(tmp_path: Path) -> None:
    conn = get_connection(tmp_path / "db" / "tiga.db")
    try:
        conn.executemany(
            "INSERT INTO files (file_id, file_path, file_name, extension, status, project_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                ("ready", "/archive/brief.pdf", "brief.pdf", ".pdf", "INDEXED", "NParks"),
                ("pending", "/archive/model.rvt", "model.rvt", ".rvt", "DISCOVERED", "NParks"),
                ("unknown", "/archive/misc.pdf", "misc.pdf", ".pdf", "INDEXED", "Unknown"),
                ("blank", "/archive/loose.pdf", "loose.pdf", ".pdf", "INDEXED", None),
            ],
        )
        projects = list_wiki_projects(conn)
        assert [project["project_id"] for project in projects] == ["NParks"]
        assert projects[0]["file_count"] == 2
        assert projects[0]["indexed_count"] == 1
    finally:
        conn.close()
