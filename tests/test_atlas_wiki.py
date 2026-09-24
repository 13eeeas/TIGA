"""tests/test_atlas_wiki.py — Published gate + overlay persist for Atlas #4."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import yaml

from config import load_config
from core.atlas_wiki import (
    compose_blurb,
    compute_health,
    evidence_pack,
    get_wiki_page,
    list_wiki_projects,
    load_overlay,
    wiki_fact,
    wiki_hide,
    project_codes_for_scope,
    wiki_overview,
    wiki_pin,
    wiki_unhide,
)
from core.db import get_connection, upsert_file


def _cfg(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.dump({"index_roots": [str(tmp_path / "archive")]}),
        encoding="utf-8",
    )
    return load_config(config_file=cfg_file, work_dir=tmp_path)


def _empty_page(**overrides) -> dict:
    page = {
        "project": {"code": "NUS", "name": "NUS"},
        "summary": (
            "Auto-draft Grokopedia page for NUS. "
            "Hunt proposed candidates — confirm pins and cite facts like a wiki."
        ),
        "pins": [],
        "facts": [],
    }
    page.update(overrides)
    return page


def _curated_project() -> dict:
    return {
        "code": "261",
        "name": "Tianmu",
        "typology": "residential",
        "client": "Far East",
        "stage": "construction",
        "location": "Taipei",
    }


# ---------------------------------------------------------------------------
# Blurb
# ---------------------------------------------------------------------------

def test_compose_blurb_needs_curation_when_empty() -> None:
    text = compose_blurb({"code": "NUS", "name": "NUS BIZ3"})
    assert "Needs curation" in text
    assert "NUS BIZ3" in text


def test_compose_blurb_human_summary_wins() -> None:
    text = compose_blurb(
        _curated_project(),
        "Staff note: hotel tower on Orchard.",
    )
    assert text == "Staff note: hotel tower on Orchard."


def test_compose_blurb_from_identity_fields() -> None:
    text = compose_blurb(_curated_project())
    assert "Tianmu" in text
    assert "residential" in text
    assert "Far East" in text
    assert "construction" in text
    assert "Taipei" in text
    assert "Needs curation" not in text


def test_placeholder_fields_do_not_count_as_filled() -> None:
    text = compose_blurb(
        {
            "code": "NUS",
            "name": "NUS",
            "typology": "Needs curation",
            "client": "Unknown",
            "stage": "n/a",
            "location": "",
        }
    )
    assert "Needs curation" in text


# ---------------------------------------------------------------------------
# Published gate
# ---------------------------------------------------------------------------

def test_empty_card_is_needs_curation_not_published_100() -> None:
    health = compute_health(_empty_page())
    assert health["published"] is False
    assert health["state"] == "Needs curation"
    assert health["show_health_score"] is False
    assert health["score"] < 100
    assert "typology" in health["missing_fields"]
    assert "client" in health["missing_fields"]
    assert "stage" in health["missing_fields"]
    assert "location" in health["missing_fields"]


def test_pin_only_does_not_publish_or_score_100() -> None:
    health = compute_health(
        _empty_page(pins=[{"role": "other", "path": "/a.pdf"}])
    )
    assert health["published"] is False
    assert health["score"] < 100
    assert health["show_health_score"] is False
    assert health["state"] == "Needs curation"


def test_needs_curation_fields_block_published_even_with_pin_and_cites() -> None:
    health = compute_health(
        {
            "project": {
                "code": "NUS",
                "name": "NUS",
                "typology": "Needs curation",
                "client": "Needs curation",
                "stage": "Needs curation",
                "location": "Needs curation",
            },
            "summary": "",
            "pins": [{"role": "other", "path": "/a.pdf"}],
            "facts": [
                {
                    "key": "client",
                    "value": "NUS",
                    "cite_paths": ["/brief.pdf"],
                }
            ],
        }
    )
    assert health["published"] is False
    assert health["score"] < 100
    assert health["show_health_score"] is False


def test_curated_fields_without_cited_facts_are_not_published() -> None:
    health = compute_health(
        {
            "project": _curated_project(),
            "summary": "Tianmu is a residential project.",
            "pins": [{"role": "other", "path": "/a.pdf"}],
            "facts": [{"key": "client", "value": "Far East", "cite_paths": []}],
        }
    )
    assert health["published"] is False
    assert health["cited_facts"] == 0


def test_published_when_curated_pin_and_cited_facts() -> None:
    health = compute_health(
        {
            "project": _curated_project(),
            "summary": "Tianmu is a residential project for Far East in Taipei.",
            "pins": [{"role": "overview_deck", "path": "/deck.pdf"}],
            "facts": [
                {
                    "key": "client",
                    "label": "Client",
                    "value": "Far East",
                    "cite_paths": ["/brief.pdf"],
                }
            ],
        }
    )
    assert health["published"] is True
    assert health["state"] == "Published"
    assert health["show_health_score"] is True
    assert health["score"] == 100
    assert health["missing_fields"] == []


# ---------------------------------------------------------------------------
# Overlay persist
# ---------------------------------------------------------------------------

def test_overlay_pin_hide_fact_overview_persist(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    code = "261_tianmu"

    wiki_pin(
        code,
        role="overview_deck",
        path="/archive/261/deck.pdf",
        title="Deck",
        cfg_obj=cfg,
    )
    wiki_hide(code, "/archive/261/Copy of junk.pdf", cfg_obj=cfg)
    wiki_fact(
        code,
        key="client",
        label="Client",
        value="Far East",
        cite_paths=["/archive/261/brief.pdf"],
        status="verified",
        cfg_obj=cfg,
    )
    wiki_overview(
        code,
        summary="Tianmu is a residential project for Far East in Taipei.",
        project_fields={
            "name": "Tianmu",
            "typology": "residential",
            "client": "Far East",
            "stage": "construction",
            "location": "Taipei",
        },
        cfg_obj=cfg,
    )

    overlay_path = cfg.work_dir / "atlas" / "261-tianmu.overlay.json"
    assert overlay_path.exists()

    reloaded = load_overlay(code, cfg)
    assert reloaded["pins"][0]["path"] == "/archive/261/deck.pdf"
    assert "/archive/261/Copy of junk.pdf" in reloaded["hidden_paths"]
    assert reloaded["facts"][0]["cite_paths"] == ["/archive/261/brief.pdf"]
    assert reloaded["summary"].startswith("Tianmu is a residential")
    assert reloaded["project"]["typology"] == "residential"
    assert reloaded["project"]["client"] == "Far East"
    assert reloaded["project"]["stage"] == "construction"
    assert reloaded["project"]["location"] == "Taipei"

    wiki_unhide(code, "/archive/261/Copy of junk.pdf", cfg_obj=cfg)
    after_unhide = load_overlay(code, cfg)
    assert after_unhide["hidden_paths"] == []
    assert after_unhide["pins"][0]["path"] == "/archive/261/deck.pdf"


# ---------------------------------------------------------------------------
# Page + list merge
# ---------------------------------------------------------------------------

def test_get_wiki_page_empty_card_is_not_published(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    conn = get_connection(cfg.get_db_path())
    upsert_file(
        conn,
        {
            "file_id": "f1",
            "file_path": "/archive/NUS/a.txt",
            "file_name": "a.txt",
            "project_id": "NUS",
        },
    )
    conn.commit()
    with patch("core.atlas_wiki._auto_candidates", return_value=[]):
        page = get_wiki_page("NUS", conn, cfg_obj=cfg)
    conn.close()

    assert page["health"]["published"] is False
    assert page["health"]["score"] < 100
    assert page["project"]["status"] == "needs-curation"
    assert "Needs curation" in page["blurb"]
    assert page["index"]["file_count"] == 1
    assert "Grokopedia" not in (page.get("summary") or "")


def test_list_wiki_projects_leads_with_blurb_not_published(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    conn = get_connection(cfg.get_db_path())
    upsert_file(
        conn,
        {
            "file_id": "f1",
            "file_path": "/archive/NUS/a.txt",
            "file_name": "a.txt",
            "project_id": "NUS",
        },
    )
    conn.commit()
    rows = list_wiki_projects(conn, cfg_obj=cfg)
    conn.close()
    assert len(rows) == 1
    item = rows[0]
    assert item["status"] == "needs-curation"
    assert item["published"] is False
    assert "Needs curation" in item["blurb"]
    assert item["file_count"] == 1
    assert item["files_indexed"] == 0
    assert item["index_state"] == "in_progress"


def test_projects_html_matches_hunt_chrome_not_atlas_dialect() -> None:
    html = (Path(__file__).resolve().parents[1] / "static" / "projects.html").read_text(
        encoding="utf-8"
    )
    assert "logo-name" in html
    assert "blob-layer" in html
    assert "the job in a glance" in html
    assert "Needs curation" in html
    assert 'href="/#settings"' in html
    assert "Project wiki" not in html
    assert "letter-spacing:.04em;text-transform:uppercase" not in html


def test_get_wiki_page_published_after_overview_pin_and_cite(tmp_path: Path) -> None:
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
    conn.commit()
    wiki_pin(code="261", role="overview_deck", path="/archive/261/brief.pdf", cfg_obj=cfg)
    wiki_fact(
        "261",
        key="client",
        label="Client",
        value="Far East",
        cite_paths=["/archive/261/brief.pdf"],
        status="verified",
        cfg_obj=cfg,
    )
    wiki_overview(
        "261",
        summary="Tianmu is a residential project for Far East in Taipei, currently at construction.",
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
    assert page["health"]["published"] is True
    assert page["health"]["score"] == 100
    assert page["project"]["status"] == "published"
    assert "Far East" in page["blurb"]


def test_typology_scope_reads_project_overlay(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    conn = get_connection(tmp_path / "db" / "tiga.db")
    wiki_overview(
        "NUS",
        summary="A business school.",
        project_fields={
            "typology": "education",
            "client": "NUS",
            "stage": "tender",
            "location": "Singapore",
        },
        cfg_obj=cfg,
    )
    wiki_overview(
        "HICA",
        summary="A hotel.",
        project_fields={"typology": "hospitality"},
        cfg_obj=cfg,
    )
    conn.close()
    assert project_codes_for_scope(typology="education", cfg_obj=cfg) == ["NUS"]
    assert project_codes_for_scope(typology="Education", cfg_obj=cfg) == ["NUS"]
    assert project_codes_for_scope(typology="hospitality", cfg_obj=cfg) == ["HICA"]


def test_auto_facts_do_not_block_published() -> None:
    health = compute_health(
        {
            "project": _curated_project(),
            "summary": "Tianmu is a residential project for Far East in Taipei.",
            "pins": [{"role": "overview_deck", "path": "/deck.pdf"}],
            "facts": [
                {"key": "typology", "value": "residential", "cite_paths": [], "source": "auto"},
                {"key": "client", "value": "Far East", "cite_paths": ["/brief.pdf"], "source": "wiki"},
            ],
        }
    )
    assert health["published"] is True
    assert health["grade"] == "A"
    assert health["uncited_facts"] == 0


def test_published_without_staff_blurb_is_grade_b() -> None:
    health = compute_health(
        {
            "project": _curated_project(),
            "summary": "",
            "pins": [{"role": "overview_deck", "path": "/deck.pdf"}],
            "facts": [
                {"key": "client", "value": "Far East", "cite_paths": ["/brief.pdf"], "source": "wiki"},
            ],
        }
    )
    assert health["published"] is True
    assert health["grade"] == "B"
    assert health["score"] == 100


def test_second_pin_same_role_is_kept(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    wiki_pin("261", role="other", path="/a.pdf", title="A", cfg_obj=cfg)
    wiki_pin("261", role="other", path="/b.pdf", title="B", cfg_obj=cfg)
    pins = load_overlay("261", cfg)["pins"]
    assert [p["path"] for p in pins] == ["/a.pdf", "/b.pdf"]


def test_ask_pack_omits_unpinned_candidates() -> None:
    pack = evidence_pack(
        {
            "pins": [{"role": "overview_deck", "path": "/deck.pdf", "title": "Deck", "note": ""}],
            "facts": [
                {"key": "client", "label": "Client", "value": "Far East", "cite_paths": ["/brief.pdf"]},
                {"key": "stage", "label": "Stage", "value": "tender", "cite_paths": [], "source": "auto"},
            ],
            "sections": {
                "key_documents": {"items": [{"title": "loose.pdf", "path": "/loose.pdf", "pinned": False}]},
            },
        }
    )
    kinds = [item["kind"] for item in pack]
    assert kinds == ["pin", "fact"]
    assert pack[1]["key"] == "client"


def test_corrupt_overlay_does_not_raise(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    path = cfg.work_dir / "atlas"
    path.mkdir(parents=True)
    (path / "nus.overlay.json").write_text("{not json", encoding="utf-8")
    overlay = load_overlay("NUS", cfg)
    assert overlay["pins"] == []
    assert overlay["project_code"] == "NUS"


def test_slug_collision_uses_a_second_file(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    wiki_pin("ABC", role="other", path="/a.pdf", cfg_obj=cfg)
    wiki_pin("abc", role="other", path="/b.pdf", cfg_obj=cfg)
    assert load_overlay("ABC", cfg)["pins"][0]["path"] == "/a.pdf"
    assert load_overlay("abc", cfg)["pins"][0]["path"] == "/b.pdf"
    files = list((cfg.work_dir / "atlas").glob("*.overlay.json"))
    assert len(files) == 2


def test_auto_draft_uses_index_classification_not_filenames(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    conn = get_connection(cfg.get_db_path())
    samples = [
        ("concept presentation.pptx", ".pptx", "unknown", "Presentation", "", "/archive/261/13 PPT/concept presentation.pptx"),
        ("GA.dwg", ".dwg", "General Arrangement", "CAD", "", "/archive/261/09 CAD/GA.dwg"),
        ("design brief.pdf", ".pdf", "unknown", "PDF", "", "/archive/261/01 Brief/design brief.pdf"),
        ("kickoff minutes.pdf", ".pdf", "Minutes", "PDF", "", "/archive/261/06 Meeting/kickoff minutes.pdf"),
        ("site overview photo.jpg", ".jpg", "unknown", "Image", "photo", "/archive/261/06 Meeting/site overview photo.jpg"),
        ("planting plan note.txt", ".txt", "unknown", "unknown", "", "/archive/261/08 Working/planting plan note.txt"),
    ]
    for i, (name, ext, doc, content, image, path) in enumerate(samples):
        upsert_file(
            conn,
            {
                "file_id": f"f{i}",
                "file_path": path,
                "file_name": name,
                "extension": ext,
                "project_id": "261",
            },
        )
        conn.execute(
            "UPDATE files SET doc_type=?, content_type=?, image_type=? WHERE file_id=?",
            (doc, content, image, f"f{i}"),
        )
    conn.commit()
    page = get_wiki_page("261", conn, cfg_obj=cfg)
    built = page["index"]["auto_built_at"]
    titles = {key: [item["title"] for item in section["items"]] for key, section in page["sections"].items()}
    assert titles["brief"] == ["design brief.pdf"]
    assert titles["drawings"] == ["GA.dwg"]
    assert titles["meetings"] == ["kickoff minutes.pdf"]
    assert titles["presentations"] == ["concept presentation.pptx"]
    assert page["sections"]["drawings"]["summary"] == ""
    assert "site overview photo.jpg" not in str(titles)
    assert "planting plan note.txt" not in str(titles)
    assert list(page["sections"]) == ["brief", "drawings", "meetings", "presentations"]
    again = get_wiki_page("261", conn, cfg_obj=cfg)
    conn.close()
    assert again["index"]["auto_built_at"] == built
    assert load_overlay("261", cfg).get("auto_built_at")
