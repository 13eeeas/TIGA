"""
tests/test_retrieval_boost.py — Google-classic Hunt ranking signals (no API).
"""

from __future__ import annotations

from core.retrieval_boost import (
    apply_archive_boosts,
    build_fts_query,
    detect_project_codes,
    domain_expand_terms,
    extract_phrases,
    path_filename_boost,
    project_code_boost,
    suppress_near_duplicates,
)


def test_domain_expand_dwellings() -> None:
    terms = domain_expand_terms("how many dwellings did the owner want")
    assert "units" in terms
    assert "residential" in terms or "client" in terms


def test_extract_green_mark_phrase() -> None:
    phrases = extract_phrases("green mark gold residential tianmu")
    assert any("green mark" in p for p in phrases)


def test_build_fts_includes_phrase_and_expand() -> None:
    q = build_fts_query("dwellings for Tianmu", use_phrases=True, use_domain_expand=True)
    assert "units" in q.lower() or "residential" in q.lower()
    assert "Tianmu" in q or "tianmu" in q.lower()


def test_path_boost_higher_when_name_matches() -> None:
    plain = path_filename_boost("tender Rev F", "/a/other/notes.txt", "notes.txt")
    hit = path_filename_boost(
        "tender Rev F",
        "/a/261_tianmu/02_Tender/tender_documentation.txt",
        "tender_documentation.txt",
    )
    assert hit > plain


def test_project_code_boost() -> None:
    assert project_code_boost("Project 261 brief", "261_tianmu") > 1.0
    assert project_code_boost("hotel spa", "261_tianmu") == 1.0
    assert detect_project_codes("look at 186 and 232") == ["186", "232"]


def test_suppress_near_duplicates() -> None:
    cands = [
        {"file_id": "a", "final_score": 1.0, "file_path": "/a"},
        {"file_id": "a", "final_score": 0.9, "file_path": "/a"},
        {"file_id": "a", "final_score": 0.8, "file_path": "/a"},
        {"file_id": "b", "final_score": 0.7, "file_path": "/b"},
    ]
    out = suppress_near_duplicates(cands, max_per_file=2)
    assert len(out) == 3
    assert sum(1 for c in out if c["file_id"] == "a") == 2


def test_apply_archive_boosts_reorders() -> None:
    cands = [
        {
            "file_id": "1",
            "file_path": "/archive/other/misc.txt",
            "file_name": "misc.txt",
            "project_id": "999_x",
            "final_score": 0.9,
        },
        {
            "file_id": "2",
            "file_path": "/archive/261_tianmu/01_Brief/project_brief.txt",
            "file_name": "project_brief.txt",
            "project_id": "261_tianmu",
            "final_score": 0.85,
        },
    ]
    apply_archive_boosts(cands, "261 Tianmu project brief")
    assert cands[0]["file_id"] == "2"
