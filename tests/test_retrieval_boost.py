"""
tests/test_retrieval_boost.py — Google-classic Hunt ranking signals (no API).
"""

from __future__ import annotations

from core.retrieval_boost import (
    apply_archive_boosts,
    apply_critical_token_honesty,
    assess_evidence_support,
    build_fts_query,
    detect_project_codes,
    domain_expand_terms,
    extract_critical_tokens,
    extract_filename_hints,
    extract_phrases,
    name_family_key,
    path_filename_boost,
    project_code_boost,
    suppress_near_duplicates,
)
from core.router import QueryRouter


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


def test_suppress_presentation_name_family_variants() -> None:
    """Tender presentation Rev/Final/dated copies must not flood the pack."""
    cands = [
        {
            "file_id": "1",
            "file_name": "Tender_Presentation_RevA.pptx",
            "file_path": "/NUS BIZ3/Tender_Presentation_RevA.pptx",
            "final_score": 1.0,
        },
        {
            "file_id": "2",
            "file_name": "Tender_Presentation_Final.pptx",
            "file_path": "/NUS BIZ3/Tender_Presentation_Final.pptx",
            "final_score": 0.95,
        },
        {
            "file_id": "3",
            "file_name": "Tender_Presentation_20240301.pdf",
            "file_path": "/NUS BIZ3/Tender_Presentation_20240301.pdf",
            "final_score": 0.9,
        },
        {
            "file_id": "4",
            "file_name": "Design_Brief.pdf",
            "file_path": "/NUS BIZ3/Design_Brief.pdf",
            "final_score": 0.8,
        },
    ]
    assert name_family_key("Tender_Presentation_RevA.pptx") == name_family_key(
        "Tender_Presentation_Final.pptx"
    )
    out = suppress_near_duplicates(cands, max_per_file=2, max_per_name_family=1)
    names = [c["file_name"] for c in out]
    assert names[0] == "Tender_Presentation_RevA.pptx"
    assert "Tender_Presentation_Final.pptx" not in names
    assert "Tender_Presentation_20240301.pdf" not in names
    assert "Design_Brief.pdf" in names


def test_exact_filename_hints_and_router_path_terms() -> None:
    hints = extract_filename_hints("NUS BIZ3 xyzzy_nonexistent_987654.dwg")
    assert any("xyzzy_nonexistent_987654" in h for h in hints)

    route = QueryRouter().classify("NUS BIZ3 xyzzy_nonexistent_987654.dwg")
    assert route.mode == "file_locator"
    assert route.filters.get("exact_filename") is True
    assert any("xyzzy_nonexistent_987654" in t for t in route.filters.get("path_terms", []))


def test_good_project_queries_have_no_exact_filename_trap() -> None:
    for q in (
        "NUS BIZ3 presentation ppt",
        "NUS BIZ3 CAD",
        "NUS BIZ3 design brief",
    ):
        assert extract_filename_hints(q) == []
        critical = extract_critical_tokens(q)
        # File-type / brief vocabulary must not become claim-critical.
        assert "presentation" not in critical
        assert "ppt" not in critical
        assert "cad" not in critical
        assert "design" not in critical
        assert "brief" not in critical


def test_critical_tokens_for_impossible_claims() -> None:
    cost = extract_critical_tokens("NUS BIZ3 approved final cost 999 trillion")
    assert "999" in cost
    assert "trillion" in cost

    moon = extract_critical_tokens("NUS BIZ3 moon base approval")
    assert "moon" in moon
    assert "approval" not in moon  # common archive vocab


def test_unsupported_claim_empties_evidence_pack() -> None:
    cands = [
        {
            "file_id": "1",
            "file_name": "QS_Agreement.pdf",
            "file_path": "/NUS BIZ3/QS_Agreement.pdf",
            "chunk_text": "Final approved cost agreement between client and QS.",
            "final_score": 0.9,
        },
        {
            "file_id": "2",
            "file_name": "Approval_Letter.pdf",
            "file_path": "/NUS BIZ3/Approval_Letter.pdf",
            "chunk_text": "Authority approval for the building submission.",
            "final_score": 0.85,
        },
    ]
    out, assessment = apply_critical_token_honesty(
        cands, "NUS BIZ3 approved final cost 999 trillion"
    )
    assert assessment["support_status"] == "unsupported"
    assert "999" in assessment["unsupported_tokens"]
    assert out == []

    out2, a2 = apply_critical_token_honesty(
        list(cands), "NUS BIZ3 moon base approval"
    )
    assert a2["support_status"] == "unsupported"
    assert "moon" in a2["unsupported_tokens"]
    assert out2 == []


def test_supported_claim_keeps_matching_evidence() -> None:
    cands = [
        {
            "file_id": "1",
            "file_name": "cost_sheet.pdf",
            "file_path": "/NUS BIZ3/cost_sheet.pdf",
            "chunk_text": "Approved final cost is 999 trillion imaginary dollars.",
            "final_score": 0.5,
        },
        {
            "file_id": "2",
            "file_name": "other.pdf",
            "file_path": "/NUS BIZ3/other.pdf",
            "chunk_text": "Generic approval letter with no figures.",
            "final_score": 0.9,
        },
    ]
    out, assessment = apply_critical_token_honesty(
        cands, "NUS BIZ3 approved final cost 999 trillion"
    )
    assert assessment["support_status"] == "supported"
    assert out
    assert out[0]["file_id"] == "1"


def test_assess_evidence_support_na_without_critical_tokens() -> None:
    assessment = assess_evidence_support(
        "NUS BIZ3 presentation ppt",
        [{"snippet": "deck", "file_name": "deck.pptx", "final_score": 1.0}],
    )
    assert assessment["support_status"] == "n/a"


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
