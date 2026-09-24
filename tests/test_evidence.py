"""tests/test_evidence.py — first-class EvidenceCitation + pack warnings."""

from __future__ import annotations

from core.evidence import (
    EvidenceCitation,
    EvidencePack,
    format_evidence_block,
    version_status_from_flags,
)
from core.compose import ResultView, _SYSTEM_PROMPT, _build_context


def test_version_status_from_flags():
    assert version_status_from_flags(1, 0) == "current"
    assert version_status_from_flags(0, 1) == "superseded"
    assert version_status_from_flags(0, 0) == "unknown"


def test_evidence_from_search_result():
    r = {
        "citation": "brief.pdf#p2",
        "chunk_id": "c1",
        "file_id": "f1",
        "file_path": "/nas/brief.pdf",
        "rel_path": "brief.pdf",
        "file_name": "brief.pdf",
        "ref_value": "p2",
        "project_id": "PP-2025-052",
        "project_code": "PP-2025-052",
        "discipline": "AR",
        "doc_type": "brief",
        "revision": 3,
        "file_date": "2026-07-01",
        "is_latest": 1,
        "is_superseded": 0,
        "snippet": "GFA 45000",
        "chunk_text": "Requested GFA 45000 sqm",
        "final_score": 0.9,
    }
    ev = EvidenceCitation.from_search_result(r)
    assert ev.citation == "brief.pdf#p2"
    assert ev.ref_value == "p2"
    assert ev.revision == 3
    assert ev.version_status == "current"
    assert "GFA" in ev.quote
    assert ev.to_dict()["discipline"] == "AR"


def test_version_warnings_on_mixed_pack():
    current = EvidenceCitation(
        citation="a.pdf#p1",
        project_id="X",
        discipline="AR",
        doc_type="brief",
        file_name="brief.pdf",
        version_status="current",
        quote="A",
    )
    old = EvidenceCitation(
        citation="a.pdf#p1",
        project_id="X",
        discipline="AR",
        doc_type="brief",
        file_name="brief.pdf",
        version_status="superseded",
        quote="B",
    )
    pack = EvidencePack(items=[current, old])
    pack.version_warnings = EvidencePack.from_search_results([
        {
            "citation": "a.pdf#p1",
            "project_id": "X",
            "discipline": "AR",
            "doc_type": "brief",
            "file_name": "brief.pdf",
            "is_latest": 1,
            "is_superseded": 0,
            "snippet": "A",
        },
        {
            "citation": "b.pdf#p1",
            "project_id": "X",
            "discipline": "AR",
            "doc_type": "brief",
            "file_name": "brief.pdf",
            "is_latest": 0,
            "is_superseded": 1,
            "snippet": "B",
        },
    ]).version_warnings
    assert pack.version_warnings
    assert "superseded" in pack.version_warnings[0].lower()


def test_format_evidence_block_includes_metadata():
    ev = EvidenceCitation(
        citation="deck.pptx#s3",
        project_id="261",
        discipline="ST",
        revision=2,
        version_status="current",
        quote="Concept frame",
    )
    block = format_evidence_block(ev, 1)
    assert "[1] deck.pptx#s3" in block
    assert "rev=2" in block
    assert "version=current" in block


def test_compose_prompt_requires_epistemic_labels():
    assert "DOCUMENTED" in _SYSTEM_PROMPT
    assert "INFERRED" in _SYSTEM_PROMPT
    assert "CONFLICTING" in _SYSTEM_PROMPT
    assert "MISSING" in _SYSTEM_PROMPT


def test_build_context_uses_evidence_metadata():
    view = ResultView.from_search_result({
        "rel_path": "brief.pdf",
        "file_name": "brief.pdf",
        "file_path": "/x/brief.pdf",
        "citation": "brief.pdf#p1",
        "snippet": "GFA",
        "chunk_text": "GFA 12000",
        "project_id": "261",
        "typology": "office",
        "final_score": 0.8,
        "ref_value": "p1",
        "chunk_id": "c1",
        "discipline": "AR",
        "is_latest": 1,
        "revision": 4,
    })
    ctx = _build_context([view])
    assert "brief.pdf#p1" in ctx
    assert "discipline=AR" in ctx
    assert "rev=4" in ctx
