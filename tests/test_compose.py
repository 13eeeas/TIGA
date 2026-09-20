"""
tests/test_compose.py — compose_answer() unit tests.

LLM calls mocked via core.compose.chat_completion.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from config import load_config
from core.db import get_connection, upsert_chunk, upsert_file
from core.compose import (
    ComposeResult,
    ResultView,
    _confidence,
    compose_answer,
)
from core.llm_providers import LLMError


def _cfg(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.dump({
            "index_roots": [str(tmp_path)],
            "compose": {"api_enabled": False, "provider": "ollama"},
        }),
        encoding="utf-8",
    )
    return load_config(config_file=cfg_file, work_dir=tmp_path)


@pytest.fixture
def conn(tmp_path: Path):
    db = get_connection(tmp_path / "db" / "tiga.db")
    yield db
    db.close()


def _seed_chunk(conn, tmp_path: Path, chunk_id: str = "cid-001") -> None:
    fid = "fid-001"
    upsert_file(conn, {
        "file_id": fid,
        "file_path": str(tmp_path / "proj" / "brief.pdf"),
        "file_name": "brief.pdf",
        "project_id": "261",
        "typology": "healthcare",
        "extension": ".pdf",
        "status": "INDEXED",
    })
    upsert_chunk(conn, {
        "chunk_id": chunk_id,
        "file_id": fid,
        "ref_value": "p1",
        "text": "hospital architecture brief design requirements for project 261",
    })


def _fake_result(score: float = 0.8, chunk_id: str = "cid-001") -> dict:
    return {
        "chunk_id": chunk_id,
        "file_id": "fid-001",
        "ref_value": "p1",
        "rel_path": "proj/brief.pdf",
        "file_path": "/tmp/proj/brief.pdf",
        "file_name": "brief.pdf",
        "project_id": "261",
        "typology": "healthcare",
        "snippet": "hospital architecture brief",
        "chunk_text": "hospital architecture brief design requirements",
        "bm25_score": score,
        "vector_score": score,
        "final_score": score,
        "citation": "proj/brief.pdf#p1",
    }


def test_result_view_from_search_result() -> None:
    v = ResultView.from_search_result(_fake_result(0.75))
    assert v.title == "brief"
    assert v.citation == "proj/brief.pdf#p1"


def test_confidence_zero_when_no_results() -> None:
    assert _confidence([]) == 0.0


def test_fallback_used_when_llm_unavailable(tmp_path: Path, conn) -> None:
    _seed_chunk(conn, tmp_path)
    cfg_obj = _cfg(tmp_path)
    with patch("core.compose.chat_completion", side_effect=LLMError("down")):
        result = compose_answer("hospital brief", [_fake_result()],
                                conn=conn, cfg_obj=cfg_obj)
    assert isinstance(result, ComposeResult)
    assert result.answer_summary.startswith("[Synthesis unavailable")
    assert result.compose_provider == "fallback"


def test_llm_answer_returned_in_summary(tmp_path: Path, conn) -> None:
    _seed_chunk(conn, tmp_path)
    cfg_obj = _cfg(tmp_path)
    expected = "The hospital brief covers structural requirements."
    with patch("core.compose.chat_completion", return_value=(expected, "openai:gpt-4o-mini")):
        result = compose_answer("hospital brief", [_fake_result()],
                                conn=conn, cfg_obj=cfg_obj)
    assert result.answer_summary == expected
    assert result.compose_provider == "openai:gpt-4o-mini"


def test_evidence_pack_uses_db_chunk_text(tmp_path: Path, conn) -> None:
    _seed_chunk(conn, tmp_path)
    cfg_obj = _cfg(tmp_path)
    cfg_obj.compose_evidence_pack_size = 1

    def capture(messages, cfg, purpose="compose"):
        user = messages[-1]["content"]
        assert "requirements for project 261" in user
        return ("ok", "openai:test")

    with patch("core.compose.chat_completion", side_effect=capture):
        compose_answer("hospital", [_fake_result()], conn=conn, cfg_obj=cfg_obj)


def test_session_persistence_saves_messages(tmp_path: Path, conn) -> None:
    _seed_chunk(conn, tmp_path)
    cfg_obj = _cfg(tmp_path)
    sid = "sess-001"
    with patch("core.compose.chat_completion", return_value=("Answer.", "openai:test")):
        compose_answer("hospital brief", [_fake_result()],
                       session_id=sid, conn=conn, cfg_obj=cfg_obj)
    rows = conn.execute(
        "SELECT role FROM messages WHERE session_id=? ORDER BY message_id",
        (sid,),
    ).fetchall()
    roles = [r["role"] for r in rows]
    assert roles == ["user", "assistant"]


def test_verify_claims_marks_supported_and_unsupported() -> None:
    from core.compose import ResultView, verify_claims

    views = [
        ResultView(
            title="brief",
            rel_path="proj/brief.pdf",
            file_path="/tmp/proj/brief.pdf",
            citation="proj/brief.pdf#p1",
            snippet="hospital architecture brief",
            project_id="261",
            typology="healthcare",
            ext=".pdf",
            final_score=0.8,
            evidence_text="hospital architecture brief design requirements for project 261",
        )
    ]
    answer = (
        "The hospital brief covers design requirements for project 261. "
        "The project budget is exactly 48 million dollars."
    )
    rewritten, verdicts, summary = verify_claims(answer, views)
    assert len(verdicts) == 2
    assert verdicts[0].status == "supported"
    assert verdicts[1].status == "unsupported"
    assert "Not enough evidence in the archive for:" in rewritten
    assert "48 million" in rewritten
    assert "1 supported" in summary
    assert "1 unsupported" in summary


def test_compose_strips_unsupported_claims(tmp_path: Path, conn) -> None:
    _seed_chunk(conn, tmp_path)
    cfg_obj = _cfg(tmp_path)
    hallucinated = (
        "The hospital brief covers design requirements for project 261. "
        "The facade is clad in titanium panels from Norway."
    )
    with patch("core.compose.chat_completion", return_value=(hallucinated, "openai:test")):
        result = compose_answer(
            "hospital brief",
            [_fake_result()],
            conn=conn,
            cfg_obj=cfg_obj,
        )
    assert "Not enough evidence in the archive for:" in result.answer_summary
    assert any(v.status == "unsupported" for v in result.claim_verdicts)
    assert result.verification_summary
    assert result.confidence <= 0.45
    assert any(v.support_status for v in result.results)


def test_fallback_skips_claim_verification(tmp_path: Path, conn) -> None:
    _seed_chunk(conn, tmp_path)
    cfg_obj = _cfg(tmp_path)
    with patch("core.compose.chat_completion", side_effect=LLMError("down")):
        result = compose_answer(
            "hospital brief",
            [_fake_result()],
            conn=conn,
            cfg_obj=cfg_obj,
        )
    assert result.claim_verdicts == []
    assert result.answer_summary.startswith("[Synthesis unavailable")
