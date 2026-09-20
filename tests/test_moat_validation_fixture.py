"""Moat fixture load + LLM-off filter (Hunt H1)."""

from __future__ import annotations

from pathlib import Path

import yaml

from core.eval import (
    _load_fixture,
    _normalize_fixture_entries,
    default_moat_fixture_path,
    filter_llm_off_entries,
)

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "moat_validation.yaml"


def test_moat_validation_fixture_loads():
    data = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    assert data["purpose"] == "moat_validation"
    queries = data["queries"]
    assert len(queries) >= 10

    filled = [q for q in queries if (q.get("query") or "").strip()]
    assert len(filled) >= 8

    llm_off = [q for q in filled if q.get("llm_off_ok")]
    assert len(llm_off) >= 3

    killer = next(q for q in queries if q["id"] == "moat_nv_education_under_24m")
    assert "natural" in killer["query"].lower() or "ventilat" in killer["query"].lower()
    assert killer.get("requires_llm") is False
    assert "Project" in killer.get("table_columns", [])


def test_normalize_dict_fixture_skips_empty_queries():
    entries = _normalize_fixture_entries(
        {
            "purpose": "moat_validation",
            "queries": [
                {"id": "a", "query": "  tender documentation  ", "expected_paths": []},
                {"id": "b", "query": "", "notes": "placeholder"},
                {"id": "c", "query": None},
                {
                    "id": "d",
                    "query": "issued façade",
                    "expected_path_substrings": ["facade", "Façade"],
                    "expected_paths": ["docs/a.pdf"],
                },
            ],
        }
    )
    assert [e["id"] for e in entries] == ["a", "d"]
    assert entries[0]["query"] == "tender documentation"
    assert entries[1]["expected_paths"] == ["docs/a.pdf", "facade", "Façade"]


def test_normalize_list_fixture_still_works():
    entries = _normalize_fixture_entries(
        [{"query": "brief", "expected_paths": ["x.txt"]}]
    )
    assert len(entries) == 1
    assert entries[0]["expected_paths"] == ["x.txt"]


def test_load_moat_fixture_via_path():
    path = default_moat_fixture_path()
    assert path == FIXTURE
    entries = _load_fixture(path)
    assert len(entries) >= 8
    assert all(e["query"].strip() for e in entries)


def test_filter_llm_off_entries():
    entries = _load_fixture(FIXTURE)
    llm_off = filter_llm_off_entries(entries)
    assert len(llm_off) >= 3
    assert all(e.get("llm_off_ok") for e in llm_off)
    assert all("llm_off" in e.get("id", "") or e.get("llm_off_ok") for e in llm_off)
