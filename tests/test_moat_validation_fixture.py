"""Smoke-load moat validation fixture (architecture-intelligence gate)."""

from __future__ import annotations

from pathlib import Path

import yaml

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
