"""
tests/test_retrieval_retry.py — thin-pack detection and retry loop.
"""

from __future__ import annotations

from core.retrieval_retry import (
    is_thin_pack,
    reformulate_query,
    run_search_with_retry,
    thin_pack_answer,
)
from core.router import ExpandedQuery, RouteResult


def test_is_thin_pack_empty() -> None:
    assert is_thin_pack([]) is True


def test_is_thin_pack_low_score() -> None:
    assert is_thin_pack([{"final_score": 0.1}]) is True
    assert is_thin_pack([{"final_score": 0.8}]) is False


def test_reformulate_adds_synonyms_and_domain() -> None:
    route = RouteResult(
        mode="semantic",
        project_code="261",
        expanded_query=ExpandedQuery(
            original="facade strategy",
            normalised="facade strategy",
            concept_tags=["facade"],
            expanded_terms=["facade", "envelope", "cladding"],
        ),
    )
    q1, reason1 = reformulate_query("facade strategy", route, attempt=1)
    assert "envelope" in q1.lower() or "cladding" in q1.lower()
    assert reason1 != "noop"

    q2, reason2 = reformulate_query("facade strategy", route, attempt=2)
    assert "261" in q2
    assert "project-code" in reason2


def test_retry_loop_second_pass_wins() -> None:
    calls: list[str] = []

    def fake_search(q: str, **kwargs):
        calls.append(q)
        if len(calls) == 1:
            return [{"final_score": 0.1, "citation": "a#1"}]
        return [{"final_score": 0.9, "citation": "b#1"}]

    route = RouteResult(
        mode="semantic",
        expanded_query=ExpandedQuery(
            original="dwellings count",
            normalised="dwellings count",
            concept_tags=[],
            expanded_terms=["units", "residential"],
        ),
    )
    outcome = run_search_with_retry(
        query="dwellings count",
        route=route,
        search_fn=fake_search,
        search_kwargs={},
        max_retries=1,
        min_top_score=0.35,
    )
    assert len(calls) == 2
    assert outcome.thin is False
    assert outcome.exhausted is False
    assert outcome.results[0]["final_score"] == 0.9
    assert len(outcome.attempts) == 2


def test_retry_exhausted_returns_honest_answer_helper() -> None:
    def always_thin(q: str, **kwargs):
        return [{"final_score": 0.05, "citation": "x#1"}]

    outcome = run_search_with_retry(
        query="obscure question",
        route=None,
        search_fn=always_thin,
        search_kwargs={},
        max_retries=1,
        min_top_score=0.35,
    )
    assert outcome.exhausted is True
    assert outcome.thin is True
    assert "Not enough evidence" in thin_pack_answer("obscure question")
