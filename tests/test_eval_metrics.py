"""tests/test_eval_metrics.py — Recall@k / NDCG / citation precision helpers."""

from __future__ import annotations

from core.eval import citation_precision, recall_at_k, simple_ndcg_at_k
from core.query import apply_project_autoscope, wants_project_scope_escape


def test_recall_at_k() -> None:
    returned = ["/a/brief.pdf", "/b/other.pdf", "/a/tender.pdf"]
    expected = ["brief.pdf", "tender.pdf"]
    assert recall_at_k(returned, expected, 3) == 1.0
    assert recall_at_k(returned, expected, 1) == 0.5


def test_simple_ndcg_perfect_and_zero() -> None:
    expected = ["brief.pdf"]
    perfect = ["/x/brief.pdf", "/y/noise.pdf"]
    miss = ["/y/noise.pdf", "/z/noise2.pdf"]
    assert simple_ndcg_at_k(perfect, expected, 5) == 1.0
    assert simple_ndcg_at_k(miss, expected, 5) == 0.0


def test_citation_precision() -> None:
    returned = ["proj/brief.pdf#p1", "proj/noise.pdf#p2"]
    expected = ["proj/brief.pdf"]
    assert citation_precision(returned, expected) == 0.5


def test_autoscope_sets_path_filter() -> None:
    scoped = apply_project_autoscope({}, "261", query="261 brief")
    assert scoped["project_path_contains"] == "261"


def test_autoscope_escape_hatch() -> None:
    assert wants_project_scope_escape("compare across projects")
    escaped = apply_project_autoscope({}, "261", query="compare across projects")
    assert "project_path_contains" not in escaped
    assert escaped.get("scope") == "all"
