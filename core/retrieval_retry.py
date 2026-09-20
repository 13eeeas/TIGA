"""
core/retrieval_retry.py — Agentic retrieval retry on thin evidence packs.

When the first hybrid search comes back thin, reformulate locally
(synonyms + domain expands + project code) and re-search a capped
number of times before compose. Prefer "not enough evidence" over
hallucinating from a weak pack.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from core.retrieval_boost import domain_expand_terms
from core.router import ExpandedQuery, RouteResult

logger = logging.getLogger(__name__)

_THIN_ANSWER = (
    "Not enough evidence in the archive for this question. "
    "Try a project code, different keywords, or a more specific file type."
)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]{1,}", re.I)


@dataclass
class RetryAttempt:
    attempt: int
    query: str
    reason: str
    top_score: float
    result_count: int
    thin: bool


@dataclass
class RetryOutcome:
    results: list[dict[str, Any]]
    query_used: str
    attempts: list[RetryAttempt] = field(default_factory=list)
    exhausted: bool = False
    thin: bool = False


def pack_top_score(results: Sequence[dict[str, Any]]) -> float:
    if not results:
        return 0.0
    return max(float(r.get("final_score") or 0.0) for r in results)


def is_thin_pack(
    results: Sequence[dict[str, Any]],
    *,
    min_results: int = 1,
    min_top_score: float = 0.35,
) -> bool:
    """True when the evidence pack is empty or too weak to compose on."""
    if len(results) < min_results:
        return True
    return pack_top_score(results) < min_top_score


def reformulate_query(
    query: str,
    route: RouteResult | None,
    attempt: int,
) -> tuple[str, str]:
    """
    Build a local reformulation for retry `attempt` (1-based).

    Returns (new_query, reason). Prefer synonyms / domain expands;
    never calls an LLM.
    """
    base = (query or "").strip()
    expanded: ExpandedQuery | None = route.expanded_query if route else None
    reason_parts: list[str] = []
    terms: list[str] = []

    # Keep original tokens first.
    seen: set[str] = set()
    for tok in _TOKEN_RE.findall(base.lower()):
        if tok not in seen:
            seen.add(tok)
            terms.append(tok)

    if attempt >= 1 and expanded and expanded.expanded_terms:
        for t in expanded.expanded_terms:
            tl = t.lower().strip()
            if tl and tl not in seen:
                seen.add(tl)
                terms.append(tl)
                reason_parts.append("synonym-expand")
                if len(reason_parts) >= 4:
                    break

    if attempt >= 1:
        for t in domain_expand_terms(base):
            tl = t.lower().strip()
            if tl and tl not in seen:
                seen.add(tl)
                terms.append(tl)
                reason_parts.append("domain-expand")

    if attempt >= 2 and route and route.project_code:
        code = str(route.project_code).strip()
        if code and code.lower() not in seen:
            terms.insert(0, code)
            reason_parts.append("project-code")

    if not terms:
        return base, "noop"

    # Cap reformulation length so FTS stays focused.
    reformulated = " ".join(terms[:12])
    if reformulated.lower() == base.lower():
        return base, "noop"
    reason = "+".join(dict.fromkeys(reason_parts)) or "expand"
    return reformulated, reason


def select_best_pack(
    packs: list[tuple[str, list[dict[str, Any]]]],
) -> tuple[str, list[dict[str, Any]]]:
    """Pick the pack with the highest top score (then most results)."""
    if not packs:
        return "", []
    best_q, best = packs[0]
    best_score = pack_top_score(best)
    for q, results in packs[1:]:
        score = pack_top_score(results)
        if score > best_score or (
            score == best_score and len(results) > len(best)
        ):
            best_q, best, best_score = q, results, score
    return best_q, best


def run_search_with_retry(
    *,
    query: str,
    route: RouteResult | None,
    search_fn,
    search_kwargs: dict[str, Any],
    max_retries: int = 1,
    min_results: int = 1,
    min_top_score: float = 0.35,
) -> RetryOutcome:
    """
    Search, and if the pack is thin, reformulate and retry up to max_retries.

    `search_fn` must accept (query, **search_kwargs) and return an iterable
    of result dicts (same shape as `core.query.search`).
    """
    attempts: list[RetryAttempt] = []
    packs: list[tuple[str, list[dict[str, Any]]]] = []

    current_q = query
    reason = "initial"

    for attempt in range(0, max(0, max_retries) + 1):
        if attempt > 0:
            current_q, reason = reformulate_query(query, route, attempt)
            if reason == "noop" and attempt > 0:
                # Nothing new to try — stop early.
                break

        raw = list(search_fn(current_q, **search_kwargs) or [])
        top = pack_top_score(raw)
        thin = is_thin_pack(
            raw, min_results=min_results, min_top_score=min_top_score
        )
        attempts.append(
            RetryAttempt(
                attempt=attempt,
                query=current_q,
                reason=reason,
                top_score=round(top, 4),
                result_count=len(raw),
                thin=thin,
            )
        )
        packs.append((current_q, raw))
        logger.info(
            "retrieval_retry attempt=%s thin=%s top=%.3f n=%s reason=%s q=%r",
            attempt, thin, top, len(raw), reason, current_q[:120],
        )
        if not thin:
            return RetryOutcome(
                results=raw,
                query_used=current_q,
                attempts=attempts,
                exhausted=False,
                thin=False,
            )

    best_q, best = select_best_pack(packs)
    still_thin = is_thin_pack(
        best, min_results=min_results, min_top_score=min_top_score
    )
    return RetryOutcome(
        results=best,
        query_used=best_q or query,
        attempts=attempts,
        exhausted=still_thin,
        thin=still_thin,
    )


def thin_pack_answer(query: str) -> str:
    _ = query  # reserved for future personalization
    return _THIN_ANSWER
