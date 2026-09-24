"""
core/compose.py — Answer composer (Einstein synthesis layer).

compose_answer(query, results, session_id, cfg_obj, conn)
  1. Hydrate evidence pack (8–15 chunks, full text capped).
  2. Load session history.
  3. Call enterprise API (or Ollama fallback) for synthesis.
  4. Optional follow-up prompts.
  5. Claim-support verification against the evidence pack.
  6. Persist session + audit log.

Hard rule: citations come from results list only — never from model output.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import cfg as _module_cfg, Config
from core.db import (
    add_message,
    create_session,
    get_connection,
    log_event,
    touch_session,
)
from core.llm_providers import LLMError, chat_completion

logger = logging.getLogger(__name__)

_SESSION_HISTORY = 6
_FALLBACK_MAX = 1200
_FALLBACK_PREFIX = "[Synthesis unavailable — showing raw excerpts] "
_UNSUPPORTED_PREFIX = "Not enough evidence in the archive for: "

_CLAIM_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_CITE_REF_RE = re.compile(r"\[(\d+)\]")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]{1,}", re.I)

_VERIFY_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "for", "to", "of", "in", "on", "at", "by", "from", "with", "into",
    "through", "during", "before", "after", "about", "it", "its", "this",
    "that", "these", "those", "what", "which", "who", "when", "where",
    "why", "how", "all", "each", "every", "both", "few", "more", "most",
    "other", "some", "such", "no", "not", "only", "same", "so", "than",
    "too", "very", "just", "also", "any", "if", "as", "i", "me", "my",
    "you", "your", "he", "she", "they", "their", "them", "we", "our", "us",
    "cannot", "can't", "find", "found", "archive", "evidence", "enough",
    "according", "based", "using", "provided", "source", "sources",
    "project", "projects", "building", "architecture", "architectural",
})

_HEDGE_PHRASES = (
    "cannot find",
    "can't find",
    "could not find",
    "not enough evidence",
    "no relevant",
    "does not contain",
    "don't have enough",
    "do not have enough",
)

_SYSTEM_PROMPT = (
    "You are TIGA Einstein, an architecture firm research assistant. "
    "Answer using ONLY the provided evidence excerpts. "
    "If the evidence does not contain the answer, say you cannot find it in the archive. "
    "Do not invent facts. Reference sources by the citation label shown in each block "
    "(e.g. [1], [2])."
)

_FOLLOWUP_PROMPT = (
    "Suggest exactly 3 short follow-up questions the user might ask next, "
    "based on the query and context above. "
    "Output one question per line, no numbering, no bullet points."
)


@dataclass
class ResultView:
    title:       str
    rel_path:    str
    file_path:   str
    citation:    str
    snippet:     str
    project_id:  str
    typology:    str
    ext:         str
    final_score: float
    evidence_text: str = ""
    support_status: str | None = None  # supported | weak | unsupported
    version_status: str = "unknown"  # current | superseded | unknown

    @classmethod
    def from_search_result(
        cls,
        r: dict[str, Any],
        *,
        evidence_text: str = "",
    ) -> "ResultView":
        rel = r.get("rel_path", "")
        fn = r.get("file_name", rel)
        snippet = r.get("snippet", "")
        ev = evidence_text or r.get("chunk_text") or snippet
        from core.evidence import version_status_from_flags
        return cls(
            title=Path(fn).stem if fn else "",
            rel_path=rel,
            file_path=r.get("file_path", ""),
            citation=r.get("citation", ""),
            snippet=snippet,
            project_id=r.get("project_id", "Unknown"),
            typology=r.get("typology", "Unknown"),
            ext=Path(rel).suffix.lower() if rel else "",
            final_score=float(r.get("final_score", 0.0)),
            evidence_text=ev,
            support_status=r.get("support_status"),
            version_status=version_status_from_flags(
                r.get("is_latest"),
                r.get("is_superseded"),
            ),
        )


@dataclass
class ClaimVerdict:
    claim: str
    status: str  # supported | weak | unsupported
    citation_indexes: list[int] = field(default_factory=list)  # 1-based
    citations: list[str] = field(default_factory=list)
    reason: str = ""


@dataclass
class ComposeResult:
    answer_summary: str
    follow_ups: list[str]
    confidence: float
    results: list[ResultView]
    latency_ms: float
    compose_provider: str = ""
    evidence_chunk_ids: list[str] = field(default_factory=list)
    claim_verdicts: list[ClaimVerdict] = field(default_factory=list)
    verification_summary: str = ""


def _hydrate_chunk_texts(
    conn: sqlite3.Connection,
    results: list[dict[str, Any]],
    char_cap: int,
) -> dict[str, str]:
    ids = [r["chunk_id"] for r in results if r.get("chunk_id")]
    if not ids:
        return {}
    placeholders = ",".join("?" * len(ids))
    rows = conn.execute(
        f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})",
        ids,
    ).fetchall()
    return {
        r["chunk_id"]: (r["text"] or "")[:char_cap]
        for r in rows
    }


def _evidence_views(
    results: list[dict[str, Any]],
    conn: sqlite3.Connection,
    cfg: Config,
) -> list[ResultView]:
    pack_size = cfg.compose_evidence_pack_size
    char_cap = cfg.compose_chunk_char_cap
    subset = results[:pack_size]
    texts = _hydrate_chunk_texts(conn, subset, char_cap)
    views: list[ResultView] = []
    for r in subset:
        cid = r.get("chunk_id", "")
        ev = texts.get(cid) or r.get("chunk_text") or r.get("snippet", "")
        views.append(ResultView.from_search_result(r, evidence_text=ev))
    return views


def _build_context(views: list[ResultView]) -> str:
    if not views:
        return "No relevant documents found."
    parts = []
    for i, v in enumerate(views, 1):
        body = v.evidence_text or v.snippet
        parts.append(f"[{i}] {v.citation}\n{body}")
    return "\n\n".join(parts)


def _confidence(views: list[ResultView]) -> float:
    if not views:
        return 0.0
    scores = [v.final_score for v in views]
    lo, hi = min(scores), max(scores)
    mean = sum(scores) / len(scores)
    if hi == lo:
        return mean
    return (mean - lo) / (hi - lo)


def _load_history(conn: sqlite3.Connection, session_id: str) -> list[dict[str, str]]:
    try:
        rows = conn.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id=? ORDER BY message_id DESC LIMIT ?",
            (session_id, _SESSION_HISTORY),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    except Exception as e:
        logger.warning("Failed to load session history: %s", e)
        return []


def _fallback_answer(views: list[ResultView]) -> str:
    parts = [(v.evidence_text or v.snippet) for v in views[:5]]
    raw = " … ".join(p for p in parts if p)
    return _FALLBACK_PREFIX + raw[:_FALLBACK_MAX]


def _claim_tokens(text: str) -> set[str]:
    return {
        t.lower()
        for t in _TOKEN_RE.findall(text or "")
        if t.lower() not in _VERIFY_STOPWORDS and len(t) > 2
    }


def _split_claims(answer: str) -> list[str]:
    text = (answer or "").strip()
    if not text:
        return []
    parts = [p.strip() for p in _CLAIM_SPLIT_RE.split(text) if p.strip()]
    return parts or [text]


def _is_hedge_claim(claim: str) -> bool:
    lowered = claim.lower()
    return any(p in lowered for p in _HEDGE_PHRASES)


def _overlap_score(claim_toks: set[str], evidence_toks: set[str]) -> float:
    if not claim_toks:
        return 0.0
    hit = len(claim_toks & evidence_toks)
    return hit / len(claim_toks)


def _score_claim(
    claim: str,
    views: list[ResultView],
) -> tuple[str, list[int], str]:
    """Return (status, 1-based citation indexes, reason)."""
    if _is_hedge_claim(claim):
        return "supported", [], "honest hedge / insufficient-evidence statement"

    claim_toks = _claim_tokens(claim)
    cited_indexes = [
        int(m.group(1))
        for m in _CITE_REF_RE.finditer(claim)
        if 1 <= int(m.group(1)) <= len(views)
    ]

    scored: list[tuple[float, int]] = []
    for i, v in enumerate(views, 1):
        body = v.evidence_text or v.snippet or ""
        score = _overlap_score(claim_toks, _claim_tokens(body))
        scored.append((score, i))

    # Prefer explicitly cited evidence when the model attached [n] refs.
    if cited_indexes:
        cited_scores = [s for s, i in scored if i in cited_indexes]
        best = max(cited_scores) if cited_scores else 0.0
        indexes = cited_indexes
    else:
        scored.sort(reverse=True)
        best = scored[0][0] if scored else 0.0
        # Keep evidence indexes with meaningful overlap.
        indexes = [i for s, i in scored if s >= 0.2][:3]

    hit_count = int(round(best * len(claim_toks))) if claim_toks else 0
    if best >= 0.45 or hit_count >= 3:
        return "supported", indexes, f"lexical overlap {best:.2f}"
    if best > 0.2 or hit_count >= 2:
        return "weak", indexes, f"partial overlap {best:.2f}"
    return "unsupported", [], "no usable overlap with evidence pack"


def _apply_support_status(
    display_views: list[ResultView],
    evidence_views: list[ResultView],
    verdicts: list[ClaimVerdict],
) -> None:
    """Aggregate claim statuses onto result rows by citation string."""
    by_cite: dict[str, list[str]] = {}
    for v in verdicts:
        for cite in v.citations:
            by_cite.setdefault(cite, []).append(v.status)

    rank = {"unsupported": 0, "weak": 1, "supported": 2}

    def worst(statuses: list[str]) -> str:
        return min(statuses, key=lambda s: rank.get(s, 1))

    for view in evidence_views:
        statuses = by_cite.get(view.citation)
        if statuses:
            view.support_status = worst(statuses)

    cite_status = {
        v.citation: v.support_status
        for v in evidence_views
        if v.support_status
    }
    for view in display_views:
        if view.citation in cite_status:
            view.support_status = cite_status[view.citation]


def verify_claims(
    answer: str,
    evidence_views: list[ResultView],
) -> tuple[str, list[ClaimVerdict], str]:
    """
    Check each answer claim against the evidence pack.

    Unsupported claims are stripped and replaced with an explicit
    insufficient-evidence note. Returns (rewritten_answer, verdicts, summary).
    """
    claims = _split_claims(answer)
    if not claims:
        return answer, [], ""

    verdicts: list[ClaimVerdict] = []
    kept: list[str] = []
    dropped: list[str] = []

    for claim in claims:
        status, indexes, reason = _score_claim(claim, evidence_views)
        citations = [
            evidence_views[i - 1].citation
            for i in indexes
            if 0 < i <= len(evidence_views)
        ]
        verdicts.append(
            ClaimVerdict(
                claim=claim,
                status=status,
                citation_indexes=indexes,
                citations=citations,
                reason=reason,
            )
        )
        if status == "unsupported":
            dropped.append(claim.rstrip(".!? "))
        else:
            kept.append(claim)

    counts = {
        "supported": sum(1 for v in verdicts if v.status == "supported"),
        "weak": sum(1 for v in verdicts if v.status == "weak"),
        "unsupported": sum(1 for v in verdicts if v.status == "unsupported"),
    }
    summary = (
        f"{counts['supported']} supported, "
        f"{counts['weak']} weak, "
        f"{counts['unsupported']} unsupported"
    )

    if not dropped:
        return answer, verdicts, summary

    parts = list(kept)
    for claim in dropped:
        parts.append(f"{_UNSUPPORTED_PREFIX}{claim}.")
    rewritten = " ".join(parts).strip()
    return rewritten, verdicts, summary


def _call_followups(
    query: str,
    context: str,
    cfg_obj: Config,
) -> list[str]:
    try:
        content, _ = chat_completion(
            [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuery: {query}\n\n{_FOLLOWUP_PROMPT}",
                },
            ],
            cfg_obj,
            purpose="followups",
        )
        lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
        return lines[:3]
    except Exception as e:
        logger.warning("Follow-up generation failed: %s", e)
        return []


def compose_answer(
    query: str,
    results: list[dict[str, Any]],
    session_id: str | None = None,
    cfg_obj: Config | None = None,
    conn: sqlite3.Connection | None = None,
) -> ComposeResult:
    _cfg = cfg_obj or _module_cfg
    _own_conn = conn is None
    _conn = conn or get_connection(_cfg.get_db_path())
    t0 = time.perf_counter()

    try:
        evidence_views = _evidence_views(results, _conn, _cfg)
        display_views = [ResultView.from_search_result(r) for r in results]
        context = _build_context(evidence_views)
        chunk_ids = [r.get("chunk_id", "") for r in results[: _cfg.compose_evidence_pack_size]]

        history: list[dict[str, str]] = []
        if session_id:
            try:
                create_session(_conn, session_id)
                touch_session(_conn, session_id)
                history = _load_history(_conn, session_id)
            except Exception as e:
                logger.warning("Session setup failed: %s", e)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": f"Evidence:\n{context}\n\nQuestion: {query}"},
        ]

        synthesis_ok = False
        provider_label = "fallback"
        try:
            answer_summary, provider_label = chat_completion(
                messages, _cfg, purpose="compose"
            )
            synthesis_ok = True
        except LLMError as e:
            logger.warning("Synthesis unavailable, using fallback: %s", e)
            answer_summary = _fallback_answer(evidence_views)

        follow_ups = (
            _call_followups(query, context, _cfg)
            if synthesis_ok and _cfg.compose_api_enabled
            else []
        )

        claim_verdicts: list[ClaimVerdict] = []
        verification_summary = ""
        if synthesis_ok and getattr(_cfg, "compose_verify_claims", True):
            answer_summary, claim_verdicts, verification_summary = verify_claims(
                answer_summary, evidence_views
            )
            _apply_support_status(display_views, evidence_views, claim_verdicts)

        confidence = _confidence(evidence_views or display_views)
        if any(v.status == "unsupported" for v in claim_verdicts):
            confidence = min(confidence, 0.45)
        elif any(v.status == "weak" for v in claim_verdicts):
            confidence = min(confidence, 0.65)

        if session_id:
            try:
                add_message(_conn, session_id, "user", query)
                citations = [v.citation for v in display_views]
                add_message(_conn, session_id, "assistant", answer_summary, citations)
                touch_session(_conn, session_id)
            except Exception as e:
                logger.warning("Session persistence failed: %s", e)

        latency_ms = (time.perf_counter() - t0) * 1000
        try:
            log_event(
                _conn,
                "compose_answer",
                detail=json.dumps({
                    "latency_ms": round(latency_ms, 1),
                    "provider": provider_label,
                    "evidence_chunks": len(evidence_views),
                    "chunk_ids": chunk_ids,
                    "verification": verification_summary,
                    "claim_counts": {
                        "supported": sum(1 for v in claim_verdicts if v.status == "supported"),
                        "weak": sum(1 for v in claim_verdicts if v.status == "weak"),
                        "unsupported": sum(1 for v in claim_verdicts if v.status == "unsupported"),
                    },
                }),
            )
        except Exception as e:
            logger.warning("Latency log failed: %s", e)

        return ComposeResult(
            answer_summary=answer_summary,
            follow_ups=follow_ups,
            confidence=confidence,
            results=display_views,
            latency_ms=latency_ms,
            compose_provider=provider_label,
            evidence_chunk_ids=[c for c in chunk_ids if c],
            claim_verdicts=claim_verdicts,
            verification_summary=verification_summary,
        )

    finally:
        if _own_conn:
            _conn.close()


# ---------------------------------------------------------------------------
# Streaming API (legacy — used by tiga.py cmd_query)
# ---------------------------------------------------------------------------

def _build_context_block(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No relevant documents found in the archive."
    lines = ["Retrieved documents (ranked by relevance):\n"]
    for i, r in enumerate(results, 1):
        file_name = r.get("file_name") or r.get("file_path", "")
        project = r.get("project_id") or r.get("project", "Unknown")
        typology = r.get("typology", "Unknown")
        preview = r.get("chunk_text") or r.get("snippet") or r.get("surrogate", "")
        citation = r.get("citation") or r.get("file_path", "")
        lines.append(
            f"[{i}] {file_name}\n"
            f"    Project: {project} | Type: {typology}\n"
            f"    Cite as: {citation}\n"
            f"    Preview: {preview[:400]}\n"
        )
    return "\n".join(lines)


def compose_stream(
    query: str,
    results: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
):
    """Yield answer tokens — local Ollama streaming only (CLI legacy)."""
    import ollama as ollama_client

    messages: list[dict[str, str]] = [{"role": "system", "content": _SYSTEM_PROMPT}]
    if history:
        messages.extend(history[-_SESSION_HISTORY:])
    messages.append({
        "role": "user",
        "content": _build_context_block(results) + f"\n\n---\nQuestion: {query}",
    })

    try:
        stream = ollama_client.chat(
            model=_module_cfg.chat_model,
            messages=messages,
            stream=True,
            options={"num_ctx": _module_cfg.num_ctx},
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield token
    except Exception as e:
        logger.error("Ollama compose failed: %s", e)
        yield f"\n\n[Error: Could not reach Ollama. Is it running? {e}]"


def compose(
    query: str,
    results: list[dict[str, Any]],
    history: list[dict[str, str]] | None = None,
) -> str:
    return "".join(compose_stream(query, results, history))
