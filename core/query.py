"""
core/query.py — Hybrid BM25 + vector search engine.

SearchResult (TypedDict)
  chunk_id, file_id, ref_value, rel_path, file_name,
  project_id, typology, snippet (≤160 chars, context around match),
  bm25_score, vector_score, final_score, citation

Six-step pipeline
-----------------
  Step 1 — BM25 lane (FTS5)
    snippet(chunks_fts, 0, '', '', ' ... ', 25) for context snippets.
    Retrieve top_k*3 candidates; apply filters as WHERE clauses.
    Normalise bm25 scores to [0,1] (most-negative raw → 1.0).

  Step 2 — Vector lane (LanceDB/Ollama)
    embed_text(query) → query_vector(top_k*3, filters).
    score = 1 - distance; normalise to [0,1].
    If Ollama unavailable: skip, log warning, BM25-only (no crash).

  Step 3 — Merge + score
    Union of chunk_ids from both lanes.
    final = alpha * vector_score + (1-alpha) * bm25_score
    Sort DESC final_score; tie-break ASC rel_path.
    Apply offset, return top_k.

  Step 4 — Citation generation
    Single root : rel_path#ref_value
    Multi root  : [root_label] rel_path#ref_value  (root_label = root.name)

  Step 5 — Citation validation
    validate_citation(citation, db_path, root_paths) for every candidate.
    Invalid citations → log ERROR + exclude.
    100% of returned results have valid citations.

  Step 6 — Session memory (if session_id given)
    Save user query to messages table.
    (compose.py loads history and prepends to Ollama prompt.)

alpha = cfg.hybrid_alpha  (default 0.6; 1.0 = pure vector, 0.0 = pure BM25)
"""

from __future__ import annotations

import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, TypedDict

from config import cfg as _module_cfg, Config
from core.db import (
    get_connection,
    add_message,
    create_session,
    touch_session,
    file_id_from_path,
)
from core.eval import validate_citation

logger = logging.getLogger(__name__)

_SNIPPET_TOKENS  = 25    # approximate tokens returned by FTS5 snippet()
_SNIPPET_MAX     = 160   # hard cap on returned snippet chars
_SESSION_HISTORY = 6     # messages loaded for compose.py context


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class SearchResult(TypedDict):
    chunk_id:     str
    file_id:      str
    ref_value:    str
    file_path:    str   # absolute path on the server filesystem
    rel_path:     str
    file_name:    str
    project_id:   str
    typology:     str
    snippet:      str
    bm25_score:   float
    vector_score: float
    final_score:  float
    citation:     str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "shall", "can",
    "for", "to", "of", "in", "on", "at", "by", "from", "with", "into",
    "through", "during", "before", "after", "above", "below", "between",
    "out", "up", "down", "about", "it", "its", "this", "that", "these",
    "those", "what", "which", "who", "whom", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "other", "some",
    "such", "no", "not", "only", "same", "so", "than", "too", "very",
    "just", "also", "any", "if", "as", "i", "me", "my", "you", "your",
    "he", "she", "they", "their", "them", "we", "our", "us",
    # Domain stopwords: near-zero IDF in an architecture firm's corpus
    "architecture", "architectural", "building", "project", "find", "get",
})


def _fts_escape(query: str) -> str:
    """
    Sanitise query for FTS5 MATCH syntax.
    Strip stopwords, then join meaningful tokens with OR so that documents
    matching ANY content word are returned (BM25 ranking surfaces best matches).
    """
    cleaned = re.sub(r'[^\w\s]', ' ', query)
    tokens = [t for t in cleaned.split() if t.lower() not in _STOPWORDS and len(t) > 1]
    if not tokens:
        return '""'
    return " OR ".join(tokens)


def _normalise_bm25(scores: list[float]) -> list[float]:
    """
    FTS5 bm25() is negative; more-negative = better match.
    Map to [0, 1]: best (most negative) → 1.0, worst → 0.0.
    """
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(hi - s) / (hi - lo) for s in scores]


def _normalise(scores: list[float]) -> list[float]:
    """Min-max normalise to [0, 1]; higher = better."""
    if not scores:
        return []
    if len(scores) == 1:
        return [1.0]
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [1.0] * len(scores)
    return [(s - lo) / (hi - lo) for s in scores]


def _rel_path(file_path: str, roots: list[Path]) -> str:
    """Return path relative to the first matching index_root, else basename."""
    p = Path(file_path)
    for root in roots:
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            continue
    return p.name


def _make_citation(
    file_path: str,
    rel_path: str,
    ref_value: str,
    roots: list[Path],
) -> str:
    """
    Single root  : "rel_path#ref_value"
    Multiple roots: "[root_label] rel_path#ref_value" where
                    root_label = the last component of whichever root owns the file.
    """
    base = f"{rel_path}#{ref_value}"
    if len(roots) <= 1:
        return base
    p = Path(file_path)
    for root in roots:
        try:
            p.relative_to(root)
            return f"[{root.name}] {base}"
        except ValueError:
            continue
    return base


# ---------------------------------------------------------------------------
# Vector filter compatibility helper
# ---------------------------------------------------------------------------

def _vector_compatible_filters(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """
    Strip filter keys that query_vector() / LanceDB doesn't understand.

    LanceDB's WHERE clause only accepts columns in the tiga_chunks schema:
    project_id, typology, ext.

    Special case for project_scope (list produced by _resolve_project_scope):
    if exactly one project is in scope, convert to a project_id filter.
    For multiple projects, skip — BM25 handles the restriction and the
    hybrid merge down-ranks out-of-scope chunks naturally.
    """
    if not filters:
        return None

    out: dict[str, Any] = {}

    for key in ("project_id", "typology", "ext"):
        if key in filters:
            out[key] = filters[key]

    scope = filters.get("project_scope")
    if scope and len(scope) == 1 and "project_id" not in out:
        out["project_id"] = scope[0]

    return out if out else None


# ---------------------------------------------------------------------------
# Project scope resolver — converts location/typology → file_id allowlist
# ---------------------------------------------------------------------------

def _resolve_project_scope(
    filters: dict[str, Any],
    conn: sqlite3.Connection,
) -> list[str] | None:
    """
    Translate location and/or typology filters into a list of project_id values
    drawn from the project_cards table.

    Returns:
      None         — no location/typology filters present; no scoping applied
      []           — filters matched zero projects (search will yield no results)
      [id, ...]    — list of project_id values to restrict BM25 + vector lanes

    This is the key mechanism that lets "hospitality projects in Singapore"
    search only within the 3-5 matching projects rather than all 30 TB of chunks.
    """
    location = filters.get("location") if filters else None
    typology = filters.get("typology") if filters else None

    if not location and not typology:
        return None  # no project-level scoping needed

    clauses: list[str] = []
    params: list[Any] = []

    if typology:
        clauses.append(
            "(typology_primary LIKE ? OR typology_secondary LIKE ?)"
        )
        params.extend([f"%{typology}%", f"%{typology}%"])

    if location:
        if location == "__overseas__":
            clauses.append(
                "(location IS NOT NULL AND location NOT LIKE '%Singapore%')"
            )
        else:
            clauses.append("location LIKE ?")
            params.append(f"%{location}%")

    sql = (
        "SELECT project_code FROM project_cards WHERE "
        + " AND ".join(clauses)
    )
    try:
        rows = conn.execute(sql, params).fetchall()
        codes = [r["project_code"] for r in rows if r["project_code"]]
        logger.debug(
            "_resolve_project_scope: location=%r typology=%r → %d projects",
            location, typology, len(codes),
        )
        return codes
    except Exception as e:
        logger.warning("_resolve_project_scope failed (skipping scope): %s", e)
        return None


# ---------------------------------------------------------------------------
# BM25 lane
# ---------------------------------------------------------------------------

def _build_boosted_query(query: str, expanded_terms: list[str]) -> str:
    """
    Build a BM25 query string that includes synonym expansion terms.
    The original query tokens are included first (higher weight via OR ordering).
    Expanded synonym terms are appended. Result is capped to avoid FTS5 limits.
    """
    base_tokens = [t for t in re.sub(r'[^\w\s]', ' ', query).split()
                   if t.lower() not in _STOPWORDS and len(t) > 1]
    # Add expanded terms (deduplicated against base tokens)
    base_set = {t.lower() for t in base_tokens}
    extra: list[str] = []
    for term in expanded_terms:
        for word in re.sub(r'[^\w\s]', ' ', term).split():
            w = word.lower()
            if w not in base_set and w not in _STOPWORDS and len(w) > 2:
                base_set.add(w)
                extra.append(word)

    all_tokens = base_tokens + extra
    if not all_tokens:
        return '""'
    # Cap at 40 tokens to keep FTS5 happy
    all_tokens = all_tokens[:40]
    return " OR ".join(all_tokens)


def _run_bm25(
    query: str,
    limit: int,
    filters: dict[str, str] | None,
    conn: sqlite3.Connection,
    expanded_terms: list[str] | None = None,
) -> list[dict[str, Any]]:
    safe_q = _build_boosted_query(query, expanded_terms or []) if expanded_terms else _fts_escape(query)
    params: list[Any] = [safe_q]

    filter_clauses: list[str] = []
    if filters:
        if "project_id" in filters:
            filter_clauses.append("f.project_id = ?")
            params.append(filters["project_id"])
        if "typology" in filters:
            filter_clauses.append("f.typology = ?")
            params.append(filters["typology"])
        if "ext" in filters:
            filter_clauses.append("f.extension = ?")
            params.append(filters["ext"])
        if "root_id" in filters:
            # root_id → prefix match on file_path
            filter_clauses.append("f.file_path LIKE ?")
            params.append(filters["root_id"].rstrip("/") + "/%")
        if "content_type" in filters:
            filter_clauses.append("f.content_type = ?")
            params.append(filters["content_type"])
        if "folder_stage" in filters:
            filter_clauses.append("f.folder_stage = ?")
            params.append(filters["folder_stage"])
        # project_scope: list of project_id values resolved from location/typology
        # filters against project_cards.  An empty list means "match nothing".
        if "project_scope" in filters:
            scope_ids = filters["project_scope"]
            if scope_ids:
                placeholders = ",".join("?" * len(scope_ids))
                filter_clauses.append(f"f.project_id IN ({placeholders})")
                params.extend(scope_ids)
            else:
                filter_clauses.append("1=0")  # no matching projects → no results

    extra = ("AND " + " AND ".join(filter_clauses)) if filter_clauses else ""

    sql = f"""
        SELECT
            c.chunk_id, c.file_id, c.ref_value, c.text,
            f.file_path,
            COALESCE(f.project_id, 'Unknown') AS project_id,
            COALESCE(f.typology,   'Unknown') AS typology,
            COALESCE(f.extension,  '')        AS extension,
            COALESCE(f.file_name,  '')        AS file_name,
            bm25(chunks_fts)                  AS bm25_raw,
            snippet(chunks_fts, 0, '', '', ' ... ', {_SNIPPET_TOKENS}) AS snippet
        FROM chunks_fts
        JOIN chunks c ON c.rowid = chunks_fts.rowid
        JOIN files  f ON f.file_id = c.file_id
        WHERE chunks_fts MATCH ?
          {extra}
        ORDER BY bm25_raw
        LIMIT ?
    """
    params.append(limit)
    try:
        rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]
    except sqlite3.OperationalError as e:
        logger.warning("FTS query failed (%s): %s", safe_q, e)
        return []


# ---------------------------------------------------------------------------
# Vector lane
# ---------------------------------------------------------------------------

def _run_vector(
    query: str,
    limit: int,
    filters: dict[str, str] | None,
    cfg_obj: Config,
) -> dict[str, float]:
    """
    Returns {chunk_id: normalised_score}.
    Returns {} on any failure (Ollama down, table empty, etc.).
    """
    try:
        from core.vectors import query_vector
        results = query_vector(query, limit, cfg=cfg_obj, filters=filters)
        if not results:
            return {}
        raw = [max(0.0, 1.0 - r["distance"]) for r in results]
        norm = _normalise(raw)
        return {r["chunk_id"]: ns for r, ns in zip(results, norm)}
    except Exception as e:
        logger.warning("Vector lane unavailable (BM25-only fallback): %s", e)
        return {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def search(
    query: str,
    top_k: int | None = None,
    offset: int = 0,
    filters: dict[str, str] | None = None,
    session_id: str | None = None,
    cfg_obj: Config | None = None,
    conn: sqlite3.Connection | None = None,
    expanded_terms: list[str] | None = None,
) -> list[SearchResult]:
    """
    Hybrid BM25 + vector search. Returns validated-citation results only.

    Args:
        query:      Natural-language search query.
        top_k:      Max results to return (default: cfg.top_k).
        offset:     Skip first N ranked results (pagination).
        filters:    Optional {project_id, typology, ext, root_id}.
        session_id: If set, saves query to messages table.
        cfg_obj:    Config override (for tests).
        conn:       SQLite connection override (for tests).

    Returns:
        list[SearchResult] — 100% valid citations, sorted by final_score DESC.
    """
    _cfg = cfg_obj or _module_cfg
    _top_k = top_k or _cfg.top_k
    _own_conn = conn is None
    _conn = conn or get_connection(_cfg.get_db_path())

    try:
        return _search_impl(query, _top_k, offset, filters, session_id, _cfg, _conn,
                            expanded_terms=expanded_terms)
    finally:
        if _own_conn:
            _conn.close()


def _search_impl(
    query: str,
    top_k: int,
    offset: int,
    filters: dict[str, str] | None,
    session_id: str | None,
    _cfg: Config,
    conn: sqlite3.Connection,
    expanded_terms: list[str] | None = None,
) -> list[SearchResult]:

    alpha = _cfg.hybrid_alpha
    roots = _cfg.index_roots
    db_path = str(_cfg.get_db_path())
    root_paths = [str(r) for r in roots]

    # ── Pre-step: Resolve project scope from location/typology filters ────────
    # If the router extracted location ("Singapore") or typology ("hospitality")
    # filters, translate them into a list of project_id values from project_cards
    # BEFORE running BM25 + vector search.  This scopes both lanes to just the
    # matching projects rather than the full 30 TB corpus.
    _active_filters: dict[str, Any] | None = dict(filters) if filters else None

    if _active_filters and (
        _active_filters.get("location") or _active_filters.get("typology")
    ):
        scope = _resolve_project_scope(_active_filters, conn)
        if scope is not None:
            _active_filters["project_scope"] = scope
        # Strip project-level keys — BM25 handles them via project_scope; the
        # raw location/typology strings are not file-table columns.
        _active_filters.pop("location", None)
        _active_filters.pop("typology", None)

    # ── Step 1: BM25 (with synonym boosting) ─────────────────────────────────
    bm25_rows = _run_bm25(query, top_k * 3, _active_filters, conn,
                          expanded_terms=expanded_terms)
    bm25_map: dict[str, dict[str, Any]] = {}
    if bm25_rows:
        raw_scores = [r["bm25_raw"] for r in bm25_rows]
        norm_scores = _normalise_bm25(raw_scores)
        for row, ns in zip(bm25_rows, norm_scores):
            row["bm25_score"] = ns
            bm25_map[row["chunk_id"]] = row

    # ── Step 2: Vector (with synonym expansion appended) ─────────────────────
    # Append expanded terms to the query string (cap at 50 words to avoid
    # embedding degradation as per Chunk 7 spec)
    vector_query = query
    if expanded_terms:
        extra_words = " ".join(expanded_terms[:30])  # keep manageable
        combined = f"{query} {extra_words}"
        vector_query = " ".join(combined.split()[:50])
    # Strip BM25-only filter keys (project_scope, content_type, folder_stage)
    # before passing to the vector lane — LanceDB only accepts project_id,
    # typology, ext.  _vector_compatible_filters() handles the conversion.
    vec_scores = _run_vector(
        vector_query, top_k * 3, _vector_compatible_filters(_active_filters), _cfg
    )

    # ── Step 3: Merge ─────────────────────────────────────────────────────────
    all_ids = set(bm25_map) | set(vec_scores)

    # Fetch metadata for vector-only chunks (not in BM25 results)
    missing = all_ids - set(bm25_map)
    if missing:
        placeholders = ",".join("?" * len(missing))
        extra = conn.execute(
            f"SELECT c.chunk_id, c.file_id, c.ref_value, c.text, "
            f"f.file_path, "
            f"COALESCE(f.project_id,'Unknown') AS project_id, "
            f"COALESCE(f.typology,'Unknown')   AS typology, "
            f"COALESCE(f.extension,'')         AS extension, "
            f"COALESCE(f.file_name,'')         AS file_name "
            f"FROM chunks c "
            f"JOIN files f ON f.file_id = c.file_id "
            f"WHERE c.chunk_id IN ({placeholders})",
            list(missing),
        ).fetchall()
        for row in extra:
            bm25_map[row["chunk_id"]] = {
                **dict(row),
                "bm25_raw":   0.0,
                "bm25_score": 0.0,
                "snippet":    row["text"][:_SNIPPET_MAX],
            }

    # Score + sort
    candidates: list[dict[str, Any]] = []
    for cid in all_ids:
        meta = bm25_map.get(cid)
        if not meta:
            continue
        bs = meta.get("bm25_score", 0.0)
        vs = vec_scores.get(cid, 0.0)
        final = alpha * vs + (1.0 - alpha) * bs
        candidates.append({
            "chunk_id":   cid,
            "file_id":    meta["file_id"],
            "ref_value":  meta["ref_value"],
            "file_path":  meta["file_path"],
            "file_name":  meta.get("file_name", Path(meta["file_path"]).name),
            "project_id": meta.get("project_id", "Unknown"),
            "typology":   meta.get("typology", "Unknown"),
            "snippet":    meta.get("snippet", meta.get("text", ""))[:_SNIPPET_MAX],
            "bm25_score": bs,
            "vector_score": vs,
            "final_score":  final,
        })

    candidates.sort(key=lambda r: (-r["final_score"], r["file_path"]))

    # ── Step 3b: Cross-encoder reranking (optional) ────────────────────────────
    # When enabled, a cross-encoder reads (query, chunk) together and replaces
    # the hybrid score with a more accurate relevance score.  This is the single
    # biggest quality jump toward ChatGPT Projects-style precision.
    # Enabled via: retrieval.reranker_enabled: true in config.yaml
    # Install:     pip install sentence-transformers
    if _cfg.reranker_enabled and candidates:
        try:
            from core.reranker import rerank_candidates
            candidates = rerank_candidates(
                query,
                candidates,
                top_k=_cfg.reranker_top_k,
                model_name=_cfg.reranker_model,
            )
        except Exception as _re_exc:
            logger.warning("reranker unavailable (using hybrid order): %s", _re_exc)

    # ── Steps 4 & 5: Citation generation + validation ─────────────────────────
    results: list[SearchResult] = []
    for cand in candidates[offset:]:
        rel = _rel_path(cand["file_path"], roots)
        citation = _make_citation(cand["file_path"], rel, cand["ref_value"], roots)

        if not validate_citation(citation, db_path, root_paths):
            logger.error("Invalid citation excluded from results: %s", citation)
            continue

        results.append(SearchResult(
            chunk_id=     cand["chunk_id"],
            file_id=      cand["file_id"],
            ref_value=    cand["ref_value"],
            file_path=    cand["file_path"],
            rel_path=     rel,
            file_name=    cand["file_name"],
            project_id=   cand["project_id"],
            typology=     cand["typology"],
            snippet=      cand["snippet"],
            bm25_score=   round(cand["bm25_score"],   4),
            vector_score= round(cand["vector_score"],  4),
            final_score=  round(cand["final_score"],   4),
            citation=     citation,
        ))

        if len(results) >= top_k:
            break

    # ── Step 6: Session memory ─────────────────────────────────────────────────
    if session_id:
        try:
            create_session(conn, session_id)
            touch_session(conn, session_id)
            add_message(conn, session_id, "user", query)
        except Exception as e:
            logger.warning("Session save failed: %s", e)

    return results


# ---------------------------------------------------------------------------
# Structured / File-locator / Cross-project executors (Chunk 3)
# ---------------------------------------------------------------------------

def execute_structured_query(
    route,   # RouteResult
    conn: sqlite3.Connection | None = None,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    """
    Query the project_cards table using route.filters.
    Returns a structured answer dict — no LLM synthesis.
    """
    _cfg = cfg_obj or _module_cfg
    _own = conn is None
    _conn = conn or get_connection(_cfg.get_db_path())
    try:
        from core.project_card import get_project_card, list_project_cards

        filters = route.filters or {}
        project_code = route.project_code or filters.get("project_code")

        if project_code:
            card = get_project_card(project_code, conn=_conn)
            if card:
                answer = _format_card_answer(card)
                return {
                    "answer_text": answer,
                    "data": card,
                    "mode": "structured",
                    "sources": ["project_cards_db"],
                    "confidence": _card_confidence(card),
                }
            return {
                "answer_text": f"No project card found for '{project_code}'. "
                               f"Try: python tiga.py card {project_code}",
                "data": None,
                "mode": "structured",
                "sources": [],
                "confidence": 0.0,
            }
        else:
            # Filter query — return matching projects
            card_filters = {k: v for k, v in filters.items()
                            if k in ("typology", "stage", "gfa_min", "gfa_max", "location")}
            cards = list_project_cards(card_filters, conn=_conn)
            if not cards:
                return {
                    "answer_text": "No projects found matching those criteria.",
                    "data": {"projects": []},
                    "mode": "structured",
                    "sources": ["project_cards_db"],
                    "confidence": 0.3,
                }
            summary = f"Found {len(cards)} project(s):\n"
            for c in cards[:10]:
                summary += f"  • {c.get('project_code','')} — {c.get('name', 'unnamed')} "
                summary += f"({c.get('typology_primary','?')}, {c.get('stage','?')})\n"
            return {
                "answer_text": summary.strip(),
                "data": {"projects": cards},
                "mode": "structured",
                "sources": ["project_cards_db"],
                "confidence": 0.9,
            }
    finally:
        if _own:
            _conn.close()


def _format_card_answer(card: dict[str, Any]) -> str:
    """Format a project card as a readable answer string."""
    lines: list[str] = []
    code = card.get("project_code", "")
    name = card.get("name") or "(name not set)"
    lines.append(f"**{code} — {name}**")
    if card.get("location"):
        lines.append(f"Location: {card['location']}")
    if card.get("typology_primary"):
        t = card["typology_primary"]
        if card.get("typology_secondary"):
            t += f" / {card['typology_secondary']}"
        lines.append(f"Typology: {t}")
    if card.get("stage"):
        lines.append(f"Stage: {card['stage']}")
    if card.get("client"):
        lines.append(f"Client: {card['client']}")
    if card.get("gfa_sqm"):
        lines.append(f"GFA: {card['gfa_sqm']:,.0f} sqm")
    if card.get("site_area_sqm"):
        lines.append(f"Site area: {card['site_area_sqm']:,.0f} sqm")
    if card.get("storeys_above"):
        storeys = str(card["storeys_above"])
        if card.get("storeys_below"):
            storeys += f" above / {card['storeys_below']} below grade"
        lines.append(f"Storeys: {storeys}")
    if card.get("units"):
        lines.append(f"Units: {card['units']}")
    if card.get("architect"):
        lines.append(f"Lead architect: {card['architect']}")
    if card.get("pm_job_captain"):
        lines.append(f"PM / Job captain: {card['pm_job_captain']}")
    if card.get("concept_summary"):
        lines.append(f"\nConcept: {card['concept_summary'][:300]}")
    return "\n".join(lines)


def _card_confidence(card: dict[str, Any]) -> float:
    """Compute confidence for a structured answer based on data_sources."""
    ds = card.get("data_sources") or {}
    if not ds:
        return 0.3
    sources = [v.get("source", "inferred") for v in ds.values() if isinstance(v, dict)]
    if not sources:
        return 0.3
    if all(s == "manual" for s in sources):
        return 1.0
    if any(s == "manual" for s in sources):
        return 0.85
    if any(s == "woha_web" for s in sources):
        return 0.75
    # All inferred
    return 0.55


def execute_file_locator_query(
    route,   # RouteResult
    conn: sqlite3.Connection | None = None,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    """
    Query the files table using route.filters.
    Returns file paths — no LLM synthesis.
    """
    _cfg = cfg_obj or _module_cfg
    _own = conn is None
    _conn = conn or get_connection(_cfg.get_db_path())
    try:
        filters = route.filters or {}
        clauses: list[str] = ["status = 'INDEXED'"]
        params: list[Any] = []

        project_code = route.project_code or filters.get("project_code")
        if project_code:
            # Match against project_code column OR project_id column
            clauses.append("(project_code = ? OR project_id LIKE ?)")
            params.extend([project_code, f"%{project_code}%"])
        if "content_type" in filters:
            clauses.append("content_type = ?")
            params.append(filters["content_type"])
        if "folder_stage" in filters:
            clauses.append("folder_stage = ?")
            params.append(filters["folder_stage"])
        if "is_issued" in filters:
            clauses.append("is_issued = ?")
            params.append(int(filters["is_issued"]))
        if "is_superseded" in filters:
            clauses.append("is_superseded = ?")
            params.append(int(filters["is_superseded"]))
        if "is_latest" in filters:
            clauses.append("is_latest = ?")
            params.append(int(filters["is_latest"]))
        if "date_from" in filters:
            clauses.append("(file_date >= ? OR file_date IS NULL)")
            params.append(filters["date_from"])
        if "date_to" in filters:
            clauses.append("(file_date <= ? OR file_date IS NULL)")
            params.append(filters["date_to"])

        where = "WHERE " + " AND ".join(clauses)
        rows = _conn.execute(
            f"""SELECT file_id, file_path, file_name, extension,
                       project_code, project_id, folder_stage, discipline,
                       doc_type, revision, file_date, content_type,
                       is_issued, is_superseded, is_latest, updated_at
                FROM files {where}
                ORDER BY is_latest DESC, file_date DESC, revision DESC
                LIMIT 30""",
            params,
        ).fetchall()

        if not rows:
            conf = 0.0
            answer = "No files found matching those criteria."
        else:
            conf = 1.0 if len(filters) >= 2 else 0.7
            answer = f"Found {len(rows)} file(s)"
            specific = filters.get("content_type") or filters.get("folder_stage")
            if specific:
                answer += f" matching {specific}"
            answer += "."

        files = [dict(r) for r in rows]
        return {
            "answer_text": answer,
            "files": files,
            "mode": "file_locator",
            "sources": [f["file_path"] for f in files[:5]],
            "confidence": conf,
        }
    finally:
        if _own:
            _conn.close()


def execute_cross_project_query(
    route,   # RouteResult
    conn: sqlite3.Connection | None = None,
    cfg_obj: Config | None = None,
) -> dict[str, Any]:
    """
    Query across multiple projects using project_cards table filters,
    then optionally run semantic search per matching project.
    """
    _cfg = cfg_obj or _module_cfg
    _own = conn is None
    _conn = conn or get_connection(_cfg.get_db_path())
    try:
        from core.project_card import list_project_cards

        filters = route.filters or {}
        card_filters = {k: v for k, v in filters.items()
                        if k in ("typology", "stage", "gfa_min", "gfa_max", "location")}
        cards = list_project_cards(card_filters, conn=_conn)

        if not cards:
            return {
                "answer_text": "No projects found matching those criteria.",
                "projects_matched": [],
                "results_per_project": {},
                "mode": "cross_project",
                "sources": [],
                "confidence": 0.3,
            }

        answer = f"Found {len(cards)} project(s) matching criteria:\n"
        for c in cards[:15]:
            answer += (f"  • {c.get('project_code', '')} — "
                       f"{c.get('name', 'unnamed')} "
                       f"({c.get('typology_primary', '?')})\n")

        return {
            "answer_text": answer.strip(),
            "projects_matched": [c.get("project_code") for c in cards],
            "results_per_project": {c.get("project_code"): c for c in cards},
            "mode": "cross_project",
            "sources": ["project_cards_db"],
            "confidence": 0.85 if cards else 0.3,
        }
    finally:
        if _own:
            _conn.close()


# ---------------------------------------------------------------------------
# Session history loader (called by compose.py)
# ---------------------------------------------------------------------------

def load_session_history(
    session_id: str,
    cfg_obj: Config | None = None,
    conn: sqlite3.Connection | None = None,
) -> list[dict[str, str]]:
    """
    Return last _SESSION_HISTORY messages as [{role, content}] for Ollama.
    Returns [] on any error.
    """
    _cfg = cfg_obj or _module_cfg
    _own = conn is None
    _conn = conn or get_connection(_cfg.get_db_path())
    try:
        rows = _conn.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id=? ORDER BY message_id DESC LIMIT ?",
            (session_id, _SESSION_HISTORY),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
    except Exception as e:
        logger.warning("Failed to load session history: %s", e)
        return []
    finally:
        if _own:
            _conn.close()
