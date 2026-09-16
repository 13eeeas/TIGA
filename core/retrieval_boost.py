"""
core/retrieval_boost.py — Google-classic ranking signals for Hunt (no API).

Closes the gap between raw BM25-OR and pre-AI Google-style document search:
  1. Domain paraphrase expansion (dwellings → units / residential)
  2. Phrase-aware FTS query building
  3. Path / filename boost (archives live in folder structure)
  4. Project-code soft filter boost
  5. Near-duplicate suppression (same file + name-family variants)
  6. Exact-filename hints (no silent broaden to all .dwg / CAD)
  7. Critical-token honesty (numbers / rare nouns → unsupported empty pack)

All local — zero API cost. Makes the evidence pack clean before Einstein runs.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# Firm / architecture paraphrase lexicon — query tokens → retrieval expand terms.
# Keep short; FTS query is capped. Synonym YAML still handles concept routing.
_DOMAIN_EXPAND: dict[str, list[str]] = {
    "dwellings": ["units", "residential", "homes", "apartments"],
    "dwelling": ["unit", "residential", "home", "apartment"],
    "homes": ["residential", "units", "housing"],
    "owner": ["client", "developer"],
    "developer": ["client", "owner"],
    "commissioned": ["client", "developer"],
    "sustainability": ["green", "mark", "leed", "environmental"],
    "certification": ["green", "mark", "gold", "platinum"],
    "envelope": ["facade", "façade", "cladding"],
    "facade": ["envelope", "cladding"],
    "spa": ["wellness", "hospitality"],
    "ballroom": ["hospitality", "hotel"],
    "pupils": ["students", "school", "campus"],
    "students": ["pupils", "school", "campus"],
    "education": ["school", "campus", "learning"],
    "contractor": ["tender", "pricing", "boq"],
    "pricing": ["tender", "boq", "contractor"],
    "boq": ["tender", "pricing", "bill"],
    "revision": ["rev", "issue"],
    "tower": ["highrise", "high", "rise", "residential"],
    "scheme": ["project", "development"],
    "accommodation": ["rooms", "keys", "hotel", "units"],
    "wellness": ["spa", "hospitality"],
}

# Multi-word phrases that should be searched as FTS5 phrases when present in query.
_KNOWN_PHRASES: tuple[str, ...] = (
    "green mark",
    "project brief",
    "tender documentation",
    "tender set",
    "competition submission",
    "facade strategy",
    "site analysis",
    "meeting minutes",
    "design development",
    "learning clusters",
    "sports hall",
)

_PROJECT_CODE_RE = re.compile(r"\b(\d{3,4})\b")
_STOP = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "for", "to", "of", "in", "on", "at", "by", "from",
    "with", "what", "which", "who", "when", "where", "why", "how",
    "find", "get", "show", "me", "my", "about", "tell", "give",
    "project", "architecture", "building",
})


def domain_expand_terms(query: str) -> list[str]:
    """Return extra retrieval tokens from domain paraphrase lexicon."""
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    out: list[str] = []
    seen = set(tokens)
    for t in tokens:
        for syn in _DOMAIN_EXPAND.get(t, []):
            if syn not in seen:
                seen.add(syn)
                out.append(syn)
    return out


def extract_phrases(query: str) -> list[str]:
    """Known multi-word phrases present in the query (lowercase)."""
    q = query.lower()
    found: list[str] = []
    for phrase in _KNOWN_PHRASES:
        if phrase in q:
            found.append(phrase)
    # Also treat quoted "..." as phrases
    for m in re.finditer(r'"([^"]{2,80})"', query):
        found.append(m.group(1).strip().lower())
    # Adjacent content bigrams (length ≥2 each, not stopwords)
    tokens = [t for t in re.findall(r"[a-z0-9]+", q) if t not in _STOP and len(t) > 1]
    for i in range(len(tokens) - 1):
        bigram = f"{tokens[i]} {tokens[i + 1]}"
        if bigram not in found and len(bigram) >= 6:
            found.append(bigram)
    # Cap to avoid FTS bloat
    return found[:6]


def detect_project_codes(query: str) -> list[str]:
    return _PROJECT_CODE_RE.findall(query)


def build_fts_query(
    query: str,
    expanded_terms: list[str] | None = None,
    *,
    use_phrases: bool = True,
    use_domain_expand: bool = True,
) -> str:
    """
    Build an FTS5 MATCH string with:
      - original content tokens (OR)
      - synonym / domain expand tokens (OR)
      - phrase clauses for known / bigram phrases (OR of "phrase")
    """
    cleaned = re.sub(r"[^\w\s]", " ", query)
    base_tokens = [
        t for t in cleaned.split()
        if t.lower() not in _STOP and len(t) > 1
    ]
    base_set = {t.lower() for t in base_tokens}

    extra: list[str] = []
    for term in expanded_terms or []:
        for word in re.sub(r"[^\w\s]", " ", term).split():
            w = word.lower()
            if w not in base_set and w not in _STOP and len(w) > 2:
                base_set.add(w)
                extra.append(word)

    if use_domain_expand:
        for syn in domain_expand_terms(query):
            if syn not in base_set and len(syn) > 2:
                base_set.add(syn)
                extra.append(syn)

    clauses: list[str] = []
    # Phrase clauses first (higher specificity — BM25 still ranks)
    if use_phrases:
        for phrase in extract_phrases(query):
            # FTS5 phrase: "green mark"
            safe = phrase.replace('"', "")
            if safe:
                clauses.append(f'"{safe}"')

    all_tokens = (base_tokens + extra)[:40]
    clauses.extend(all_tokens)

    if not clauses:
        return '""'
    return " OR ".join(clauses)


def path_filename_boost(
    query: str,
    file_path: str,
    file_name: str,
    *,
    path_weight: float = 0.18,
    name_weight: float = 0.22,
) -> float:
    """
    Soft multiplicative boost when query tokens appear in path / filename.
    Classic Google-for-files signal — critical for NAS folder archives.
    Returns multiplier ≥ 1.0.
    """
    q_tokens = {
        t for t in re.findall(r"[a-z0-9]+", query.lower())
        if t not in _STOP and len(t) > 1
    }
    if not q_tokens:
        return 1.0

    path_l = (file_path or "").lower().replace("\\", "/")
    name_l = (file_name or "").lower()
    path_hits = sum(1 for t in q_tokens if t in path_l)
    name_hits = sum(1 for t in q_tokens if t in name_l)
    boost = 1.0 + path_weight * (path_hits / max(len(q_tokens), 1))
    boost += name_weight * (name_hits / max(len(q_tokens), 1))
    return min(boost, 1.55)


def project_code_boost(
    query: str,
    project_id: str,
    *,
    weight: float = 0.25,
) -> float:
    """Boost candidates whose project_id matches a code mentioned in the query."""
    codes = detect_project_codes(query)
    if not codes or not project_id:
        return 1.0
    pid = str(project_id).lower()
    for code in codes:
        if code in pid or pid.startswith(code) or pid.endswith(code):
            return 1.0 + weight
        # "261_tianmu" / "2023_HOSP" style
        if pid.startswith(f"{code}_") or pid.startswith(f"{code}-"):
            return 1.0 + weight
    return 1.0


def apply_archive_boosts(
    candidates: list[dict[str, Any]],
    query: str,
    *,
    path_boost: bool = True,
    project_boost: bool = True,
) -> None:
    """Mutate final_score with path/name + project-code signals; re-sort."""
    if not candidates:
        return
    for c in candidates:
        score = float(c.get("final_score", 0.0))
        if path_boost:
            score *= path_filename_boost(
                query,
                str(c.get("file_path", "")),
                str(c.get("file_name", "")),
            )
        if project_boost:
            score *= project_code_boost(query, str(c.get("project_id", "")))
        c["final_score"] = score
    candidates.sort(key=lambda r: (-r["final_score"], r.get("file_path", "")))


# Archive / query vocabulary that should NOT count as claim-critical tokens.
# File-type and project-scope words must stay non-critical so good queries
# like "NUS BIZ3 presentation ppt" / "CAD" / "design brief" keep working.
_NON_CRITICAL = _STOP | frozenset({
    "nus", "biz", "biz3", "approved", "final", "cost", "approval", "approvals",
    "presentation", "presentations", "ppt", "pptx", "powerpoint", "deck", "slides",
    "cad", "dwg", "dxf", "bim", "revit", "design", "brief", "report", "drawing",
    "drawings", "tender", "submission", "document", "documents", "file", "files",
    "folder", "latest", "version", "revision", "issue", "issued", "pdf", "docx",
    "agreement", "contract", "qs", "quantity", "surveyor", "meeting", "minutes",
    "client", "consultant", "architecture", "architectural", "building", "project",
    "base", "set", "package", "documentation", "spec", "specification",
    "discussion", "clarification", "requirement", "requirements", "programme",
    "program", "summary", "evidence", "content", "details", "please", "need",
    "looking", "want", "like", "using", "related", "regarding", "concerning",
    "update", "updates", "current", "status", "info", "information", "check",
    "search", "query", "ask", "asking", "there", "here", "have", "been",
    "model", "site", "residential", "hospitality", "commercial", "education",
    "healthcare", "hospital", "mixed", "masterplan", "hotel", "resort", "office", "school",
    "green", "mark", "gold", "platinum", "leed", "tianmu",
})

_NUMBER_WORDS = frozenset({
    "million", "billion", "trillion", "thousand", "hundred",
})

_FILENAME_TOKEN_RE = re.compile(
    r"\b([a-z0-9][\w.-]{2,80}\.[a-z0-9]{1,12})\b",
    re.IGNORECASE,
)
# Unique archive-style stems: at least two underscore segments (rare in prose).
_UNIQUE_STEM_RE = re.compile(
    r"\b([a-z][a-z0-9]*(?:_[a-z0-9]+){2,})\b",
    re.IGNORECASE,
)

_STEM_NOISE_RE = re.compile(
    r"(?:^|[\s_\-])(?:"
    r"v\d+|rev(?:ision)?[\s_\-]?[a-z0-9]*|r\d+|final|draft|copy|wip|"
    r"issue[\s_\-]?\d+|\d{8}|\d{4}[\s_\-]?\d{2}[\s_\-]?\d{2}"
    r")(?=$|[\s_\-./])",
    re.IGNORECASE,
)


def extract_filename_hints(query: str) -> list[str]:
    """
    Detect exact-filename / unique-stem tokens so file lookup does not
    silently broaden to every matching extension (e.g. all .dwg files).
    Returns stem and full-name hints suitable for path LIKE filters.
    """
    hints: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        t = token.strip().lower()
        if len(t) < 3 or t in seen or t in _NON_CRITICAL:
            return
        seen.add(t)
        hints.append(t)

    for m in _FILENAME_TOKEN_RE.finditer(query):
        full = m.group(1)
        _add(full)
        stem = full.rsplit(".", 1)[0]
        _add(stem)

    for m in _UNIQUE_STEM_RE.finditer(query):
        _add(m.group(1))

    return hints


def extract_critical_tokens(query: str) -> list[str]:
    """
    Claim-critical tokens that evidence must support: specific numbers,
    magnitude words, exact filenames, and rare nouns.
    Common archive / file-type vocabulary is excluded.
    """
    critical: list[str] = []
    seen: set[str] = set()

    def _add(token: str) -> None:
        t = token.lower().strip()
        if not t or t in seen or t in _NON_CRITICAL:
            return
        seen.add(t)
        critical.append(t)

    for hint in extract_filename_hints(query):
        _add(hint)

    for word in re.findall(r"[a-z0-9]+", query.lower()):
        if word.isdigit() and len(word) >= 3:
            _add(word)
        elif word in _NUMBER_WORDS:
            _add(word)
        elif len(word) >= 4 and word not in _NON_CRITICAL and not word.isdigit():
            # Rare-ish content noun (e.g. "moon", "xyzzy") — skip file-type vocab.
            _add(word)

    # Prefer stronger signals first (filenames / numbers already added).
    return critical


def _candidate_haystack(candidate: dict[str, Any]) -> str:
    parts = [
        str(candidate.get("chunk_text") or ""),
        str(candidate.get("snippet") or ""),
        str(candidate.get("file_name") or ""),
        str(candidate.get("file_path") or ""),
        str(candidate.get("rel_path") or ""),
    ]
    return " ".join(parts).lower()


def tokens_supported_by_candidate(
    candidate: dict[str, Any],
    critical_tokens: list[str],
) -> list[str]:
    """Return which critical tokens appear in the candidate evidence."""
    if not critical_tokens:
        return []
    hay = _candidate_haystack(candidate)
    return [t for t in critical_tokens if t in hay]


def assess_evidence_support(
    query: str,
    candidates: list[dict[str, Any]],
    *,
    top_n: int = 8,
) -> dict[str, Any]:
    """
    Honesty signal for API consumers: do top hits actually mention
    query-critical tokens (numbers, rare nouns, exact filenames)?

    Returns:
      {
        "critical_tokens": [...],
        "supported_tokens": [...],
        "unsupported_tokens": [...],
        "support_status": "supported" | "partial" | "unsupported" | "n/a",
      }
    """
    critical = extract_critical_tokens(query)
    if not critical:
        return {
            "critical_tokens": [],
            "supported_tokens": [],
            "unsupported_tokens": [],
            "support_status": "n/a",
        }

    pool = candidates[: max(top_n, 1)]
    supported: list[str] = []
    seen: set[str] = set()
    for c in pool:
        for tok in tokens_supported_by_candidate(c, critical):
            if tok not in seen:
                seen.add(tok)
                supported.append(tok)

    unsupported = [t for t in critical if t not in seen]
    if not supported:
        status = "unsupported"
    elif unsupported:
        status = "partial"
    else:
        status = "supported"

    return {
        "critical_tokens": critical,
        "supported_tokens": supported,
        "unsupported_tokens": unsupported,
        "support_status": status,
    }


def apply_critical_token_honesty(
    candidates: list[dict[str, Any]],
    query: str,
    *,
    empty_when_unsupported: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Demote candidates that lack query-critical tokens. When the whole pack
    fails to support any critical token, optionally return an empty list
    (honest no-support) instead of flooding with generic near-misses.
    """
    assessment = assess_evidence_support(query, candidates)
    critical = assessment["critical_tokens"]
    if not critical or not candidates:
        return candidates, assessment

    for c in candidates:
        hits = tokens_supported_by_candidate(c, critical)
        c["_supported_critical"] = hits
        if not hits:
            # Strong demotion — keep ranking shape but surface unsupported pack.
            c["final_score"] = float(c.get("final_score", 0.0)) * 0.12

    candidates.sort(key=lambda r: (-r["final_score"], r.get("file_path", "")))
    assessment = assess_evidence_support(query, candidates)

    if empty_when_unsupported and assessment["support_status"] == "unsupported":
        return [], assessment
    return candidates, assessment


def name_family_key(file_name: str, file_path: str = "") -> str:
    """
    Normalise presentation / tender variants into one family key so
    RevA / Final / dated copies do not flood the evidence pack.
    """
    name = file_name or Path(file_path.replace("\\", "/")).name
    stem = Path(name).stem.lower()
    cleaned = _STEM_NOISE_RE.sub(" ", stem)
    cleaned = re.sub(r"[^a-z0-9]+", "", cleaned)
    return cleaned or stem or name.lower()


def suppress_near_duplicates(
    candidates: list[dict[str, Any]],
    *,
    max_per_file: int = 2,
    max_per_name_family: int = 1,
) -> list[dict[str, Any]]:
    """
    Keep at most max_per_file chunks per file_id and at most
    max_per_name_family files per normalised name family (already score-sorted).

    Name-family capping stops tender-presentation variants (RevA / Final /
    dated exports) from flooding the top evidence pack.
    """
    if max_per_file <= 0 and max_per_name_family <= 0:
        return candidates

    file_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for c in candidates:
        fid = str(c.get("file_id") or c.get("file_path") or "")
        if max_per_file > 0:
            n = file_counts.get(fid, 0)
            if n >= max_per_file:
                continue
        family = name_family_key(
            str(c.get("file_name") or ""),
            str(c.get("file_path") or ""),
        )
        if max_per_name_family > 0 and family:
            # Only count a new family member when this is the first chunk
            # from this file (later chunks of the same file share the family
            # slot already reserved by max_per_file).
            first_chunk_of_file = file_counts.get(fid, 0) == 0
            if first_chunk_of_file:
                fn = family_counts.get(family, 0)
                if fn >= max_per_name_family:
                    continue
                family_counts[family] = fn + 1

        file_counts[fid] = file_counts.get(fid, 0) + 1
        out.append(c)
    return out
