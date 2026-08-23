"""
core/retrieval_boost.py — Google-classic ranking signals for Hunt (no API).

Closes the gap between raw BM25-OR and pre-AI Google-style document search:
  1. Domain paraphrase expansion (dwellings → units / residential)
  2. Phrase-aware FTS query building
  3. Path / filename boost (archives live in folder structure)
  4. Project-code soft filter boost
  5. Near-duplicate suppression (same file, keep best chunk)

All local — zero API cost. Makes the evidence pack clean before Einstein runs.
"""

from __future__ import annotations

import re
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


def suppress_near_duplicates(
    candidates: list[dict[str, Any]],
    *,
    max_per_file: int = 2,
) -> list[dict[str, Any]]:
    """
    Keep at most max_per_file chunks per file_id (already score-sorted).
    Prevents one long PDF from flooding the evidence pack — Google-style diversity.
    """
    if max_per_file <= 0:
        return candidates
    counts: dict[str, int] = {}
    out: list[dict[str, Any]] = []
    for c in candidates:
        fid = str(c.get("file_id") or c.get("file_path") or "")
        n = counts.get(fid, 0)
        if n >= max_per_file:
            continue
        counts[fid] = n + 1
        out.append(c)
    return out
