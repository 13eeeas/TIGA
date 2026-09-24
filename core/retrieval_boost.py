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
    "latest", "please", "need", "between", "versus", "vs", "not",
    "covering", "called", "using", "showing", "show",
    "can", "you", "your",
    "issued", "sent", "compared", "across", "prefer", "among",
    "current", "inside", "within", "during", "this", "that",
    "pull", "want", "looking", "says", "saying",
})

# Words that name a document. When one is present, do not paraphrase
# the rest of the query into nearby materials (facade → cladding).
_DOC_WORDS = frozenset({
    "minutes", "minute", "brief", "drawing", "drawings", "schedule",
    "specification", "specifications", "spec", "tender", "report",
    "submission", "booklet", "email",
})

_COMMON_WORDS = frozenset({
    "brief", "drawing", "drawings", "minutes", "minute", "meeting",
    "tender", "design", "report", "schedule", "level", "sheet",
    "revision", "hotel", "project", "file", "document", "drawings",
})


def content_tokens(query: str) -> list[str]:
    """Words worth searching, filler removed, drawing codes kept."""
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z0-9]+", query or ""):
        word = raw.lower()
        if word in _STOP:
            continue
        if len(word) <= 2 and not any(ch.isdigit() for ch in word):
            continue
        tokens.append(raw)
    return tokens


def name_match_tokens(query: str) -> list[str]:
    """Tokens a file name should contain.

    When the question names a document, those words are required. A glazing
    specification is not a meeting-minutes file.
    """
    phrases = extract_phrases(query or "", include_bigrams=False)
    proper = distinctive_tokens(query, limit=3)
    if not phrases:
        return proper
    words = phrases[0].split()
    for token in proper:
        if token.lower() not in phrases[0].lower() and token not in words:
            words.append(token)
            break
    return words


def distinctive_tokens(query: str, *, limit: int = 3) -> list[str]:
    """The specific words in a sentence: names and codes before generic ones."""
    tokens = content_tokens(query)
    proper = [t for t in tokens if t.lower() not in _COMMON_WORDS]
    ranked = sorted(proper or tokens, key=lambda t: (-len(t), t.lower()))
    return ranked[:limit]


def domain_expand_terms(query: str) -> list[str]:
    """Return extra retrieval tokens from domain paraphrase lexicon.

    A query that already names a document (minutes, brief, drawing) is not
    paraphrased. "Facade meeting minutes" must not become a cladding search.
    """
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    if _DOC_WORDS.intersection(tokens):
        return []
    out: list[str] = []
    seen = set(tokens)
    for t in tokens:
        for syn in _DOMAIN_EXPAND.get(t, []):
            if syn not in seen:
                seen.add(syn)
                out.append(syn)
    return out


def extract_phrases(query: str, *, include_bigrams: bool = False) -> list[str]:
    """Known multi-word phrases present in the query (lowercase)."""
    q = query.lower()
    found: list[str] = []
    for phrase in _KNOWN_PHRASES:
        if phrase in q:
            found.append(phrase)
    # Also treat quoted "..." as phrases
    for m in re.finditer(r'"([^"]{2,80})"', query):
        found.append(m.group(1).strip().lower())
    if include_bigrams:
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
    mode: str = "and_phrase",
) -> str:
    """
    Build an FTS5 MATCH string.

    Default ``and_phrase`` (tighter):
      - content tokens combined with AND (not OR sprawl)
      - known phrases as OR alternatives for recall
      - at most a few domain/synonym expands as OR alternatives

    Legacy ``or_legacy``: previous OR-of-everything behaviour.
    A phrase in quotes is kept as a phrase.
    """
    quoted = re.findall(r'"([^"]{2,80})"', query or "")
    bare = re.sub(r'"[^"]*"', " ", query or "")
    cleaned = re.sub(r"[^\w\s]", " ", bare)
    base_tokens = [
        t for t in cleaned.split()
        if t.lower() not in _STOP and len(t) > 1
    ]
    # A sentence is the few specific words, not every word ANDed together.
    required = distinctive_tokens(bare, limit=3)

    extra: list[str] = []
    seen = {t.lower() for t in required}
    for term in expanded_terms or []:
        for word in re.sub(r"[^\w\s]", " ", term).split():
            w = word.lower()
            if w not in seen and w not in _STOP and len(w) > 2:
                seen.add(w)
                extra.append(word)

    if use_domain_expand:
        for syn in domain_expand_terms(query):
            if syn not in seen and len(syn) > 2:
                seen.add(syn)
                extra.append(syn)

    if mode == "or_legacy":
        clauses: list[str] = []
        if use_phrases:
            for phrase in extract_phrases(query, include_bigrams=True):
                safe = phrase.replace('"', "")
                if safe:
                    clauses.append(f'"{safe}"')
        clauses.extend((base_tokens + extra)[:40])
        return " OR ".join(clauses) if clauses else '""'

    # and_phrase (default)
    alts: list[str] = []
    phrases = []
    if use_phrases:
        phrases = [
            re.sub(r"[^\w\s]", " ", phrase).strip()
            for phrase in list(quoted) + extract_phrases(query, include_bigrams=False)
        ]
        phrases = [phrase for phrase in phrases if phrase]
    if phrases:
        head = phrases[0]
        extra_word = next(
            (token for token in required if token.lower() not in head.lower()),
            "",
        )
        if extra_word:
            alts.append(f'("{head}" AND {extra_word})')
        alts.extend(f'"{phrase}"' for phrase in phrases[:2])
    elif required:
        if len(required) == 1:
            alts.append(required[0])
        else:
            alts.append("(" + " AND ".join(required) + ")")

    # A named document is not widened into synonyms.
    if not phrases:
        for term in extra[:3]:
            alts.append(term)

    if not alts:
        return '""'
    if len(alts) == 1:
        return alts[0]
    return " OR ".join(alts)


def apply_document_rank(candidates: list[dict[str, Any]], query: str) -> None:
    """Prefer a file that is the document asked for, and sink lookalikes.

    A design brief outranks a contract, a wellness deck, or a site photo that
    only shares the word. Mutates final_score and re-sorts.
    """
    q = (query or "").lower()
    kind = ""
    if "drawing list" in q or "drawings list" in q:
        kind = "drawing list"
    elif "minute" in q:
        kind = "minutes"
    elif "schedule" in q:
        kind = "schedule"
    elif "specification" in q or re.search(r"\bspec\b", q):
        kind = "specification"
    elif "brief" in q:
        kind = "brief"
    elif "report" in q:
        kind = "report"
    if not kind or not candidates:
        return

    name_marks = {
        "minutes": ("minute",),
        "schedule": ("schedule",),
        "specification": ("specification", " spec"),
        "report": ("report",),
        "brief": ("brief",),
        "drawing list": ("drawing list", "dwg list", "drawings list"),
    }[kind]
    junk = (
        "shortcut",
        "infopedia",
        "whatsapp",
        "list views on sheet",
        "site progress pic",
        "progress pic",
    )
    brief_impostors = (
        "redas",
        "conditions of",
        "wellness",
        "programme",
        "program",
        "phasing",
    )
    for candidate in candidates:
        name = f"{candidate.get('file_name') or ''} {candidate.get('file_path') or ''}".lower()
        if any(mark in name for mark in name_marks):
            candidate["final_score"] = max(float(candidate.get("final_score") or 0), 2.2)
            if kind == "brief" and "design brief" in name:
                candidate["final_score"] += 0.6
        if any(mark in name for mark in junk) or "logo" in name:
            candidate["final_score"] = float(candidate.get("final_score") or 0) * 0.2
        if kind == "report" and any(mark in name for mark in (" pic", "photo", ".jpg", ".png")):
            candidate["final_score"] = float(candidate.get("final_score") or 0) * 0.2
        if kind == "drawing list" and "drawing" not in name:
            candidate["final_score"] = float(candidate.get("final_score") or 0) * 0.3
        if kind == "brief" and any(mark in name for mark in brief_impostors):
            candidate["final_score"] = float(candidate.get("final_score") or 0) * 0.35
        if kind == "drawing list" and "list views" in name:
            candidate["final_score"] = float(candidate.get("final_score") or 0) * 0.2
    candidates.sort(key=lambda row: (-float(row.get("final_score") or 0), row.get("file_path") or ""))


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
