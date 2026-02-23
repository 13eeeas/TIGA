"""
core/router.py — Query classifier and router with synonym expansion.

Classifies every incoming query into one of four modes:
  - "structured"    : identity/scale/stakeholder queries → projects table
  - "file_locator"  : find-file queries → files table (no LLM)
  - "semantic"      : design/technical/narrative queries → hybrid RAG
  - "cross_project" : multi-project comparison queries

Routing pipeline:
  1. expand_query() — map query to canonical concept_tags via query_synonyms.yml
  2. Score each mode based on concept_tags + raw keyword fallback
  3. Detect project code, parse mode-specific filters
  4. Return RouteResult

Public API
----------
  router = QueryRouter()
  result = router.classify(query, project_code=None)
  expanded = router.expand_query(query)   # concept_tags, expanded_terms
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExpandedQuery:
    original: str           # unchanged original query
    normalised: str         # lowercased, stripped, punctuation removed
    concept_tags: list[str] # matched canonical concept names
    expanded_terms: list[str]  # all synonyms for matched concepts (for FTS boost)


@dataclass
class RouteResult:
    mode: str             # "structured" | "file_locator" | "semantic" | "cross_project"
    project_code: str | None = None
    filters: dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    reason: str = ""
    secondary_mode: str | None = None   # populated when confidence < 0.7
    expanded_query: ExpandedQuery | None = None  # synonym expansion result


# ---------------------------------------------------------------------------
# Concept → mode mapping
# ---------------------------------------------------------------------------

# Concepts that route to STRUCTURED (project data card)
_STRUCTURED_CONCEPTS = frozenset([
    "project_overview", "typology", "client", "project_stage",
    "gfa", "storeys", "units", "plot_ratio",
    "project_architect", "pm", "consultants", "contractor",
])

# Concepts that route to FILE_LOCATOR (find files on disk)
_FILE_LOCATOR_CONCEPTS = frozenset([
    "bim_model", "cad_drawings", "ifc_set", "renders", "presentations",
    "transmittals", "meeting_minutes", "specifications", "emails",
])

# Concepts that route to SEMANTIC (RAG search)
_SEMANTIC_CONCEPTS = frozenset([
    "design_concept", "facade", "sustainability", "materials",
    "waivers", "action_items", "authority_comments",
])

# Concept → content_type filter for file_locator
_CONCEPT_CONTENT_TYPE: dict[str, str] = {
    "bim_model":      "BIM",
    "cad_drawings":   "CAD",
    "renders":        "Image",
    "presentations":  "Presentation",
    "meeting_minutes": "Minutes",
    "specifications": "PDF",
    "emails":         "Email",
}

# Concept → is_issued / folder_stage flags for file_locator
_CONCEPT_FLAGS: dict[str, dict[str, Any]] = {
    "ifc_set":     {"folder_stage": "IFC"},
    "transmittals": {"is_issued": 1},
}


# ---------------------------------------------------------------------------
# Keyword sets (fallback when no concept matches)
# ---------------------------------------------------------------------------

_STRUCTURED_KEYWORDS = frozenset([
    "floor area", "site area", "plot ratio", "storeys", "floors",
    "number of units", "how many units", "how many keys", "how many beds",
    "who is the client", "who is the architect", "who is the pm",
    "project manager", "job captain", "what typology", "building type",
    "what stage", "project status", "contract value",
    "where is the project", "where is project",
    "project name", "project code", "what is the project",
    "other names", "goes by", "also known as", "project alias",
    "client name", "developer", "building height", "height of",
    "area of", "size of", "gross floor",
])

_FILE_LOCATOR_KEYWORDS = frozenset([
    "where is", "where are", "where can i find", "find", "locate",
    "show me", "list all", "get all", "pull up", "get me",
    "latest", "most recent", "current version", "current revision",
    "latest version",
])

_FILE_TYPE_KEYWORDS = frozenset([
    "bim", "revit", ".rvt", "cad", ".dwg", "autocad",
    "photoshop", ".psd", "illustrator", ".ai",
    "indesign", ".indd", "powerpoint", ".pptx", "presentation", "deck",
    "spreadsheet", ".xlsx", "excel", ".xls",
    "render", "viz", "visualisation", "visualization", "image", "jpg", "png",
    "minutes", "transmittal", "rfi", "specification", "spec", "brief",
    "report", "drawing", "drawings", "programme", "schedule", "gantt",
])

_QUALIFIER_KEYWORDS = frozenset([
    "issued to client", "outgoing", "superseded", "archived", "old versions",
    "for construction", "for approval", "for tender",
])

_SEMANTIC_KEYWORDS = frozenset([
    "design concept", "concept", "what is the idea", "design intent",
    "facade", "facade strategy", "materiality", "sustainability", "climate",
    "waiver", "deviation", "exemption", "approval", "authority", "compliance",
    "meeting", "minutes content", "decision", "action item", "unresolved",
    "specification details", "technical", "build-up", "glazing",
    "waterproofing", "what was discussed", "history of", "change",
    "revision history", "why", "how was", "what does", "explain",
    "describe", "summary of",
    "apply for", "did we", "what are the", "tell me about the",
])

_CROSS_PROJECT_KEYWORDS = frozenset([
    "across", "all projects", "compare", "which projects", "show all",
    "list projects", "how many projects", "other projects", "portfolio",
    "all our projects", "every project", "all our",
])

# Folder stage keyword → folder_stage value
_STAGE_KEYWORDS: dict[str, str] = {
    "ifc": "IFC",
    "issued for construction": "IFC",
    "for construction": "IFC",
    "tender": "tender",
    "pre-tender": "tender",
    "submission": "submission",
    "authority submission": "submission",
    "superseded": "superseded",
    "archived": "superseded",
    "old versions": "superseded",
    "design": "design",
    "schematic": "design",
    "concept design": "design",
    "renders": "renders",
    "render": "renders",
    "visualisation": "renders",
    "visualization": "renders",
    "presentation": "presentations",
}

# File type keywords → content_type value
_CONTENT_TYPE_MAP: dict[str, str] = {
    "bim": "BIM", "revit": "BIM", ".rvt": "BIM",
    "cad": "CAD", ".dwg": "CAD", "autocad": "CAD",
    "photoshop": "Photoshop", ".psd": "Photoshop",
    "illustrator": "Illustrator", ".ai": "Illustrator",
    "indesign": "InDesign", ".indd": "InDesign",
    "powerpoint": "Presentation", ".pptx": "Presentation",
    "presentation": "Presentation", "deck": "Presentation",
    "spreadsheet": "Spreadsheet", ".xlsx": "Spreadsheet", "excel": "Spreadsheet",
    ".xls": "Spreadsheet",
    "render": "Render Scene", "viz": "Render Scene",
    "visualisation": "Render Scene", "visualization": "Render Scene",
    "image": "Image", "jpg": "Image", "png": "Image",
    "minutes": "Minutes", "transmittal": "Transmittal", "rfi": "RFI",
    "word": "Word", ".docx": "Word", ".doc": "Word",
    "pdf": "PDF",
    "video": "Video",
    "email": "Email", "msg": "Email", "eml": "Email",
}


# ---------------------------------------------------------------------------
# Synonym loader (cached, auto-reloads on file change)
# ---------------------------------------------------------------------------

_synonyms_data: dict[str, Any] | None = None
_synonyms_mtime: float = 0.0
_synonyms_path: str = ""


def _get_synonyms_path() -> Path:
    """Find query_synonyms.yml in work_dir."""
    try:
        from config import cfg
        return cfg.work_dir / "query_synonyms.yml"
    except Exception:
        return Path(__file__).parent.parent / "tiga_work" / "query_synonyms.yml"


def load_synonyms(yaml_path: str | Path | None = None) -> dict[str, Any]:
    """
    Load and cache query_synonyms.yml.
    Auto-reloads if the file modification time changes.
    Returns dict of {concept_name: {canonical, synonyms}} or {} on failure.
    """
    global _synonyms_data, _synonyms_mtime, _synonyms_path

    path = Path(yaml_path) if yaml_path else _get_synonyms_path()
    path_str = str(path)

    try:
        mtime = path.stat().st_mtime if path.exists() else 0.0
    except OSError:
        mtime = 0.0

    if _synonyms_data is not None and path_str == _synonyms_path and mtime == _synonyms_mtime:
        return _synonyms_data

    if not path.exists():
        logger.debug("query_synonyms.yml not found at %s — synonym expansion disabled", path)
        _synonyms_data = {}
        _synonyms_mtime = 0.0
        _synonyms_path = path_str
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _synonyms_data = data
        _synonyms_mtime = mtime
        _synonyms_path = path_str
        logger.debug("Loaded %d synonym concepts from %s", len(data), path)
        return data
    except Exception as e:
        logger.warning("Failed to load query_synonyms.yml: %s", e)
        _synonyms_data = {}
        return {}


# ---------------------------------------------------------------------------
# QueryRouter
# ---------------------------------------------------------------------------

class QueryRouter:
    """
    Classifies a natural-language query into a routing mode.
    Concept-based matching via query_synonyms.yml, with raw keyword fallback.
    No LLM, no external calls — latency < 10 ms.
    """

    def __init__(self, known_project_codes: list[str] | None = None):
        self._known_codes = set(known_project_codes or [])

    def load_project_codes(self, conn) -> None:
        """Reload known project codes from DB for better project detection."""
        try:
            rows = conn.execute(
                "SELECT DISTINCT project_code FROM project_cards "
                "WHERE project_code IS NOT NULL"
            ).fetchall()
            self._known_codes = {r["project_code"] for r in rows}
            # Also pull from files table project_id column
            rows2 = conn.execute(
                "SELECT DISTINCT project_id FROM files "
                "WHERE project_id IS NOT NULL AND project_id != 'Unknown'"
            ).fetchall()
            self._known_codes.update(r["project_id"] for r in rows2)
        except Exception as e:
            logger.warning("Failed to load project codes: %s", e)

    # ── Synonym expansion ─────────────────────────────────────────────────────

    def expand_query(self, query: str) -> ExpandedQuery:
        """
        Expand query using synonym map.

        Returns ExpandedQuery with:
          - concept_tags: canonical concept names matched
          - expanded_terms: all synonyms for matched concepts (FTS/vector boost)
        """
        synonyms = load_synonyms()
        normalised = re.sub(r"[^\w\s]", " ", query.lower()).strip()
        # Collapse multiple spaces
        normalised = re.sub(r"\s+", " ", normalised)

        concept_tags: list[str] = []
        expanded_terms: list[str] = []

        for concept_name, concept_data in synonyms.items():
            if not isinstance(concept_data, dict):
                continue
            canonical = str(concept_data.get("canonical", "")).lower()
            syn_list: list[str] = [
                str(s).lower() for s in concept_data.get("synonyms", [])
                if isinstance(s, str)
            ]
            all_terms = [canonical] + syn_list

            # Check if any synonym (or canonical) appears in the normalised query
            matched = False
            for term in all_terms:
                term_norm = re.sub(r"[^\w\s]", " ", term).strip()
                if term_norm and term_norm in normalised:
                    matched = True
                    break

            if matched:
                concept_tags.append(concept_name)
                expanded_terms.extend(all_terms)

        # Deduplicate expanded_terms, preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for t in expanded_terms:
            if t not in seen:
                seen.add(t)
                deduped.append(t)

        return ExpandedQuery(
            original=query,
            normalised=normalised,
            concept_tags=concept_tags,
            expanded_terms=deduped,
        )

    # ── Main classify ─────────────────────────────────────────────────────────

    def classify(
        self,
        query: str,
        project_code: str | None = None,
    ) -> RouteResult:
        """
        Classify a query and return a RouteResult.

        Args:
            query:        User's natural-language query.
            project_code: Optional project scope already selected by UI.
        """
        q = query.lower().strip()

        # Step 1: Expand query with synonyms
        expanded = self.expand_query(query)
        tags = set(expanded.concept_tags)

        # Extract any project code mentioned in query
        detected_code = self._detect_project_code(q) or project_code

        # ── CROSS_PROJECT check (before others) ──────────────────────────────
        if not detected_code and any(kw in q for kw in _CROSS_PROJECT_KEYWORDS):
            filters = self._parse_cross_project_filters(q)
            return RouteResult(
                mode="cross_project",
                project_code=None,
                filters=filters,
                confidence=0.85,
                reason="Cross-project keywords detected",
                expanded_query=expanded,
            )

        # ── Score each mode (concept-based + keyword fallback) ────────────────
        structured_score = self._score_structured(q, tags)
        file_locator_score = self._score_file_locator(q, tags)
        semantic_score = self._score_semantic(q, tags)

        scores = {
            "structured":   structured_score,
            "file_locator": file_locator_score,
            "semantic":     semantic_score,
        }

        best_mode = max(scores, key=lambda k: scores[k])
        best_score = scores[best_mode]

        # Default fallback to semantic if all scores are low
        if best_score == 0:
            best_mode = "semantic"
            best_score = 0.5

        # Project code present → prefer structured over low-signal semantic
        # (e.g. "What's project 261 about?" has no concept tags but clear project intent)
        if detected_code and best_mode == "semantic" and best_score <= 0.25:
            best_mode = "structured"
            best_score = 0.35

        # Secondary mode for ambiguous cases
        second_mode: str | None = None
        sorted_modes = sorted(scores, key=lambda k: -scores[k])
        if len(sorted_modes) >= 2:
            second = sorted_modes[1]
            if scores[second] > 0 and scores[second] >= best_score * 0.7:
                second_mode = second

        confidence = min(best_score, 1.0)

        # Parse mode-specific filters
        filters: dict[str, Any] = {}
        if detected_code:
            filters["project_code"] = detected_code

        # Apply concept-based filters first (more reliable than string matching)
        if best_mode == "file_locator":
            filters.update(self._parse_file_locator_filters_from_concepts(q, tags))
        elif best_mode == "structured":
            filters.update(self._parse_structured_filters(q))

        # Log ambiguous queries
        if confidence < 0.7:
            self._log_ambiguous(query, best_mode, confidence, scores)

        return RouteResult(
            mode=best_mode,
            project_code=detected_code,
            filters=filters,
            confidence=confidence,
            reason=self._explain(best_mode, scores, tags),
            secondary_mode=second_mode if confidence < 0.7 else None,
            expanded_query=expanded,
        )

    # ── Scoring methods ──────────────────────────────────────────────────────

    def _score_structured(self, q: str, tags: set[str]) -> float:
        score = 0.0
        # Concept-based scoring (primary signal)
        for tag in tags:
            if tag in _STRUCTURED_CONCEPTS:
                score += 0.5  # strong signal
        # Raw keyword fallback
        for kw in _STRUCTURED_KEYWORDS:
            if kw in q:
                weight = 0.5 if len(kw) > 8 else 0.35
                score += weight
        return min(score, 1.0)

    def _score_file_locator(self, q: str, tags: set[str]) -> float:
        score = 0.0
        # Concept-based scoring
        for tag in tags:
            if tag in _FILE_LOCATOR_CONCEPTS:
                score += 0.55  # strong signal — know exactly what file type
        # Trigger verbs (still useful even when concept is known)
        for kw in _FILE_LOCATOR_KEYWORDS:
            if kw in q:
                score += 0.25
        # File type mentions (fallback)
        if not tags & _FILE_LOCATOR_CONCEPTS:
            for kw in _FILE_TYPE_KEYWORDS:
                if kw in q:
                    score += 0.25
        # Qualifier phrases
        for kw in _QUALIFIER_KEYWORDS:
            if kw in q:
                score += 0.25
        # Date scoping
        if re.search(r'\b(in|from|around)\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4})', q):
            score += 0.15
        return min(score, 1.0)

    def _score_semantic(self, q: str, tags: set[str]) -> float:
        # Base score is low — semantic is the fallback
        score = 0.2
        # Concept-based scoring
        for tag in tags:
            if tag in _SEMANTIC_CONCEPTS:
                score += 0.4
        # Raw keyword fallback
        for kw in _SEMANTIC_KEYWORDS:
            if kw in q:
                score += 0.25
        return min(score, 1.0)

    # ── Filter parsers ───────────────────────────────────────────────────────

    def _parse_file_locator_filters_from_concepts(
        self, q: str, tags: set[str]
    ) -> dict[str, Any]:
        """Build file_locator filters from concept_tags first, then raw string fallback."""
        f: dict[str, Any] = {}

        # Apply concept-based content_type (first matching concept wins)
        for tag in ("bim_model", "cad_drawings", "renders", "presentations",
                    "meeting_minutes", "specifications", "emails", "ifc_set", "transmittals"):
            if tag in tags:
                ct = _CONCEPT_CONTENT_TYPE.get(tag)
                if ct:
                    f["content_type"] = ct
                extra = _CONCEPT_FLAGS.get(tag, {})
                f.update(extra)
                break

        # If no concept match, fall back to raw string matching
        if "content_type" not in f:
            for kw, ct in _CONTENT_TYPE_MAP.items():
                if kw in q:
                    f["content_type"] = ct
                    break

        # Folder stage from raw string (if not already set by concept)
        if "folder_stage" not in f:
            for kw, stage in _STAGE_KEYWORDS.items():
                if kw in q:
                    f["folder_stage"] = stage
                    break

        # Flags
        if "transmittals" in tags or any(kw in q for kw in ["issued to client", "outgoing", "sent to client"]):
            f["is_issued"] = 1
        if any(kw in q for kw in ["superseded", "archived", "old versions", "old"]):
            f["is_superseded"] = 1
        if any(kw in q for kw in ["latest", "current version", "most recent", "current revision"]):
            f["is_latest"] = 1

        # Date range
        date_m = re.search(
            r'(?:in|from)\s+(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|'
            r'jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|'
            r'nov(?:ember)?|dec(?:ember)?)\s*(\d{4})?',
            q,
        )
        if date_m:
            f["date_hint"] = date_m.group(0)
        year_m = re.search(r'\b(20\d{2})\b', q)
        if year_m:
            year = year_m.group(1)
            f["date_from"] = f"{year}-01-01"
            f["date_to"]   = f"{year}-12-31"

        return f

    def _parse_structured_filters(self, q: str) -> dict[str, Any]:
        f: dict[str, Any] = {}
        # GFA range
        m = re.search(r'(?:more than|above|over)\s+([\d,]+)\s*(?:sqm|m2|m²)', q)
        if m:
            f["gfa_min"] = float(m.group(1).replace(",", ""))
        m = re.search(r'(?:less than|below|under)\s+([\d,]+)\s*(?:sqm|m2|m²)', q)
        if m:
            f["gfa_max"] = float(m.group(1).replace(",", ""))
        # Typology
        for typ in ["residential", "hospitality", "hotel", "commercial", "civic",
                    "mixed-use", "education", "healthcare"]:
            if typ in q:
                f["typology"] = typ
                break
        # Stage
        for stage in ["design", "tender", "construction", "completed"]:
            if stage in q:
                f["stage"] = stage.title()
                break
        return f

    def _parse_cross_project_filters(self, q: str) -> dict[str, Any]:
        f: dict[str, Any] = {}
        for typ in ["residential", "hospitality", "hotel", "commercial",
                    "civic", "mixed-use", "education"]:
            if typ in q:
                f["typology"] = typ
                break
        return f

    # ── Project code detection ───────────────────────────────────────────────

    def _detect_project_code(self, q: str) -> str | None:
        """Look for a known project code or 3-4 digit number in the query."""
        # Check known codes first
        for code in sorted(self._known_codes, key=lambda c: -len(c)):
            if re.search(rf'\b{re.escape(code)}\b', q):
                return code
        # Fallback: standalone 3-4 digit number (non-zero first digit to avoid "000" from "50,000")
        m = re.search(r'\b([1-9]\d{2,3})\b', q)
        if m:
            return m.group(1)
        return None

    # ── Logging ──────────────────────────────────────────────────────────────

    def _log_ambiguous(
        self,
        query: str,
        mode: str,
        confidence: float,
        scores: dict[str, float],
    ) -> None:
        """Append ambiguous query to router_ambiguous.log."""
        try:
            from config import cfg as _cfg
            log_dir = _cfg.work_dir / "logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            log_file = log_dir / "router_ambiguous.log"
            from datetime import datetime as _dt
            line = (
                f"{_dt.now().isoformat()} | {query!r} | "
                f"mode={mode} conf={confidence:.2f} "
                f"scores={scores}\n"
            )
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

    def _explain(self, mode: str, scores: dict[str, float], tags: set[str]) -> str:
        tags_str = ",".join(sorted(tags)) if tags else "none"
        return (
            f"mode={mode} concepts=[{tags_str}] "
            f"(structured={scores['structured']:.2f}, "
            f"file_locator={scores['file_locator']:.2f}, "
            f"semantic={scores['semantic']:.2f})"
        )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_router: QueryRouter | None = None


def get_router() -> QueryRouter:
    """Return (or create) the module-level router singleton."""
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router
