"""
core/evidence.py — First-class evidence / citation objects.

Citations are not free text appended by the LLM. Every substantive claim in
Atlas/Hunt should eventually point at an EvidenceCitation built from indexed
chunks + file metadata.

Epistemic labels (answer contract):
  documented  — stated in a cited source
  inferred    — conclusion drawn from cited sources (must say so)
  conflicting — sources disagree; present both
  missing     — archive does not contain usable evidence
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

VersionStatus = Literal["current", "superseded", "unknown"]
EpistemicLabel = Literal["documented", "inferred", "conflicting", "missing"]

EPISTEMIC_LABELS: tuple[EpistemicLabel, ...] = (
    "documented",
    "inferred",
    "conflicting",
    "missing",
)


def version_status_from_flags(
    is_latest: int | bool | None = None,
    is_superseded: int | bool | None = None,
) -> VersionStatus:
    if is_superseded:
        return "superseded"
    if is_latest:
        return "current"
    return "unknown"


@dataclass
class EvidenceCitation:
    """Traceable evidence unit returned by Hunt / consumed by Einstein."""

    citation: str
    chunk_id: str = ""
    file_id: str = ""
    source_path: str = ""
    rel_path: str = ""
    file_name: str = ""
    ref_value: str = ""  # page / slide / section / meta
    project_id: str = ""
    project_code: str | None = None
    discipline: str | None = None
    doc_type: str | None = None
    content_type: str | None = None
    revision: int | None = None
    file_date: str | None = None
    version_status: VersionStatus = "unknown"
    quote: str = ""
    final_score: float = 0.0
    typology: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_search_result(cls, r: dict[str, Any], *, quote: str | None = None) -> "EvidenceCitation":
        rel = str(r.get("rel_path") or "")
        file_name = str(r.get("file_name") or (Path(rel).name if rel else ""))
        snippet = str(r.get("snippet") or "")
        body = quote if quote is not None else (
            str(r.get("chunk_text") or r.get("evidence_text") or snippet)
        )
        rev = r.get("revision")
        revision: int | None
        try:
            revision = int(rev) if rev is not None and str(rev).strip() != "" else None
        except (TypeError, ValueError):
            revision = None

        return cls(
            citation=str(r.get("citation") or ""),
            chunk_id=str(r.get("chunk_id") or ""),
            file_id=str(r.get("file_id") or ""),
            source_path=str(r.get("file_path") or ""),
            rel_path=rel,
            file_name=file_name,
            ref_value=str(r.get("ref_value") or ""),
            project_id=str(r.get("project_id") or "Unknown"),
            project_code=(str(r["project_code"]) if r.get("project_code") else None),
            discipline=(str(r["discipline"]) if r.get("discipline") else None),
            doc_type=(str(r["doc_type"]) if r.get("doc_type") else None),
            content_type=(str(r["content_type"]) if r.get("content_type") else None),
            revision=revision,
            file_date=(str(r["file_date"]) if r.get("file_date") else None),
            version_status=version_status_from_flags(
                r.get("is_latest"),
                r.get("is_superseded"),
            ),
            quote=body[:2000],
            final_score=float(r.get("final_score") or 0.0),
            typology=str(r.get("typology") or ""),
        )


@dataclass
class EvidencePack:
    """Ordered evidence set for an answer, with light version conflict hints."""

    items: list[EvidenceCitation] = field(default_factory=list)
    version_warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "items": [e.to_dict() for e in self.items],
            "version_warnings": list(self.version_warnings),
        }

    @classmethod
    def from_search_results(
        cls,
        results: list[dict[str, Any]],
        *,
        quote_by_chunk: dict[str, str] | None = None,
    ) -> "EvidencePack":
        items: list[EvidenceCitation] = []
        for r in results:
            cid = str(r.get("chunk_id") or "")
            quote = (quote_by_chunk or {}).get(cid)
            items.append(EvidenceCitation.from_search_result(r, quote=quote))
        return cls(items=items, version_warnings=_version_warnings(items))


def _version_warnings(items: list[EvidenceCitation]) -> list[str]:
    """Flag packs that mix current and superseded hits for the same doc family."""
    warnings: list[str] = []
    groups: dict[str, set[VersionStatus]] = {}
    for e in items:
        key = "|".join([
            e.project_id or "",
            e.discipline or "",
            e.doc_type or Path(e.file_name).stem.lower()[:40],
        ])
        groups.setdefault(key, set()).add(e.version_status)

    for key, statuses in groups.items():
        if "current" in statuses and "superseded" in statuses:
            warnings.append(
                "Mixed current and superseded evidence for related documents — "
                "do not average revisions; prefer current unless the question is historical."
            )
            break
    return warnings


def format_evidence_block(ev: EvidenceCitation, index: int) -> str:
    """Human + LLM readable evidence header with architecture metadata."""
    meta_bits: list[str] = []
    if ev.project_id:
        meta_bits.append(f"project={ev.project_id}")
    if ev.discipline:
        meta_bits.append(f"discipline={ev.discipline}")
    if ev.doc_type:
        meta_bits.append(f"doc_type={ev.doc_type}")
    if ev.revision is not None:
        meta_bits.append(f"rev={ev.revision}")
    if ev.file_date:
        meta_bits.append(f"date={ev.file_date}")
    meta_bits.append(f"version={ev.version_status}")
    meta = "; ".join(meta_bits)
    body = ev.quote or ""
    return f"[{index}] {ev.citation}\n({meta})\n{body}"
