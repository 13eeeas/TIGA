"""
core/knowledge.py - deterministic helpers for Knowledge Corpus Mode.

This module keeps corpus-specific classification cheap and inspectable.  It
uses path, filename, extension, and small text previews before any LLM work.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from config import Config


SLIDE_EXTS = {".ppt", ".pptx", ".key"}
ARTICLE_EXTS = {".pdf", ".doc", ".docx", ".txt", ".md"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".gif", ".tif", ".tiff", ".bmp"}
SPREADSHEET_EXTS = {".xls", ".xlsx", ".csv"}
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".wmv"}


TOPIC_KEYWORDS: dict[str, list[str]] = {
    "ai_architecture": ["ai", "artificial intelligence", "machine learning", "llm", "chatgpt", "generative"],
    "design_process": ["design process", "concept", "iteration", "workflow", "collaboration", "critique"],
    "fire_safety": ["fire safety", "scdf", "fire code", "egress", "compliance", "regulation"],
    "sustainability": ["sustainability", "climate", "green", "carbon", "environment", "passive"],
    "presentation": ["presentation", "storytelling", "slide", "visual communication", "public speaking"],
    "practice_management": ["contract", "fee", "client", "project management", "procurement"],
    "exhibition": ["exhibition", "gallery", "curation", "installation", "showcase"],
    "training": ["training", "workshop", "seminar", "course", "learning"],
}


EVENT_TYPE_KEYWORDS: dict[str, list[str]] = {
    "seminar": ["seminar", "talk", "lecture", "sharing"],
    "training": ["training", "workshop", "course", "cpd"],
    "exhibition": ["exhibition", "expo", "showcase"],
    "presentation": ["presentation", "deck", "slides"],
    "transcript": ["transcript", "notes", "minutes"],
}


@dataclass
class KnowledgeMetadata:
    title: str
    speaker: str
    event_name: str
    event_type: str
    date: str
    source_file_path: str
    file_type: str
    topic: str
    subtopic: str
    organization: str
    people_mentioned: list[str]
    keywords: list[str]
    summary: str
    confidence: float
    extraction_method: str
    language: str
    duplicate_group: str
    parent_document_id: str
    relationship: str

    def to_db(self) -> dict[str, Any]:
        data = asdict(self)
        data["people_mentioned"] = json.dumps(self.people_mentioned, ensure_ascii=False)
        data["keywords"] = json.dumps(self.keywords, ensure_ascii=False)
        return data


def classify_file_type(path: Path, text_preview: str = "") -> str:
    ext = path.suffix.lower()
    name = path.name.lower()
    text = text_preview.lower()
    if ext in SLIDE_EXTS:
        return "slide_deck"
    if "transcript" in name or "transcript" in text[:500]:
        return "transcript"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "mixed_media"
    if ext in SPREADSHEET_EXTS:
        return "mixed_media"
    if ext in ARTICLE_EXTS:
        if any(w in name for w in ("brochure", "flyer", "leaflet")):
            return "brochure"
        return "article"
    return "mixed_media"


def infer_event_type(path: Path, text_preview: str = "") -> str:
    haystack = f"{path.as_posix()} {text_preview[:1200]}".lower()
    for event_type, terms in EVENT_TYPE_KEYWORDS.items():
        if any(term in haystack for term in terms):
            return event_type
    return "unknown"


def infer_topic(path: Path, text_preview: str = "") -> tuple[str, list[str]]:
    haystack = f"{path.as_posix()} {text_preview[:3000]}".lower()
    matched: list[tuple[str, int]] = []
    keywords: list[str] = []
    for topic, terms in TOPIC_KEYWORDS.items():
        score = 0
        for term in terms:
            if term in haystack:
                score += 1
                keywords.append(term)
        if score:
            matched.append((topic, score))
    if not matched:
        return "unknown", sorted(set(keywords))
    matched.sort(key=lambda x: (-x[1], x[0]))
    return matched[0][0], sorted(set(keywords))[:20]


def infer_date(path: Path) -> str:
    s = path.as_posix()
    patterns = [
        r"\b(20\d{2})[-_. ](0[1-9]|1[0-2])[-_. ]([0-3]\d)\b",
        r"\b([0-3]\d)[-_. ](0[1-9]|1[0-2])[-_. ](20\d{2})\b",
        r"\b(20\d{2})(0[1-9]|1[0-2])([0-3]\d)\b",
        r"\b(20\d{2})\b",
    ]
    for pat in patterns:
        m = re.search(pat, s)
        if not m:
            continue
        groups = m.groups()
        if len(groups) == 1:
            return groups[0]
        if groups[0].startswith("20"):
            return f"{groups[0]}-{groups[1]}-{groups[2]}"
        return f"{groups[2]}-{groups[1]}-{groups[0]}"
    return ""


def title_from_path(path: Path) -> str:
    stem = re.sub(r"[_\-]+", " ", path.stem)
    stem = re.sub(r"\b20\d{2}(?:[ ._-]?\d{2})?(?:[ ._-]?\d{2})?\b", "", stem)
    return re.sub(r"\s+", " ", stem).strip() or path.stem


def infer_event_name(path: Path, cfg_obj: Config) -> str:
    try:
        rel = next(path.relative_to(root) for root in cfg_obj.index_roots if path.is_relative_to(root))
        parts = rel.parts
    except Exception:
        parts = path.parts[-4:]
    folders = [p for p in parts[:-1] if p and not re.fullmatch(r"\d+", p)]
    if not folders:
        return ""
    return folders[-1]


def infer_speaker(path: Path, text_preview: str = "") -> str:
    name = path.stem
    patterns = [
        r"(?:speaker|presenter|by)[: _-]+([A-Z][A-Za-z .'-]{2,60})",
        r"\bby\s+([A-Z][A-Za-z .'-]{2,60})\b",
    ]
    for source in (text_preview[:1200], name):
        for pat in patterns:
            m = re.search(pat, source, flags=re.IGNORECASE)
            if m:
                return re.sub(r"\s+", " ", m.group(1)).strip(" ._-")
    return ""


def infer_people(text_preview: str) -> list[str]:
    # Conservative proper-name heuristic; used only as weak metadata.
    candidates = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", text_preview[:4000])
    blocked = {"Power Point", "Microsoft Office", "New York", "United States"}
    out: list[str] = []
    seen: set[str] = set()
    for c in candidates:
        if c in blocked or c.lower() in seen:
            continue
        seen.add(c.lower())
        out.append(c)
        if len(out) >= 20:
            break
    return out


def infer_language(text_preview: str) -> str:
    if not text_preview.strip():
        return "unknown"
    ascii_letters = sum(1 for ch in text_preview[:2000] if "a" <= ch.lower() <= "z")
    if ascii_letters / max(len(text_preview[:2000]), 1) > 0.35:
        return "en"
    return "unknown"


def duplicate_group_for(text: str, path: Path) -> str:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if len(normalized) >= 200:
        return "text:" + hashlib.sha256(normalized[:5000].encode()).hexdigest()[:16]
    key = f"{path.name.lower()}:{path.stat().st_size if path.exists() else 0}"
    return "file:" + hashlib.sha256(key.encode()).hexdigest()[:16]


def parent_document_id(path: Path) -> str:
    # Group derived transcript/PDF/image exports from the same deck/folder.
    base = re.sub(r"(?i)\b(transcript|notes|ocr|export|copy)\b", "", path.stem)
    base = re.sub(r"[_\-\s]+", " ", base).strip().lower()
    parent_key = f"{path.parent.as_posix()}::{base}"
    return hashlib.sha256(parent_key.encode()).hexdigest()[:20]


def relationship_for(path: Path, file_type: str) -> str:
    if file_type == "slide_deck":
        return "parent"
    if file_type == "transcript":
        return "transcript"
    if file_type == "image":
        return "derived_media"
    return "source"


def build_contextual_header(meta: KnowledgeMetadata, ref_value: str) -> str:
    fields = [
        ("Title", meta.title),
        ("Event", meta.event_name),
        ("Speaker", meta.speaker),
        ("Topic", meta.topic),
        ("Type", meta.file_type),
        ("Date", meta.date),
        ("Ref", ref_value),
    ]
    return "\n".join(f"{k}: {v}" for k, v in fields if v)


def infer_metadata(
    path: Path,
    cfg_obj: Config,
    text_preview: str = "",
    extraction_method: str = "path",
) -> KnowledgeMetadata:
    file_type = classify_file_type(path, text_preview)
    topic, keywords = infer_topic(path, text_preview)
    event_type = infer_event_type(path, text_preview)
    confidence = 0.35
    if topic != "unknown":
        confidence += 0.2
    if event_type != "unknown":
        confidence += 0.15
    if text_preview.strip():
        confidence += 0.15
    return KnowledgeMetadata(
        title=title_from_path(path),
        speaker=infer_speaker(path, text_preview),
        event_name=infer_event_name(path, cfg_obj),
        event_type=event_type,
        date=infer_date(path),
        source_file_path=path.as_posix(),
        file_type=file_type,
        topic=topic,
        subtopic="",
        organization="",
        people_mentioned=infer_people(text_preview),
        keywords=keywords,
        summary=" ".join(text_preview.split()[:80]),
        confidence=round(min(confidence, 0.9), 2),
        extraction_method=extraction_method,
        language=infer_language(text_preview),
        duplicate_group=duplicate_group_for(text_preview, path),
        parent_document_id=parent_document_id(path),
        relationship=relationship_for(path, file_type),
    )
