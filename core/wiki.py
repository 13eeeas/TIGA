"""
core/wiki.py - first-pass durable wiki generation for Knowledge Corpus Mode.

The wiki is deliberately deterministic: it groups indexed chunks by knowledge
metadata and emits Markdown pages with citations/backlinks.  LLM synthesis can
be added later, but the first artifact must be auditable.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import cfg as _module_cfg, Config
from core.db import get_connection


def _slug(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "unknown"


def _wiki_dir(cfg_obj: Config) -> Path:
    return cfg_obj.work_dir / "wiki"


def _rel_path(file_path: str, cfg_obj: Config) -> str:
    p = Path(file_path)
    for root in cfg_obj.index_roots:
        try:
            return p.relative_to(root).as_posix()
        except ValueError:
            continue
    return p.name


def _citation(row: sqlite3.Row, cfg_obj: Config) -> str:
    return f"{_rel_path(row['file_path'], cfg_obj)}#{row['ref_value']}"


def _load_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """SELECT
              f.file_id, f.file_path, f.file_name,
              COALESCE(f.knowledge_title, f.file_name, '') AS title,
              COALESCE(f.speaker, '') AS speaker,
              COALESCE(f.event_name, '') AS event_name,
              COALESCE(f.event_type, '') AS event_type,
              COALESCE(f.event_date, '') AS event_date,
              COALESCE(f.file_type, '') AS file_type,
              COALESCE(f.topic, 'unknown') AS topic,
              COALESCE(f.subtopic, '') AS subtopic,
              COALESCE(f.organization, '') AS organization,
              COALESCE(f.keywords, '[]') AS keywords,
              COALESCE(f.summary, '') AS summary,
              COALESCE(f.metadata_confidence, 0) AS metadata_confidence,
              COALESCE(f.parent_document_id, '') AS parent_document_id,
              COALESCE(f.document_relationship, '') AS document_relationship,
              c.chunk_id, c.ref_value, c.text
           FROM files f
           JOIN chunks c ON c.file_id = f.file_id
           WHERE f.status IN ('EXTRACTED', 'EMBEDDED', 'INDEXED')
           ORDER BY f.topic, f.event_name, f.speaker, f.file_name, c.ref_value"""
    ).fetchall()


def _frontmatter(title: str, page_type: str, extra: dict[str, Any] | None = None) -> str:
    payload = {
        "title": title,
        "type": page_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if extra:
        payload.update(extra)
    lines = ["---"]
    for k, v in payload.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{k}: {json.dumps(v) if isinstance(v, str) else v}")
    lines.append("---\n")
    return "\n".join(lines)


def _sample_sources(rows: list[sqlite3.Row], cfg_obj: Config, limit: int = 8) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        c = _citation(row, cfg_obj)
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
        if len(out) >= limit:
            break
    return out


def _write_page(path: Path, title: str, page_type: str, body: str, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_frontmatter(title, page_type, extra) + body.strip() + "\n", encoding="utf-8")


def _keywords(rows: list[sqlite3.Row]) -> list[str]:
    counter: Counter[str] = Counter()
    for row in rows:
        try:
            vals = json.loads(row["keywords"] or "[]")
            if isinstance(vals, list):
                counter.update(str(v) for v in vals if v)
        except Exception:
            pass
    return [k for k, _ in counter.most_common(12)]


def _excerpt(row: sqlite3.Row) -> str:
    text = re.sub(r"\s+", " ", row["text"] or "").strip()
    return text[:360]


def _topic_page(topic: str, rows: list[sqlite3.Row], cfg_obj: Config) -> str:
    events = sorted({r["event_name"] for r in rows if r["event_name"]})
    speakers = sorted({r["speaker"] for r in rows if r["speaker"]})
    kws = _keywords(rows)
    sources = _sample_sources(rows, cfg_obj)
    body = [
        f"# {topic.replace('_', ' ').title()}",
        "",
        f"Evidence base: {len(rows)} chunk(s) from {len({r['file_id'] for r in rows})} document(s).",
    ]
    if len(sources) < getattr(cfg_obj, "wiki_min_sources", 2):
        body.append("Uncertainty: evidence is thin; treat this page as provisional.")
    if kws:
        body.extend(["", "## Repeated Keywords", "", ", ".join(kws)])
    if events:
        body.extend(["", "## Events", "", "\n".join(f"- [[events/{_slug(e)}|{e}]]" for e in events[:20])])
    if speakers:
        body.extend(["", "## Speakers", "", "\n".join(f"- [[speakers/{_slug(s)}|{s}]]" for s in speakers[:20])])
    body.extend(["", "## Evidence", ""])
    for row in rows[:8]:
        body.append(f"- `{_citation(row, cfg_obj)}` - {_excerpt(row)}")
    body.extend(["", "## Source Citations", "", "\n".join(f"- `{s}`" for s in sources)])
    return "\n".join(body)


def _entity_page(kind: str, name: str, rows: list[sqlite3.Row], cfg_obj: Config) -> str:
    topics = sorted({r["topic"] for r in rows if r["topic"] and r["topic"] != "unknown"})
    sources = _sample_sources(rows, cfg_obj)
    body = [
        f"# {name}",
        "",
        f"Page type: {kind}.",
        f"Evidence base: {len(rows)} chunk(s) from {len({r['file_id'] for r in rows})} document(s).",
    ]
    if topics:
        body.extend(["", "## Related Topics", "", "\n".join(f"- [[topics/{_slug(t)}|{t}]]" for t in topics)])
    body.extend(["", "## Evidence", ""])
    for row in rows[:10]:
        label = row["title"] or row["file_name"]
        body.append(f"- `{_citation(row, cfg_obj)}` - {label}: {_excerpt(row)}")
    body.extend(["", "## Source Citations", "", "\n".join(f"- `{s}`" for s in sources)])
    return "\n".join(body)


def _theme_page(rows: list[sqlite3.Row], cfg_obj: Config) -> str:
    topic_counts = Counter(r["topic"] or "unknown" for r in rows)
    keyword_counts = Counter(_keywords(rows))
    sources = _sample_sources(rows, cfg_obj, limit=12)
    body = [
        "# What This Corpus Teaches",
        "",
        "This page is generated from retrieval metadata and indexed excerpts. It is a map of repeated evidence, not an LLM essay.",
        "",
        "## Recurring Topics",
        "",
    ]
    for topic, count in topic_counts.most_common(15):
        body.append(f"- [[topics/{_slug(topic)}|{topic}]] - {count} chunk(s)")
    if keyword_counts:
        body.extend(["", "## Repeated Terms", "", ", ".join(k for k, _ in keyword_counts.most_common(20))])
    body.extend(["", "## Representative Evidence", ""])
    for row in rows[:12]:
        body.append(f"- `{_citation(row, cfg_obj)}` - {_excerpt(row)}")
    body.extend(["", "## Source Citations", "", "\n".join(f"- `{s}`" for s in sources)])
    return "\n".join(body)


def build_wiki(
    cfg_obj: Config | None = None,
    conn: sqlite3.Connection | None = None,
) -> dict[str, int]:
    _cfg = cfg_obj or _module_cfg
    own_conn = conn is None
    _conn = conn or get_connection(_cfg.get_db_path())
    try:
        rows = _load_rows(_conn)
        out_dir = _wiki_dir(_cfg)
        counts = {"pages": 0, "topics": 0, "speakers": 0, "events": 0, "concepts": 0}
        by_topic: dict[str, list[sqlite3.Row]] = defaultdict(list)
        by_speaker: dict[str, list[sqlite3.Row]] = defaultdict(list)
        by_event: dict[str, list[sqlite3.Row]] = defaultdict(list)
        by_keyword: dict[str, list[sqlite3.Row]] = defaultdict(list)

        for row in rows:
            by_topic[row["topic"] or "unknown"].append(row)
            if row["speaker"]:
                by_speaker[row["speaker"]].append(row)
            if row["event_name"]:
                by_event[row["event_name"]].append(row)
            for kw in _keywords([row]):
                by_keyword[kw].append(row)

        _write_page(
            out_dir / "index.md",
            "Knowledge Corpus Wiki",
            "index",
            _theme_page(rows, _cfg),
            {"corpus": _cfg.corpus_name},
        )
        counts["pages"] += 1

        for topic, topic_rows in by_topic.items():
            _write_page(out_dir / "topics" / f"{_slug(topic)}.md", topic, "topic", _topic_page(topic, topic_rows, _cfg))
            counts["pages"] += 1
            counts["topics"] += 1

        for speaker, speaker_rows in by_speaker.items():
            _write_page(out_dir / "speakers" / f"{_slug(speaker)}.md", speaker, "speaker", _entity_page("speaker", speaker, speaker_rows, _cfg))
            counts["pages"] += 1
            counts["speakers"] += 1

        for event, event_rows in by_event.items():
            _write_page(out_dir / "events" / f"{_slug(event)}.md", event, "event", _entity_page("event", event, event_rows, _cfg))
            counts["pages"] += 1
            counts["events"] += 1

        for keyword, keyword_rows in sorted(by_keyword.items()):
            if len({r["file_id"] for r in keyword_rows}) < getattr(_cfg, "wiki_min_sources", 2):
                continue
            _write_page(out_dir / "concepts" / f"{_slug(keyword)}.md", keyword, "concept", _entity_page("concept", keyword, keyword_rows, _cfg))
            counts["pages"] += 1
            counts["concepts"] += 1

        return counts
    finally:
        if own_conn:
            _conn.close()
