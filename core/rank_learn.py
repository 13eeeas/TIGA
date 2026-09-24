"""
core/rank_learn.py — Light preference learning for Hunt ranking.

Not model fine-tuning. Collects thumbs / explicit preferences and applies
capped soft boosts at rank time. Changes that lack enough agreement are ignored.

Store: tiga_work/rank_preferences.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import cfg as _module_cfg, Config

logger = logging.getLogger(__name__)

_STORE_NAME = "rank_preferences.json"
_MIN_AGREE = 2
_MAX_BOOST = 1.25
_MAX_PENALTY = 0.85


def _store_path(cfg: Config | None = None) -> Path:
    c = cfg or _module_cfg
    return Path(c.work_dir) / _STORE_NAME


def _norm_query(q: str) -> str:
    q = (q or "").lower().strip()
    q = re.sub(r"[^\w\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q


def query_fingerprint(q: str) -> str:
    return hashlib.sha1(_norm_query(q).encode("utf-8")).hexdigest()[:16]


def _path_key(path: str) -> str:
    p = (path or "").replace("\\", "/").strip()
    # Prefer stable suffix (filename + 1–2 parent folders)
    parts = [x for x in p.split("/") if x]
    if len(parts) >= 3:
        return "/".join(parts[-3:]).lower()
    return p.lower()


@dataclass
class Preference:
    query_fp: str
    path_key: str
    score: int  # net +1 / -1 votes
    votes: int
    last_query: str
    updated_at: str

    def multiplier(self) -> float:
        if self.votes < _MIN_AGREE:
            return 1.0
        if self.score > 0:
            # Soft ramp: 2 agrees → ~1.1, more → up to MAX
            return min(_MAX_BOOST, 1.0 + 0.05 * min(self.score, 5))
        if self.score < 0:
            return max(_MAX_PENALTY, 1.0 + 0.05 * max(self.score, -5))
        return 1.0


def load_preferences(cfg: Config | None = None) -> dict[str, Preference]:
    path = _store_path(cfg)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.warning("rank_learn load failed: %s", e)
        return {}
    out: dict[str, Preference] = {}
    for row in raw.get("preferences", []):
        pref = Preference(
            query_fp=row["query_fp"],
            path_key=row["path_key"],
            score=int(row.get("score", 0)),
            votes=int(row.get("votes", 0)),
            last_query=row.get("last_query", ""),
            updated_at=row.get("updated_at", ""),
        )
        out[f"{pref.query_fp}|{pref.path_key}"] = pref
    return out


def save_preferences(prefs: dict[str, Preference], cfg: Config | None = None) -> Path:
    path = _store_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "min_agree": _MIN_AGREE,
        "max_boost": _MAX_BOOST,
        "preferences": [
            {
                "query_fp": p.query_fp,
                "path_key": p.path_key,
                "score": p.score,
                "votes": p.votes,
                "last_query": p.last_query,
                "updated_at": p.updated_at,
                "multiplier": round(p.multiplier(), 4),
            }
            for p in sorted(prefs.values(), key=lambda x: (-abs(x.score), x.query_fp))
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def record_preference(
    query: str,
    result_path: str,
    rating: int,
    *,
    cfg: Config | None = None,
) -> Preference | None:
    """rating: +1 useful / -1 not useful."""
    if rating not in (1, -1):
        return None
    if not query or not result_path:
        return None
    prefs = load_preferences(cfg)
    fp = query_fingerprint(query)
    key = _path_key(result_path)
    pk = f"{fp}|{key}"
    now = datetime.now(timezone.utc).isoformat()
    pref = prefs.get(pk) or Preference(
        query_fp=fp,
        path_key=key,
        score=0,
        votes=0,
        last_query=_norm_query(query)[:200],
        updated_at=now,
    )
    pref.score += rating
    pref.votes += 1
    pref.last_query = _norm_query(query)[:200]
    pref.updated_at = now
    prefs[pk] = pref
    save_preferences(prefs, cfg)
    return pref


def mine_feedback_table(
    conn: sqlite3.Connection,
    *,
    cfg: Config | None = None,
) -> int:
    """Import existing feedback rows into the preference store."""
    rows = conn.execute(
        "SELECT query, result_id, rating FROM feedback "
        "WHERE rating IN (1, -1) AND result_id IS NOT NULL AND query IS NOT NULL"
    ).fetchall()
    n = 0
    for r in rows:
        rating = int(r["rating"] if isinstance(r, sqlite3.Row) else r[2])
        query = r["query"] if isinstance(r, sqlite3.Row) else r[0]
        path = r["result_id"] if isinstance(r, sqlite3.Row) else r[1]
        if record_preference(query, str(path), rating, cfg=cfg):
            n += 1
    return n


def apply_learned_boosts(
    candidates: list[dict[str, Any]],
    query: str,
    *,
    cfg: Config | None = None,
) -> None:
    """Multiply final_score in-place using agreed preferences."""
    prefs = load_preferences(cfg)
    if not prefs:
        return
    fp = query_fingerprint(query)
    relevant = [p for p in prefs.values() if p.query_fp == fp and p.multiplier() != 1.0]
    if not relevant:
        return
    for c in candidates:
        path = c.get("file_path") or c.get("rel_path") or c.get("citation") or ""
        key = _path_key(str(path))
        for p in relevant:
            if key.endswith(p.path_key) or p.path_key in key or key.endswith(
                p.path_key.split("/")[-1]
            ):
                c["final_score"] = float(c.get("final_score") or 0.0) * p.multiplier()
                c["rank_learn_mult"] = p.multiplier()
                break
