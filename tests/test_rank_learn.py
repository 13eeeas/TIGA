"""tests/test_rank_learn.py — light preference boosts (no model training)."""

from __future__ import annotations

from pathlib import Path

from config import load_config
from core.rank_learn import (
    apply_learned_boosts,
    load_preferences,
    query_fingerprint,
    record_preference,
)


def _cfg(tmp_path: Path):
    import yaml
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"index_roots": [str(tmp_path)]}), encoding="utf-8")
    return load_config(config_file=cfg_file, work_dir=tmp_path)


def test_preference_requires_agreement(tmp_path: Path):
    cfg = _cfg(tmp_path)
    record_preference("NUS BIZ3 brief", "a/b/brief.pdf", 1, cfg=cfg)
    cands = [{"file_path": "x/a/b/brief.pdf", "final_score": 1.0}]
    apply_learned_boosts(cands, "NUS BIZ3 brief", cfg=cfg)
    assert cands[0]["final_score"] == 1.0  # one vote — no boost yet


def test_preference_boosts_after_two_agrees(tmp_path: Path):
    cfg = _cfg(tmp_path)
    record_preference("NUS BIZ3 brief", "a/b/brief.pdf", 1, cfg=cfg)
    record_preference("NUS BIZ3 brief", "a/b/brief.pdf", 1, cfg=cfg)
    cands = [
        {"file_path": "root/a/b/brief.pdf", "final_score": 1.0},
        {"file_path": "root/other.pdf", "final_score": 1.0},
    ]
    apply_learned_boosts(cands, "NUS BIZ3 brief", cfg=cfg)
    assert cands[0]["final_score"] > 1.0
    assert cands[0]["final_score"] <= 1.25
    assert cands[1]["final_score"] == 1.0


def test_fingerprint_stable():
    assert query_fingerprint("Hello, World!") == query_fingerprint("hello world")
