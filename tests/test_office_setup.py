"""Tests for office one-click setup helpers."""

from __future__ import annotations

from pathlib import Path

import yaml

from tools.office_setup import ensure_config, roots_need_setup


def test_roots_need_setup_when_missing(tmp_path: Path) -> None:
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump({"index_roots": [str(tmp_path / "nope")]}), encoding="utf-8")
    assert roots_need_setup(cfg) is True


def test_roots_need_setup_when_present(tmp_path: Path) -> None:
    proj = tmp_path / "261_tianmu"
    proj.mkdir()
    cfg = tmp_path / "config.yaml"
    cfg.write_text(yaml.dump({"index_roots": [str(proj)]}), encoding="utf-8")
    assert roots_need_setup(cfg) is False


def test_ensure_config_from_poc_template(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("TIGA_WORK_DIR", str(tmp_path))
    path = ensure_config(tmp_path / "config.yaml")
    assert path.exists()
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "index_roots" in data
    assert data.get("dedupe", {}).get("enabled") is True
