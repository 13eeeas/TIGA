"""
tests/test_config.py — Config loader unit tests.

Run with: pytest tests/test_config.py -v
"""

import os
from pathlib import Path

import pytest
import yaml

# Import without triggering the module-level cfg singleton
import importlib
import sys


def _write_config(tmp_path: Path, data: dict) -> Path:
    """Write a config dict to a temp config.yaml and return its path."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml.dump(data), encoding="utf-8")
    return config_file


def _minimal_config(tmp_path: Path) -> dict:
    """Minimum valid config dict."""
    return {
        "index_roots": [str(tmp_path / "archive")],
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_config_loads_defaults(tmp_path: Path) -> None:
    """Config with only required fields uses correct defaults."""
    from config import load_config

    config_file = _write_config(tmp_path, _minimal_config(tmp_path))
    cfg = load_config(config_file=config_file, work_dir=tmp_path)

    assert cfg.top_k == 5
    assert cfg.fts_weight == pytest.approx(0.4)
    assert cfg.vector_weight == pytest.approx(0.6)
    assert cfg.embed_model == "nomic-embed-text"
    assert cfg.chat_model == "mistral"
    assert cfg.server_port == 7860
    assert cfg.server_host == "0.0.0.0"
    assert cfg.ocr_enabled is False
    assert cfg.einstein_enable is False
    assert len(cfg.index_roots) == 1


def test_env_var_overrides_work_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """TIGA_WORK_DIR env var is used as work_dir when set."""
    from config import load_config

    custom_work = tmp_path / "custom_work"
    custom_work.mkdir()
    config_file = _write_config(custom_work, _minimal_config(tmp_path))

    monkeypatch.setenv("TIGA_WORK_DIR", str(custom_work))

    # Load without explicit work_dir — should pick up the env var
    # We still pass config_file explicitly to avoid filesystem side effects
    cfg = load_config(config_file=config_file)
    assert cfg.work_dir.resolve() == custom_work.resolve()


def test_missing_required_field_raises(tmp_path: Path) -> None:
    """ConfigError is raised when index_roots is absent."""
    from config import load_config, ConfigError

    config_file = _write_config(tmp_path, {"work_dir": str(tmp_path)})
    with pytest.raises(ConfigError, match="index_roots"):
        load_config(config_file=config_file, work_dir=tmp_path)


def test_path_normalization_returns_posix(tmp_path: Path) -> None:
    """index_roots are resolved to absolute paths; as_posix() uses forward slashes."""
    from config import load_config

    archive = tmp_path / "my archive"
    data = {"index_roots": [str(archive)]}
    config_file = _write_config(tmp_path, data)

    cfg = load_config(config_file=config_file, work_dir=tmp_path)

    for root in cfg.index_roots:
        posix = root.as_posix()
        # POSIX string must not contain backslashes
        assert "\\" not in posix, f"Backslash found in POSIX path: {posix}"
        # Must be absolute
        assert posix.startswith("/") or (
            len(posix) >= 3 and posix[1] == ":"
        ), f"Path is not absolute: {posix}"


def test_ollama_available_returns_bool_not_raises(tmp_path: Path) -> None:
    """ollama_available() always returns a bool, never raises."""
    from config import ollama_available

    # Point at a port nothing is listening on — must return False, not raise
    result = ollama_available("http://127.0.0.1:19999")
    assert isinstance(result, bool)
    assert result is False
