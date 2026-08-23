"""tests/test_llm_providers.py — enterprise API provider routing (mocked HTTP)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from config import load_config
from core.llm_providers import LLMError, chat_completion, resolve_api_key


def _cfg(tmp_path: Path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.dump({
            "index_roots": [str(tmp_path)],
            "compose": {
                "provider": "openai",
                "api_enabled": True,
                "model": "gpt-4o-mini",
            },
        }),
        encoding="utf-8",
    )
    return load_config(config_file=cfg_file, work_dir=tmp_path)


def test_resolve_api_key_from_generic_env(monkeypatch) -> None:
    monkeypatch.setenv("TIGA_LLM_API_KEY", "sk-test")
    assert resolve_api_key("openai") == "sk-test"


def test_chat_openai_success(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TIGA_OPENAI_API_KEY", "sk-test")
    cfg = _cfg(tmp_path)

    def fake_http(url, payload, headers, timeout):
        assert "gpt-4o-mini" in str(payload.get("model", ""))
        return {"choices": [{"message": {"content": "Enterprise answer."}}]}

    with patch("core.llm_providers._http_json", side_effect=fake_http):
        text, label = chat_completion(
            [{"role": "user", "content": "hi"}],
            cfg,
        )
    assert text == "Enterprise answer."
    assert "gpt-4o-mini" in label


def test_falls_back_to_ollama_when_no_api_key(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TIGA_LLM_API_KEY", raising=False)
    monkeypatch.delenv("TIGA_OPENAI_API_KEY", raising=False)
    cfg = _cfg(tmp_path)

    with patch("core.llm_providers._chat_ollama", return_value="local answer") as mock_ollama:
        text, label = chat_completion([{"role": "user", "content": "hi"}], cfg)
    assert text == "local answer"
    assert label.startswith("ollama:")
    mock_ollama.assert_called_once()


def test_raises_when_api_and_ollama_fail(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("TIGA_OPENAI_API_KEY", "sk-test")
    cfg = _cfg(tmp_path)
    cfg.compose_fallback_to_ollama = False

    with patch("core.llm_providers._http_json", side_effect=OSError("network")):
        with pytest.raises(LLMError):
            chat_completion([{"role": "user", "content": "hi"}], cfg)
