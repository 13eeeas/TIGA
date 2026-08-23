"""
core/llm_providers.py — Pluggable LLM backends for Einstein synthesis.

Supports enterprise API providers (OpenAI-compatible, Azure OpenAI, Anthropic)
and local Ollama fallback. API keys come from environment variables only.

Env vars (first match wins for the active provider):
  TIGA_LLM_API_KEY          — generic override
  TIGA_OPENAI_API_KEY       — OpenAI / Azure
  TIGA_ANTHROPIC_API_KEY    — Anthropic
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any

from config import Config

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when no provider can complete the chat request."""


def resolve_api_key(provider: str) -> str | None:
    """Return API key for provider from environment (never from config files)."""
    generic = os.environ.get("TIGA_LLM_API_KEY", "").strip()
    if generic:
        return generic
    if provider in ("openai", "azure"):
        key = os.environ.get("TIGA_OPENAI_API_KEY", "").strip()
        return key or None
    if provider == "anthropic":
        key = os.environ.get("TIGA_ANTHROPIC_API_KEY", "").strip()
        return key or None
    return None


def _http_json(
    url: str,
    payload: dict[str, Any],
    headers: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={**headers, "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def _chat_openai(
    messages: list[dict[str, str]],
    cfg: Config,
    *,
    api_key: str,
) -> str:
    if cfg.compose_azure_endpoint:
        base = cfg.compose_azure_endpoint.rstrip("/")
        deployment = cfg.compose_azure_deployment or cfg.compose_model
        url = (
            f"{base}/openai/deployments/{deployment}/chat/completions"
            f"?api-version={cfg.compose_azure_api_version}"
        )
        headers = {"api-key": api_key}
    else:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}"}

    data = _http_json(
        url,
        {
            "model": cfg.compose_model,
            "messages": messages,
            "max_tokens": cfg.compose_max_tokens,
            "temperature": 0.2,
        },
        headers,
        cfg.compose_timeout,
    )
    return data["choices"][0]["message"]["content"]


def _chat_anthropic(
    messages: list[dict[str, str]],
    cfg: Config,
    *,
    api_key: str,
) -> str:
    system_parts: list[str] = []
    chat_messages: list[dict[str, str]] = []
    for m in messages:
        if m["role"] == "system":
            system_parts.append(m["content"])
        else:
            chat_messages.append({"role": m["role"], "content": m["content"]})

    payload: dict[str, Any] = {
        "model": cfg.compose_model,
        "max_tokens": cfg.compose_max_tokens,
        "messages": chat_messages,
        "temperature": 0.2,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)

    data = _http_json(
        "https://api.anthropic.com/v1/messages",
        payload,
        {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        cfg.compose_timeout,
    )
    parts = data.get("content") or []
    texts = [p.get("text", "") for p in parts if p.get("type") == "text"]
    return "".join(texts)


def _chat_ollama(messages: list[dict[str, str]], cfg: Config) -> str:
    url = cfg.ollama_base_url.rstrip("/") + "/api/chat"
    data = _http_json(
        url,
        {
            "model": cfg.chat_model,
            "messages": messages,
            "stream": False,
            "options": {"num_ctx": cfg.num_ctx},
        },
        {},
        cfg.ollama_timeout,
    )
    return data["message"]["content"]


def chat_completion(
    messages: list[dict[str, str]],
    cfg: Config,
    *,
    purpose: str = "compose",
) -> tuple[str, str]:
    """
    Run chat completion using configured provider.

    Returns:
        (content, provider_label) e.g. ("answer...", "openai:gpt-4o-mini")
    """
    provider = (cfg.compose_provider or "ollama").lower()
    use_api = cfg.compose_api_enabled and provider in ("openai", "azure", "anthropic")

    if use_api:
        api_key = resolve_api_key(provider)
        if not api_key:
            logger.warning(
                "%s: compose.api_enabled but no API key in env; falling back to ollama",
                purpose,
            )
            use_api = False

    if use_api:
        try:
            if provider == "anthropic":
                text = _chat_anthropic(messages, cfg, api_key=api_key)  # type: ignore[arg-type]
                label = f"anthropic:{cfg.compose_model}"
            else:
                text = _chat_openai(messages, cfg, api_key=api_key)  # type: ignore[arg-type]
                label = f"{'azure' if cfg.compose_azure_endpoint else 'openai'}:{cfg.compose_model}"
            return text, label
        except Exception as exc:
            logger.warning("%s: API provider %s failed: %s", purpose, provider, exc)
            if cfg.compose_fallback_to_ollama:
                text = _chat_ollama(messages, cfg)
                return text, f"ollama:{cfg.chat_model}(api-fallback)"
            raise LLMError(str(exc)) from exc

    try:
        text = _chat_ollama(messages, cfg)
        return text, f"ollama:{cfg.chat_model}"
    except Exception as exc:
        raise LLMError(str(exc)) from exc
