from __future__ import annotations

from dataclasses import dataclass
import json
import os
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class LLMConfig:
    provider: str = "disabled"
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    timeout_seconds: float = 45.0

    @property
    def enabled(self) -> bool:
        return self.provider not in {"", "disabled", "none", "off", "false"}


def load_llm_config(
    *,
    provider: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
) -> LLMConfig:
    """Load cloud/local LLM settings from explicit values and environment."""
    resolved_provider = (provider or os.getenv("STOCK_ADVISOR_LLM_PROVIDER") or "disabled").strip().lower()
    if resolved_provider in {"openai", "openai-compatible", "openai_compatible", "compatible"}:
        resolved_provider = "openai_compatible"
    if resolved_provider in {"local", "ollama"}:
        resolved_provider = "ollama"

    resolved_model = model or os.getenv("STOCK_ADVISOR_LLM_MODEL")
    resolved_base_url = base_url or os.getenv("STOCK_ADVISOR_LLM_BASE_URL")
    if resolved_provider == "openai_compatible":
        resolved_base_url = resolved_base_url or "https://api.openai.com/v1"
        resolved_model = resolved_model or "gpt-4o-mini"
    elif resolved_provider == "ollama":
        resolved_base_url = resolved_base_url or "http://localhost:11434"
        resolved_model = resolved_model or "llama3.1"

    timeout_raw = os.getenv("STOCK_ADVISOR_LLM_TIMEOUT_SECONDS", "45")
    try:
        timeout = float(timeout_raw)
    except ValueError:
        timeout = 45.0

    return LLMConfig(
        provider=resolved_provider,
        model=resolved_model,
        base_url=resolved_base_url,
        api_key=os.getenv("STOCK_ADVISOR_LLM_API_KEY") or os.getenv("OPENAI_API_KEY"),
        timeout_seconds=timeout,
    )


def synthesize_with_llm(
    *,
    system_prompt: str,
    user_prompt: str,
    config: LLMConfig,
) -> dict[str, Any]:
    """Generate a synthesis through a configured cloud/local LLM backend."""
    if not config.enabled:
        return {"used": False, "provider": config.provider, "warning": "LLM synthesis is disabled."}
    if not config.model:
        return {"used": False, "provider": config.provider, "warning": "LLM model is not configured."}

    if config.provider == "openai_compatible":
        return _openai_compatible_chat(system_prompt=system_prompt, user_prompt=user_prompt, config=config)
    if config.provider == "ollama":
        return _ollama_chat(system_prompt=system_prompt, user_prompt=user_prompt, config=config)
    return {"used": False, "provider": config.provider, "warning": f"Unsupported LLM provider: {config.provider}."}


def _openai_compatible_chat(*, system_prompt: str, user_prompt: str, config: LLMConfig) -> dict[str, Any]:
    if not config.api_key:
        return {
            "used": False,
            "provider": config.provider,
            "model": config.model,
            "warning": "Set STOCK_ADVISOR_LLM_API_KEY or OPENAI_API_KEY for OpenAI-compatible LLM synthesis.",
        }
    url = f"{str(config.base_url).rstrip('/')}/chat/completions"
    payload = {
        "model": config.model,
        "temperature": 0.2,
        "max_tokens": 900,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }
    return _post_json(url, payload, headers=headers, timeout=config.timeout_seconds, provider=config.provider, model=config.model)


def _ollama_chat(*, system_prompt: str, user_prompt: str, config: LLMConfig) -> dict[str, Any]:
    url = f"{str(config.base_url).rstrip('/')}/api/chat"
    payload = {
        "model": config.model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    result = _post_json(url, payload, headers={"Content-Type": "application/json"}, timeout=config.timeout_seconds, provider=config.provider, model=config.model)
    if result.get("used") and isinstance(result.get("raw"), dict):
        message = result["raw"].get("message") or {}
        result["content"] = message.get("content")
        result.pop("raw", None)
    return result


def _post_json(
    url: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str],
    timeout: float,
    provider: str,
    model: str | None,
) -> dict[str, Any]:
    try:
        request = Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urlopen(request, timeout=timeout) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except (OSError, HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
        return {"used": False, "provider": provider, "model": model, "warning": f"LLM request failed: {exc}"}

    if provider == "openai_compatible":
        choices = raw.get("choices") if isinstance(raw, dict) else []
        content = None
        if choices:
            content = ((choices[0] or {}).get("message") or {}).get("content")
        return {
            "used": bool(content),
            "provider": provider,
            "model": model,
            "content": content,
            "warning": None if content else "OpenAI-compatible response did not include message content.",
        }
    return {"used": True, "provider": provider, "model": model, "raw": raw}
