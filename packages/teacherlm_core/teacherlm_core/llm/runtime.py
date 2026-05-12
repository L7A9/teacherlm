from __future__ import annotations

from contextvars import ContextVar
from typing import Any

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"

_CURRENT_LLM_OPTIONS: ContextVar[dict[str, Any] | None] = ContextVar(
    "teacherlm_current_llm_options",
    default=None,
)


def set_current_llm_options(options: dict[str, Any] | None) -> None:
    """Set per-request LLM provider overrides for the current async context."""

    raw = (options or {}).get("llm") if isinstance(options, dict) else None
    _CURRENT_LLM_OPTIONS.set(raw if isinstance(raw, dict) and raw.get("enabled") else None)


def get_current_llm_options() -> dict[str, Any] | None:
    return _CURRENT_LLM_OPTIONS.get()


def has_llm_override(options: dict[str, Any] | None = None) -> bool:
    raw = options if options is not None else get_current_llm_options()
    return isinstance(raw, dict) and bool(raw.get("enabled")) and bool(raw.get("model"))


def build_llm_client_kwargs(
    *,
    default_base_url: str,
    default_model: str,
    options: dict[str, Any] | None = None,
) -> dict[str, str | None]:
    raw = options if options is not None else get_current_llm_options()
    if not has_llm_override(raw):
        return {
            "provider": "ollama",
            "base_url": default_base_url,
            "model": default_model,
            "api_key": None,
        }

    provider = _normalize_provider(raw.get("provider"))
    raw_base_url = raw.get("base_url") or raw.get("api_base_url")
    base_url = str(raw_base_url or _default_base_url(provider, default_base_url)).strip()
    model = str(raw.get("model") or default_model).strip()
    api_key = raw.get("api_key")
    return {
        "provider": provider,
        "base_url": base_url,
        "model": model,
        "api_key": str(api_key).strip() if api_key else None,
    }


def _normalize_provider(value: object) -> str:
    provider = str(value or "ollama").strip().lower().replace("-", "_")
    if provider in {"openai_compatible", "openai_compat", "openai_compat_api"}:
        return "openai_compatible"
    if provider in {"anthropic", "claude"}:
        return "anthropic"
    return provider


def _default_base_url(provider: str, default_base_url: str) -> str:
    if provider == "ollama":
        return default_base_url
    if provider == "anthropic":
        return ANTHROPIC_DEFAULT_BASE_URL
    return OPENAI_DEFAULT_BASE_URL
