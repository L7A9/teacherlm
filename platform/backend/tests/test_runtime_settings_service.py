from __future__ import annotations

import json

import pytest

from config import Settings
from db.models import AppRuntimeSettingsRecord
from services.runtime_settings_service import RuntimeSettingsError, RuntimeSettingsService


def _settings() -> Settings:
    return Settings(
        settings_encryption_key="unit-test-secret",
        llama_cloud_api_key="env-parser-key",
        ollama_host="http://ollama.local:11434",
        ollama_chat_model="local-model",
    )


def _record(service: RuntimeSettingsService, **overrides: object) -> AppRuntimeSettingsRecord:
    values = {
        "id": "global",
        "llm_enabled": False,
        "llm_provider": "ollama",
        "llm_model": "",
        "llm_base_url": "",
        "llm_api_key_encrypted": None,
        "llama_cloud_api_key_encrypted": None,
    }
    values.update(overrides)
    if values.get("llm_api_key_encrypted") == "__secret__":
        values["llm_api_key_encrypted"] = service._encrypt_secret("provider-secret")
    if values.get("llama_cloud_api_key_encrypted") == "__secret__":
        values["llama_cloud_api_key_encrypted"] = service._encrypt_secret("parser-secret")
    return AppRuntimeSettingsRecord(**values)


def test_encrypts_keys_and_public_read_never_returns_secret() -> None:
    service = RuntimeSettingsService(_settings())
    encrypted = service._encrypt_secret("provider-secret")

    assert encrypted != "provider-secret"
    assert service._decrypt_secret(encrypted) == "provider-secret"

    public = service._public_read(
        _record(
            service,
            llm_enabled=True,
            llm_api_key_encrypted=encrypted,
            llama_cloud_api_key_encrypted=service._encrypt_secret("parser-secret"),
        )
    )
    payload = json.dumps(public.model_dump())
    assert public.llm.api_key_set is True
    assert public.parser.api_key_set is True
    assert "provider-secret" not in payload
    assert "parser-secret" not in payload


def test_parser_key_uses_database_first_then_env_fallback() -> None:
    service = RuntimeSettingsService(_settings())

    assert service._parser_api_key_from_record(None) == "env-parser-key"
    assert service._parser_api_key_from_record(_record(service)) == "env-parser-key"
    assert (
        service._parser_api_key_from_record(
            _record(service, llama_cloud_api_key_encrypted="__secret__")
        )
        == "parser-secret"
    )


@pytest.mark.parametrize(
    ("provider", "model", "base_url"),
    [
        ("ollama", "local-override", "http://ollama.settings:11434"),
        ("openai", "gpt-4.1-mini", "https://api.openai.com/v1"),
        ("anthropic", "claude-sonnet-4-5", "https://api.anthropic.com"),
        ("openai_compatible", "mistral-large", "https://api.mistral.ai/v1"),
    ],
)
def test_resolver_builds_backend_llm_profile(
    provider: str,
    model: str,
    base_url: str,
) -> None:
    service = RuntimeSettingsService(_settings())
    record = _record(
        service,
        llm_enabled=True,
        llm_provider=provider,
        llm_model=model,
        llm_base_url=base_url,
        llm_api_key_encrypted="__secret__",
    )

    resolved = service._resolved_options_from_record(
        record,
        {
            "language": "fr-fr",
            "llm": {
                "provider": "openai_compatible",
                "model": "client-model",
                "api_key": "client-secret",
            },
        },
    )

    assert resolved["language"] == "fr-fr"
    assert resolved["llm"] == {
        "enabled": True,
        "provider": provider,
        "model": model,
        "base_url": base_url,
        "api_key": "provider-secret",
    }


def test_client_sent_llm_options_are_ignored_when_db_profile_disabled() -> None:
    service = RuntimeSettingsService(_settings())
    resolved = service._resolved_options_from_record(
        _record(service, llm_enabled=False),
        {"language": "en-us", "llm": {"enabled": True, "api_key": "client-secret"}},
    )

    assert resolved == {"language": "en-us"}


def test_missing_encryption_key_blocks_secret_storage() -> None:
    service = RuntimeSettingsService(Settings(settings_encryption_key=""))

    with pytest.raises(RuntimeSettingsError):
        service._encrypt_secret("secret")
