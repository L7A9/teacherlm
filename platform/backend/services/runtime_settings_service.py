from __future__ import annotations

import base64
import hashlib
from typing import Any

from cryptography.fernet import Fernet, InvalidToken
from sqlalchemy.ext.asyncio import AsyncSession

from config import Settings, get_settings
from db.models import AppRuntimeSettingsRecord
from schemas.runtime_settings import (
    LlmRuntimeSettingsRead,
    RuntimeSettingsRead,
    RuntimeSettingsUpdate,
)


SETTINGS_RECORD_ID = "global"
SUPPORTED_LLM_PROVIDERS = {"ollama", "openai", "anthropic", "openai_compatible"}
OPENAI_DEFAULT_BASE_URL = "https://api.openai.com/v1"
ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"


class RuntimeSettingsError(RuntimeError):
    pass


class RuntimeSettingsService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def ensure_schema(self, session: AsyncSession) -> None:
        def create(sync_session: Any) -> None:
            AppRuntimeSettingsRecord.__table__.create(sync_session.connection(), checkfirst=True)

        await session.run_sync(create)

    async def get_public_settings(self, session: AsyncSession) -> RuntimeSettingsRead:
        record = await self._get_or_create_record(session)
        return self._public_read(record)

    async def update_settings(
        self,
        session: AsyncSession,
        update: RuntimeSettingsUpdate,
    ) -> RuntimeSettingsRead:
        record = await self._get_or_create_record(session)

        if update.llm is not None:
            llm = update.llm
            fields = llm.model_fields_set
            if "enabled" in fields and llm.enabled is not None:
                record.llm_enabled = bool(llm.enabled)
            if "provider" in fields and llm.provider is not None:
                record.llm_provider = self._normalize_provider(llm.provider)
            if "model" in fields and llm.model is not None:
                record.llm_model = llm.model.strip()
            if "api_link" in fields and llm.api_link is not None:
                record.llm_base_url = llm.api_link.strip()
            if "api_key" in fields:
                key = (llm.api_key or "").strip()
                record.llm_api_key_encrypted = self._encrypt_secret(key) if key else None

        if update.parser is not None:
            parser = update.parser
            if "api_key" in parser.model_fields_set:
                key = (parser.api_key or "").strip()
                record.llama_cloud_api_key_encrypted = self._encrypt_secret(key) if key else None

        await session.flush()
        return self._public_read(record)

    def sanitize_client_options(self, options: dict[str, Any] | None) -> dict[str, Any]:
        if not isinstance(options, dict):
            return {}
        sanitized = dict(options)
        sanitized.pop("llm", None)
        return sanitized

    async def resolve_options(
        self,
        session: AsyncSession,
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if not _is_async_session_like(session):
            return self.sanitize_client_options(options)
        record = await self._get_or_create_record(session)
        return self._resolved_options_from_record(record, options)

    async def parser_api_key(self, session: AsyncSession) -> str:
        if not _is_async_session_like(session):
            return self._settings.llama_cloud_api_key
        record = await self._get_or_create_record(session)
        return self._parser_api_key_from_record(record)

    async def _get_or_create_record(self, session: AsyncSession) -> AppRuntimeSettingsRecord:
        await self.ensure_schema(session)
        record = await session.get(AppRuntimeSettingsRecord, SETTINGS_RECORD_ID)
        if record is not None:
            return record
        record = AppRuntimeSettingsRecord(
            id=SETTINGS_RECORD_ID,
            llm_enabled=False,
            llm_provider="ollama",
            llm_model="",
            llm_base_url="",
        )
        session.add(record)
        await session.flush()
        return record

    def _public_read(self, record: AppRuntimeSettingsRecord) -> RuntimeSettingsRead:
        return RuntimeSettingsRead(
            llm=LlmRuntimeSettingsRead(
                enabled=bool(record.llm_enabled),
                provider=self._normalize_provider(record.llm_provider),
                model=record.llm_model or "",
                api_link=record.llm_base_url or "",
                api_key_set=bool(record.llm_api_key_encrypted),
            ),
            parser={"api_key_set": bool(record.llama_cloud_api_key_encrypted)},
        )

    def _resolved_options_from_record(
        self,
        record: AppRuntimeSettingsRecord,
        options: dict[str, Any] | None,
    ) -> dict[str, Any]:
        resolved = self.sanitize_client_options(options)
        if not record.llm_enabled:
            return resolved

        provider = self._normalize_provider(record.llm_provider)
        model = (record.llm_model or "").strip() or self._default_model(provider)
        if not model:
            return resolved

        base_url = (record.llm_base_url or "").strip() or self._default_base_url(provider)
        llm: dict[str, Any] = {
            "enabled": True,
            "provider": provider,
            "model": model,
            "base_url": base_url,
        }
        if record.llm_api_key_encrypted:
            llm["api_key"] = self._decrypt_secret(record.llm_api_key_encrypted)
        resolved["llm"] = llm
        return resolved

    def _parser_api_key_from_record(self, record: AppRuntimeSettingsRecord | None) -> str:
        if record is not None and record.llama_cloud_api_key_encrypted:
            return self._decrypt_secret(record.llama_cloud_api_key_encrypted)
        return self._settings.llama_cloud_api_key

    def _encrypt_secret(self, value: str) -> str:
        if not value:
            return ""
        return self._fernet().encrypt(value.encode("utf-8")).decode("utf-8")

    def _decrypt_secret(self, value: str) -> str:
        try:
            return self._fernet().decrypt(value.encode("utf-8")).decode("utf-8")
        except InvalidToken as exc:
            raise RuntimeSettingsError(
                "Stored runtime setting could not be decrypted. Check SETTINGS_ENCRYPTION_KEY."
            ) from exc

    def _fernet(self) -> Fernet:
        secret = self._settings.settings_encryption_key.strip()
        if not secret:
            raise RuntimeSettingsError("SETTINGS_ENCRYPTION_KEY is required to store API keys.")
        try:
            return Fernet(secret.encode("utf-8"))
        except ValueError:
            digest = hashlib.sha256(secret.encode("utf-8")).digest()
            return Fernet(base64.urlsafe_b64encode(digest))

    def _normalize_provider(self, provider: object) -> str:
        value = str(provider or "ollama").strip().lower().replace("-", "_")
        if value in {"openai_compat", "openai_compat_api"}:
            value = "openai_compatible"
        if value in {"claude"}:
            value = "anthropic"
        return value if value in SUPPORTED_LLM_PROVIDERS else "ollama"

    def _default_base_url(self, provider: str) -> str:
        if provider == "ollama":
            return self._settings.ollama_host
        if provider == "anthropic":
            return ANTHROPIC_DEFAULT_BASE_URL
        return OPENAI_DEFAULT_BASE_URL

    def _default_model(self, provider: str) -> str:
        if provider == "openai":
            return "gpt-4.1-mini"
        if provider == "anthropic":
            return "claude-sonnet-4-5"
        return self._settings.ollama_chat_model


_runtime_settings_service: RuntimeSettingsService | None = None


def get_runtime_settings_service() -> RuntimeSettingsService:
    global _runtime_settings_service
    if _runtime_settings_service is None:
        _runtime_settings_service = RuntimeSettingsService()
    return _runtime_settings_service


def _is_async_session_like(session: object) -> bool:
    return hasattr(session, "run_sync") and hasattr(session, "get") and hasattr(session, "flush")
