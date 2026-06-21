from __future__ import annotations

from typing import Any

import httpx
from teacherlm_core.llm.providers import LLMProviderConfig

from local_api.config import get_settings
from local_api.db import get_store, new_id, utc_now
from local_api.schemas import (
    CourseBuilderSettingsRead,
    CourseBuilderSettingsUpdate,
    GeneratorSettingsRead,
    GeneratorSettingsUpdate,
    ParserSettingsRead,
    ParserSettingsUpdate,
    ProviderPatch,
    ProviderRead,
    ProviderWrite,
    RetrievalSettingsRead,
    RetrievalSettingsUpdate,
    RuntimeSettingsRead,
)
from local_api.services.secrets import get_secret_box


class SettingsService:
    def list_providers(self) -> list[ProviderRead]:
        rows = get_store().query("SELECT * FROM llm_providers ORDER BY is_default_chat DESC, display_name ASC")
        return [self._public_provider(row) for row in rows]

    def get_provider(self, provider_id: str) -> ProviderRead | None:
        row = get_store().one("SELECT * FROM llm_providers WHERE id = ?", (provider_id,))
        return self._public_provider(row) if row else None

    def get_default_chat_provider_row(self) -> dict[str, Any] | None:
        return get_store().one(
            "SELECT * FROM llm_providers WHERE is_default_chat = 1 ORDER BY updated_at DESC LIMIT 1"
        )

    def get_default_chat_provider_config(self) -> LLMProviderConfig | None:
        row = self.get_default_chat_provider_row()
        if row is None:
            return None
        secret = get_secret_box().decrypt(row.get("api_key_ciphertext"))
        return LLMProviderConfig(
            provider_id=row["id"],
            display_name=row["display_name"],
            provider_type=row["provider_type"],
            base_url=row["base_url"],
            model_name=row["model_name"],
            api_key=secret,
        )

    def create_provider(self, payload: ProviderWrite) -> ProviderRead:
        provider_id = new_id("provider")
        now = utc_now()
        if payload.is_default_chat:
            get_store().execute("UPDATE llm_providers SET is_default_chat = 0")
        if payload.is_default_embedding:
            get_store().execute("UPDATE llm_providers SET is_default_embedding = 0")
        get_store().execute(
            """
            INSERT INTO llm_providers
              (id, display_name, provider_type, base_url, model_name, api_key_ciphertext,
               is_default_chat, is_default_embedding, status, metadata_json, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'unknown', '{}', ?, ?)
            """,
            (
                provider_id,
                payload.display_name,
                payload.provider_type,
                payload.base_url,
                payload.model_name,
                get_secret_box().encrypt(payload.api_key),
                1 if payload.is_default_chat else 0,
                1 if payload.is_default_embedding else 0,
                now,
                now,
            ),
        )
        return self.get_provider(provider_id) or ProviderRead(id=provider_id, display_name=payload.display_name, provider_type=payload.provider_type, base_url=payload.base_url, model_name=payload.model_name)

    def update_provider(self, provider_id: str, payload: ProviderPatch) -> ProviderRead:
        row = get_store().one("SELECT * FROM llm_providers WHERE id = ?", (provider_id,))
        if row is None:
            raise KeyError(provider_id)
        values: dict[str, Any] = {}
        for key in ("display_name", "provider_type", "base_url", "model_name"):
            value = getattr(payload, key)
            if value is not None:
                values[key] = value
        if payload.api_key is not None:
            values["api_key_ciphertext"] = get_secret_box().encrypt(payload.api_key)
        if payload.is_default_chat is not None:
            if payload.is_default_chat:
                get_store().execute("UPDATE llm_providers SET is_default_chat = 0")
            values["is_default_chat"] = 1 if payload.is_default_chat else 0
        if payload.is_default_embedding is not None:
            if payload.is_default_embedding:
                get_store().execute("UPDATE llm_providers SET is_default_embedding = 0")
            values["is_default_embedding"] = 1 if payload.is_default_embedding else 0
        values["updated_at"] = utc_now()
        assignments = ", ".join(f"{key} = ?" for key in values)
        get_store().execute(f"UPDATE llm_providers SET {assignments} WHERE id = ?", (*values.values(), provider_id))
        return self.get_provider(provider_id) or self._public_provider(row)

    def delete_provider(self, provider_id: str) -> None:
        row = get_store().one("SELECT * FROM llm_providers WHERE id = ?", (provider_id,))
        if row is None:
            raise KeyError(provider_id)
        get_store().execute("DELETE FROM llm_providers WHERE id = ?", (provider_id,))
        if row.get("is_default_chat") and self.get_default_chat_provider_row() is None:
            replacement = get_store().one("SELECT id FROM llm_providers ORDER BY updated_at DESC LIMIT 1")
            if replacement:
                get_store().execute("UPDATE llm_providers SET is_default_chat = 1, updated_at = ? WHERE id = ?", (utc_now(), replacement["id"]))
        if row.get("is_default_embedding"):
            replacement = get_store().one("SELECT id FROM llm_providers ORDER BY updated_at DESC LIMIT 1")
            if replacement:
                get_store().execute("UPDATE llm_providers SET is_default_embedding = 1, updated_at = ? WHERE id = ?", (utc_now(), replacement["id"]))

    async def test_provider(self, provider_id: str) -> ProviderRead:
        row = get_store().one("SELECT * FROM llm_providers WHERE id = ?", (provider_id,))
        if row is None:
            raise KeyError(provider_id)
        status = "ok"
        try:
            if row["provider_type"] == "ollama":
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{row['base_url'].rstrip('/')}/api/tags")
                    response.raise_for_status()
            elif not row.get("api_key_ciphertext"):
                status = "missing_api_key"
        except Exception:  # noqa: BLE001
            status = "unreachable"
        get_store().execute(
            "UPDATE llm_providers SET status = ?, updated_at = ? WHERE id = ?",
            (status, utc_now(), provider_id),
        )
        return self.get_provider(provider_id) or self._public_provider(row)

    def get_parser_settings(self) -> ParserSettingsRead:
        row = get_store().one("SELECT * FROM parser_settings WHERE id = 'default'")
        env_key_set = bool(get_settings().llama_cloud_api_key.strip())
        if row is None:
            get_store().execute(
                """
                INSERT INTO parser_settings (id, use_local_parsers_only, status, updated_at)
                VALUES ('default', 1, 'local', ?)
                """,
                (utc_now(),),
            )
            return ParserSettingsRead(llama_cloud_api_key_set=env_key_set)
        return ParserSettingsRead(
            llama_cloud_api_key_set=bool(row.get("llama_cloud_api_key_ciphertext")) or env_key_set,
            use_local_parsers_only=bool(row.get("use_local_parsers_only")),
            status=row.get("status") or "local",
        )

    def parser_api_key(self) -> str:
        row = get_store().one("SELECT llama_cloud_api_key_ciphertext FROM parser_settings WHERE id = 'default'")
        if row and row.get("llama_cloud_api_key_ciphertext"):
            secret = get_secret_box().decrypt(row.get("llama_cloud_api_key_ciphertext"))
            if secret:
                return secret
        return get_settings().llama_cloud_api_key.strip()

    def update_parser_settings(self, payload: ParserSettingsUpdate) -> ParserSettingsRead:
        current = self.get_parser_settings()
        api_key_ciphertext = None
        if payload.llama_cloud_api_key is not None:
            api_key_ciphertext = get_secret_box().encrypt(payload.llama_cloud_api_key)
        local_only = current.use_local_parsers_only if payload.use_local_parsers_only is None else payload.use_local_parsers_only
        has_key = bool(api_key_ciphertext) or current.llama_cloud_api_key_set or bool(get_settings().llama_cloud_api_key.strip())
        status = "local" if local_only else ("llamaparse_configured" if has_key else "missing_api_key")
        if payload.clear_llama_cloud_api_key:
            api_key_ciphertext = None
            local_only = True
            status = "local"
        key_update_sql = (
            "NULL"
            if payload.clear_llama_cloud_api_key
            else "COALESCE(excluded.llama_cloud_api_key_ciphertext, parser_settings.llama_cloud_api_key_ciphertext)"
        )
        get_store().execute(
            f"""
            INSERT INTO parser_settings
              (id, llama_cloud_api_key_ciphertext, use_local_parsers_only, status, updated_at)
            VALUES ('default', ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              llama_cloud_api_key_ciphertext = {key_update_sql},
              use_local_parsers_only = excluded.use_local_parsers_only,
              status = excluded.status,
              updated_at = excluded.updated_at
            """,
            (api_key_ciphertext, 1 if local_only else 0, status, utc_now()),
        )
        return self.get_parser_settings()

    def get_coursebuilder_settings(self) -> CourseBuilderSettingsRead:
        row = get_store().one("SELECT * FROM coursebuilder_settings WHERE id = 'default'")
        if row is None:
            get_store().execute(
                """
                INSERT INTO coursebuilder_settings (id, sequential_unlocking_enabled, updated_at)
                VALUES ('default', 1, ?)
                """,
                (utc_now(),),
            )
            return CourseBuilderSettingsRead()
        return CourseBuilderSettingsRead(
            sequential_unlocking_enabled=bool(row.get("sequential_unlocking_enabled")),
        )

    def update_coursebuilder_settings(
        self,
        payload: CourseBuilderSettingsUpdate,
    ) -> CourseBuilderSettingsRead:
        current = self.get_coursebuilder_settings()
        enabled = (
            current.sequential_unlocking_enabled
            if payload.sequential_unlocking_enabled is None
            else payload.sequential_unlocking_enabled
        )
        get_store().execute(
            """
            INSERT INTO coursebuilder_settings (id, sequential_unlocking_enabled, updated_at)
            VALUES ('default', ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              sequential_unlocking_enabled = excluded.sequential_unlocking_enabled,
              updated_at = excluded.updated_at
            """,
            (1 if enabled else 0, utc_now()),
        )
        return self.get_coursebuilder_settings()

    def get_generator_settings(self) -> GeneratorSettingsRead:
        row = get_store().one("SELECT * FROM generator_settings WHERE id = 'default'")
        if row is None:
            get_store().execute(
                """
                INSERT INTO generator_settings (id, podcast_audio_enabled, updated_at)
                VALUES ('default', 1, ?)
                """,
                (utc_now(),),
            )
            return GeneratorSettingsRead()
        return GeneratorSettingsRead(
            podcast_audio_enabled=bool(row.get("podcast_audio_enabled")),
        )

    def update_generator_settings(
        self,
        payload: GeneratorSettingsUpdate,
    ) -> GeneratorSettingsRead:
        current = self.get_generator_settings()
        audio_enabled = (
            current.podcast_audio_enabled
            if payload.podcast_audio_enabled is None
            else payload.podcast_audio_enabled
        )
        get_store().execute(
            """
            INSERT INTO generator_settings (id, podcast_audio_enabled, updated_at)
            VALUES ('default', ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              podcast_audio_enabled = excluded.podcast_audio_enabled,
              updated_at = excluded.updated_at
            """,
            (1 if audio_enabled else 0, utc_now()),
        )
        return self.get_generator_settings()

    def get_retrieval_settings(self) -> RetrievalSettingsRead:
        row = self._retrieval_row()
        from local_api.services.vector_service import get_vector_service

        return self._retrieval_public(row, index_status=get_vector_service().index_status())

    def effective_retrieval_settings(self) -> RetrievalSettingsRead:
        return self._retrieval_public(self._retrieval_row(), index_status={})

    def update_retrieval_settings(self, payload: RetrievalSettingsUpdate) -> RetrievalSettingsRead:
        current = self.effective_retrieval_settings()
        values = current.model_dump()
        values.pop("index_status", None)
        values.pop("embedding_model_candidates", None)
        for key, value in payload.model_dump(exclude_unset=True).items():
            if value is not None:
                values[key] = value
        values["embedding_batch_size"] = max(1, int(values["embedding_batch_size"]))
        values["embedding_dim"] = max(1, int(values["embedding_dim"]))
        values["retrieval_top_k"] = max(1, int(values["retrieval_top_k"]))
        values["retrieval_dense_candidate_k"] = max(values["retrieval_top_k"], int(values["retrieval_dense_candidate_k"]))
        values["retrieval_sparse_candidate_k"] = max(values["retrieval_top_k"], int(values["retrieval_sparse_candidate_k"]))
        now = utc_now()
        get_store().execute(
            """
            INSERT INTO retrieval_settings
              (id, embedding_model, embedding_dim, embedding_batch_size,
               retrieval_top_k, retrieval_dense_candidate_k, retrieval_sparse_candidate_k,
               retrieval_hyde_enabled, retrieval_rerank_enabled, retrieval_reranker_model,
               retrieval_graph_enabled, updated_at)
            VALUES ('default', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              embedding_model = excluded.embedding_model,
              embedding_dim = excluded.embedding_dim,
              embedding_batch_size = excluded.embedding_batch_size,
              retrieval_top_k = excluded.retrieval_top_k,
              retrieval_dense_candidate_k = excluded.retrieval_dense_candidate_k,
              retrieval_sparse_candidate_k = excluded.retrieval_sparse_candidate_k,
              retrieval_hyde_enabled = excluded.retrieval_hyde_enabled,
              retrieval_rerank_enabled = excluded.retrieval_rerank_enabled,
              retrieval_reranker_model = excluded.retrieval_reranker_model,
              retrieval_graph_enabled = excluded.retrieval_graph_enabled,
              updated_at = excluded.updated_at
            """,
            (
                values["embedding_model"],
                int(values["embedding_dim"]),
                int(values["embedding_batch_size"]),
                int(values["retrieval_top_k"]),
                int(values["retrieval_dense_candidate_k"]),
                int(values["retrieval_sparse_candidate_k"]),
                1 if values["retrieval_hyde_enabled"] else 0,
                1 if values["retrieval_rerank_enabled"] else 0,
                values["retrieval_reranker_model"],
                1 if values["retrieval_graph_enabled"] else 0,
                now,
            ),
        )
        return self.get_retrieval_settings()

    def runtime_settings(self) -> RuntimeSettingsRead:
        providers = self.list_providers()
        parser = self.get_parser_settings()
        chat = next((provider for provider in providers if provider.is_default_chat), None)
        embedding = next((provider for provider in providers if provider.is_default_embedding), None)
        return RuntimeSettingsRead(
            default_chat_provider=chat,
            default_embedding_provider=embedding,
            parser=parser,
            retrieval=self.get_retrieval_settings(),
        )

    def _public_provider(self, row: dict[str, Any]) -> ProviderRead:
        return ProviderRead(
            id=row["id"],
            display_name=row["display_name"],
            provider_type=row["provider_type"],
            base_url=row["base_url"],
            model_name=row["model_name"],
            api_key_set=bool(row.get("api_key_ciphertext")),
            is_default_chat=bool(row.get("is_default_chat")),
            is_default_embedding=bool(row.get("is_default_embedding")),
            status=row.get("status") or "unknown",
        )

    def _retrieval_row(self) -> dict[str, Any]:
        row = get_store().one("SELECT * FROM retrieval_settings WHERE id = 'default'")
        if row is not None:
            return row
        defaults = get_settings()
        now = utc_now()
        get_store().execute(
            """
            INSERT INTO retrieval_settings
              (id, embedding_model, embedding_dim, embedding_batch_size,
               retrieval_top_k, retrieval_dense_candidate_k, retrieval_sparse_candidate_k,
               retrieval_hyde_enabled, retrieval_rerank_enabled, retrieval_reranker_model,
               retrieval_graph_enabled, updated_at)
            VALUES ('default', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                defaults.embedding_model,
                defaults.embedding_dim,
                defaults.embedding_batch_size,
                defaults.retrieval_top_k,
                defaults.retrieval_dense_candidate_k,
                defaults.retrieval_sparse_candidate_k,
                1 if defaults.retrieval_hyde_enabled else 0,
                1 if defaults.retrieval_rerank_enabled else 0,
                defaults.retrieval_reranker_model,
                1 if defaults.retrieval_graph_enabled else 0,
                now,
            ),
        )
        return get_store().one("SELECT * FROM retrieval_settings WHERE id = 'default'") or {}

    def _retrieval_public(self, row: dict[str, Any], *, index_status: dict[str, Any]) -> RetrievalSettingsRead:
        defaults = get_settings()
        return RetrievalSettingsRead(
            embedding_model=row.get("embedding_model") or defaults.embedding_model,
            embedding_dim=int(row.get("embedding_dim") or defaults.embedding_dim),
            embedding_batch_size=int(row.get("embedding_batch_size") or defaults.embedding_batch_size),
            embedding_model_candidates=list(defaults.embedding_model_candidates),
            retrieval_top_k=int(row.get("retrieval_top_k") or defaults.retrieval_top_k),
            retrieval_dense_candidate_k=int(row.get("retrieval_dense_candidate_k") or defaults.retrieval_dense_candidate_k),
            retrieval_sparse_candidate_k=int(row.get("retrieval_sparse_candidate_k") or defaults.retrieval_sparse_candidate_k),
            retrieval_hyde_enabled=bool(row.get("retrieval_hyde_enabled")),
            retrieval_rerank_enabled=bool(row.get("retrieval_rerank_enabled")),
            retrieval_reranker_model=row.get("retrieval_reranker_model") or defaults.retrieval_reranker_model,
            retrieval_graph_enabled=bool(row.get("retrieval_graph_enabled")),
            index_status=index_status,
        )


_settings_service: SettingsService | None = None


def get_settings_service() -> SettingsService:
    global _settings_service
    if _settings_service is None:
        _settings_service = SettingsService()
    return _settings_service
