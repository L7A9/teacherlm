from __future__ import annotations

import uuid
from dataclasses import dataclass

from llama_cloud import AsyncLlamaCloud

from config import Settings, get_settings
from services.storage_service import StorageService, get_storage


class ParsingError(RuntimeError):
    pass


class ParsingTimeout(ParsingError):
    pass


@dataclass(slots=True)
class ParseResult:
    markdown: str
    markdown_key: str  # MinIO object key where markdown was persisted
    job_id: str


class ParsingService:
    """Parses uploaded files to markdown via LlamaCloud (llama-cloud>=2.0).

    Does NOT use llama-parse or llama-cloud-services (both deprecated).
    """

    def __init__(
        self,
        settings: Settings | None = None,
        storage: StorageService | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._storage = storage or get_storage()
        self._client = AsyncLlamaCloud(
            api_key=self._settings.llama_cloud_api_key,
            base_url=self._settings.llama_cloud_base_url,
        )

    async def parse_to_markdown(
        self,
        *,
        conversation_id: uuid.UUID | str,
        filename: str,
        data: bytes,
    ) -> ParseResult:
        result = await self._client.parsing.parse(
            tier="cost_effective",
            version="latest",
            upload_file=(filename, data),
            expand=["markdown"],
            polling_interval=self._settings.llama_cloud_poll_interval_s,
            timeout=self._settings.llama_cloud_timeout_s,
        )

        markdown = self._extract_markdown(result)
        job_id = self._extract_job_id(result)

        key = self._storage.parsed_key(conversation_id, str(uuid.uuid4()))
        await self._storage.put_text(key, markdown)
        return ParseResult(markdown=markdown, markdown_key=key, job_id=job_id)

    @staticmethod
    def _extract_markdown(result: object) -> str:
        for attr in ("markdown_full", "markdown", "text_full", "text"):
            value = getattr(result, attr, None)
            if isinstance(value, str) and value:
                return value
            pages = getattr(value, "pages", None)
            if pages:
                joined = "\n\n".join(
                    p for p in (getattr(page, "markdown", None) for page in pages) if p
                )
                if joined:
                    return joined
        raise ParsingError(f"LlamaCloud parse result had no markdown payload (got {result!r})")

    @staticmethod
    def _extract_job_id(result: object) -> str:
        job = getattr(result, "job", None)
        for holder in (job, result):
            for attr in ("id", "job_id"):
                value = getattr(holder, attr, None)
                if value:
                    return str(value)
        return ""


_parser: ParsingService | None = None


def get_parser() -> ParsingService:
    global _parser
    if _parser is None:
        _parser = ParsingService()
    return _parser
