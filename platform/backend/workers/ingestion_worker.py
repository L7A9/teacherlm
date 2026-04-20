from __future__ import annotations

import logging
import uuid
from typing import Any

from arq.connections import RedisSettings
from sqlalchemy import select

from config import get_settings
from db.models import UploadedFile
from db.session import session_scope
from services.chunking_service import get_chunker
from services.parsing_service import get_parser
from services.storage_service import get_storage
from services.vector_service import get_vector_service


logger = logging.getLogger(__name__)


async def _set_status(
    file_pk: uuid.UUID,
    status: str,
    *,
    error: str | None = None,
    chunk_count: int | None = None,
    parsed_markdown_path: str | None = None,
) -> None:
    async with session_scope() as session:
        record = await session.get(UploadedFile, file_pk)
        if record is None:
            return
        record.status = status
        if error is not None:
            record.error = error
        if chunk_count is not None:
            record.chunk_count = chunk_count
        if parsed_markdown_path is not None:
            record.parsed_markdown_path = parsed_markdown_path


async def ingest_file(ctx: dict[str, Any], file_pk: str) -> dict[str, Any]:
    """arq job: parse → chunk → embed → upsert for one UploadedFile row."""
    pk = uuid.UUID(file_pk)

    async with session_scope() as session:
        result = await session.execute(select(UploadedFile).where(UploadedFile.id == pk))
        record = result.scalar_one_or_none()
        if record is None:
            logger.warning("ingest_file: UploadedFile %s not found", pk)
            return {"ok": False, "error": "not_found"}
        conversation_id = record.conversation_id
        filename = record.filename
        object_key = record.file_id

    storage = get_storage()
    parser = get_parser()
    chunker = get_chunker()
    vectors = get_vector_service()

    try:
        # --- parsing ---
        await _set_status(pk, "parsing")
        data = await storage.get_bytes(object_key)
        parse_result = await parser.parse_to_markdown(
            conversation_id=conversation_id,
            filename=filename,
            data=data,
        )
        await _set_status(pk, "chunking", parsed_markdown_path=parse_result.markdown_key)

        # --- chunking ---
        chunks = chunker.chunk_text(parse_result.markdown, source=filename)
        if not chunks:
            await _set_status(pk, "ready", chunk_count=0)
            return {"ok": True, "chunks": 0}

        # --- embedding + upsert ---
        await _set_status(pk, "embedding")
        await vectors.ensure_collection(conversation_id)
        upserted = await vectors.upsert_chunks(conversation_id, chunks, file_id=object_key)

        await _set_status(pk, "ready", chunk_count=upserted)
        return {"ok": True, "chunks": upserted, "markdown_key": parse_result.markdown_key}

    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        logger.exception("ingest_file failed for %s", pk)
        await _set_status(pk, "failed", error=f"{type(exc).__name__}: {exc}")
        return {"ok": False, "error": str(exc)}


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    logger.info("ingestion worker starting, env=%s", settings.environment)
    # Warm the MinIO bucket so the first upload doesn't race the check.
    await get_storage().ensure_bucket()


async def shutdown(ctx: dict[str, Any]) -> None:
    await get_vector_service().aclose()


class WorkerSettings:
    functions = [ingest_file]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
