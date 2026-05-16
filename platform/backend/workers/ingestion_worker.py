from __future__ import annotations

import logging
import uuid
from typing import Any

from arq.connections import RedisSettings
from sqlalchemy import select

from config import Settings, get_settings
from db.models import UploadedFile
from db.session import session_scope
from services.chunking_service import get_chunker
from services.course_content_store import get_course_content_store
from services.course_structure_service import get_course_structure_extractor
from services.document_cleaning_service import get_document_cleaner
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
        if status != "failed":
            record.error = None
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
    cleaner = get_document_cleaner()
    extractor = get_course_structure_extractor()
    chunker = get_chunker()
    content_store = get_course_content_store()
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

        # --- cleaning, structure extraction, and chunking ---
        cleaned_markdown = cleaner.clean_markdown(parse_result.markdown)
        cleaned_key = storage.cleaned_text_key(conversation_id, object_key)
        await storage.put_text(cleaned_key, cleaned_markdown)
        course_document = extractor.extract(
            cleaned_markdown,
            conversation_id=conversation_id,
            source_file_id=object_key,
            source_filename=filename,
        )
        chunks = chunker.chunk_course_document(course_document, source_file_id=object_key)

        async with session_scope() as session:
            await content_store.replace_document(
                session,
                conversation_id=conversation_id,
                uploaded_file_id=pk,
                source_file_id=object_key,
                source_filename=filename,
                raw_markdown_path=parse_result.markdown_key,
                cleaned_text_path=cleaned_key,
                cleaned_text=cleaned_markdown,
                document=course_document,
                chunks=chunks,
            )

        await vectors.delete_by_file(conversation_id, object_key)
        if not chunks:
            await _set_status(pk, "ready", chunk_count=0)
            return {"ok": True, "chunks": 0, "cleaned_key": cleaned_key}

        # --- embedding + upsert ---
        await _set_status(pk, "embedding")
        logger.info("embedding %s chunks for %s", len(chunks), filename)
        await vectors.ensure_collection(conversation_id)
        upserted = await vectors.upsert_chunks(conversation_id, chunks, file_id=object_key)
        logger.info("embedded and upserted %s chunks for %s", upserted, filename)

        await _set_status(pk, "ready", chunk_count=upserted)
        return {
            "ok": True,
            "chunks": upserted,
            "sections": len(course_document.sections),
            "markdown_key": parse_result.markdown_key,
            "cleaned_key": cleaned_key,
        }

    except Exception as exc:  # noqa: BLE001 — surface any failure to the UI
        logger.exception("ingest_file failed for %s", pk)
        await _set_status(pk, "failed", error=f"{type(exc).__name__}: {exc}")
        return {"ok": False, "error": str(exc)}


async def startup(ctx: dict[str, Any]) -> None:
    settings = get_settings()
    logger.info(
        "ingestion worker starting, env=%s, max_jobs=%s",
        settings.environment,
        settings.ingestion_max_jobs,
    )
    # Warm the MinIO bucket so the first upload doesn't race the check.
    await get_storage().ensure_bucket()
    await _recover_interrupted_uploads(ctx, settings)


async def _recover_interrupted_uploads(ctx: dict[str, Any], settings: Settings) -> None:
    """Requeue files left mid-ingestion by a worker crash/restart."""

    redis = ctx.get("redis")
    if redis is None:
        logger.warning("cannot recover interrupted uploads: arq redis handle missing")
        return

    interrupted_statuses = ("parsing", "chunking", "embedding")
    async with session_scope() as session:
        result = await session.execute(
            select(UploadedFile)
            .where(UploadedFile.status.in_(interrupted_statuses))
            .order_by(UploadedFile.created_at.asc())
        )
        records = list(result.scalars().all())
        for record in records:
            previous_status = record.status
            record.status = "uploaded"
            record.error = f"requeued after worker restart from status={previous_status}"

    if not records:
        return

    for record in records:
        await redis.enqueue_job("ingest_file", str(record.id))
    logger.warning("requeued %s interrupted ingestion jobs", len(records))


async def shutdown(ctx: dict[str, Any]) -> None:
    await get_vector_service().aclose()


class WorkerSettings:
    functions = [ingest_file]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = get_settings().ingestion_max_jobs
    job_timeout = get_settings().ingestion_job_timeout_s
