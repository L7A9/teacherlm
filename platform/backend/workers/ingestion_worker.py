from __future__ import annotations

import logging
import uuid
from typing import Any

from arq.connections import RedisSettings
from sqlalchemy import select
from teacherlm_core.llm.language import reset_current_language, set_current_language

from config import Settings, get_settings
from db.models import UploadedFile
from db.session import session_scope
from services.chunking_service import get_chunker
from services.chunk_question_generator import get_chunk_question_generator
from services.concept_inventory_service import get_concept_inventory_service
from services.course_intake_normalizer import get_course_intake_normalizer
from services.coursebuilder_jobs import coursebuilder_job_id
from services.coursebuilder_service import get_coursebuilder_service
from services.course_content_store import get_course_content_store
from services.course_structure_service import get_course_structure_extractor
from services.document_cleaning_service import get_document_cleaner
from services.learning_map_service import get_learning_map_service
from services.knowledge_graph_service import get_knowledge_graph_service
from services.parsing_service import get_parser
from services.runtime_settings_service import get_runtime_settings_service
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


async def _all_conversation_files_ready(conversation_id: uuid.UUID) -> bool:
    async with session_scope() as session:
        result = await session.execute(
            select(UploadedFile).where(UploadedFile.conversation_id == conversation_id)
        )
        files = list(result.scalars().all())
    return bool(files) and all(file.status == "ready" for file in files)


async def _rebuild_learning_course_if_ready(
    ctx: dict[str, Any],
    conversation_id: uuid.UUID,
    *,
    llm_options: dict[str, Any] | None = None,
) -> None:
    if not await _all_conversation_files_ready(conversation_id):
        return

    runtime_settings = get_runtime_settings_service()
    queue_options = runtime_settings.sanitize_client_options(llm_options)
    async with session_scope() as session:
        resolved_options = await runtime_settings.resolve_options(session, queue_options)
        language_token = set_current_language(_language_from_options(resolved_options))
        try:
            await get_concept_inventory_service().rebuild_concepts(
                session,
                conversation_id,
                llm_options=resolved_options,
            )
            await get_learning_map_service().rebuild_map(
                session,
                conversation_id,
                llm_options=resolved_options,
            )
            await get_knowledge_graph_service().rebuild_graph(
                session,
                conversation_id,
                llm_options=resolved_options,
            )
            service = get_coursebuilder_service()
            existing = await service.current_course(session, conversation_id)
            if existing and existing.status in {
                "queued",
                "analyzing",
                "generating_outline",
                "generating_chapters",
                "generating_lessons",
                "generating_quizzes",
                "validating",
            }:
                return
            redis = ctx.get("redis")
            if redis is not None:
                course = await service.queue_course(session, conversation_id, llm_options=resolved_options)
                generation_id = _coursebuilder_generation_id(course.generation_metadata)
                await session.commit()
                try:
                    await redis.enqueue_job(
                        "build_coursebuilder_course",
                        str(conversation_id),
                        queue_options,
                        generation_id,
                        _job_id=coursebuilder_job_id(conversation_id, generation_id),
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("coursebuilder enqueue failed; building inline")
                    await service.generate_course(
                        session,
                        conversation_id,
                        llm_options=resolved_options,
                        course=course,
                    )
            else:
                await service.queue_course(session, conversation_id, llm_options=resolved_options)
                logger.warning("coursebuilder queue unavailable; building inline")
                await service.generate_course(session, conversation_id, llm_options=resolved_options)
        finally:
            reset_current_language(language_token)


def _language_from_options(llm_options: dict[str, Any] | None) -> str | None:
    if not isinstance(llm_options, dict):
        return None
    language = str(llm_options.get("language") or "").strip().lower()
    return language if language and language not in {"auto", "__auto__"} else None


async def build_coursebuilder_course(
    ctx: dict[str, Any],
    conversation_id: str,
    llm_options: dict[str, Any] | None = None,
    generation_id: str | None = None,
) -> dict[str, Any]:
    conversation_uuid = uuid.UUID(conversation_id)
    runtime_settings = get_runtime_settings_service()
    queue_options = runtime_settings.sanitize_client_options(llm_options)
    async with session_scope() as session:
        resolved_options = await runtime_settings.resolve_options(session, queue_options)
        language_token = set_current_language(_language_from_options(resolved_options))
        try:
            service = get_coursebuilder_service()
            course_record = await service.current_course(session, conversation_uuid)
            if generation_id and course_record is not None:
                current_generation_id = _coursebuilder_generation_id(course_record.generation_metadata)
                if current_generation_id != generation_id:
                    return {
                        "ok": False,
                        "status": course_record.status,
                        "stale": True,
                        "course_id": str(course_record.id),
                    }
            course = await service.generate_course(
                session,
                conversation_uuid,
                llm_options=resolved_options,
                course=course_record,
            )
            return {"ok": course.status == "ready", "status": course.status, "course_id": str(course.id) if course.id else None}
        finally:
            reset_current_language(language_token)


async def ingest_file(
    ctx: dict[str, Any],
    file_pk: str,
    llm_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """arq job: parse → chunk → embed → upsert for one UploadedFile row."""
    pk = uuid.UUID(file_pk)
    runtime_settings = get_runtime_settings_service()
    queue_options = runtime_settings.sanitize_client_options(llm_options)

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
    normalizer = get_course_intake_normalizer()
    extractor = get_course_structure_extractor()
    chunker = get_chunker()
    question_generator = get_chunk_question_generator()
    content_store = get_course_content_store()
    vectors = get_vector_service()

    try:
        # --- parsing ---
        await _set_status(pk, "parsing")
        data = await storage.get_bytes(object_key)
        async with session_scope() as session:
            parser_api_key = await runtime_settings.parser_api_key(session)
        parse_result = await parser.parse_to_markdown(
            conversation_id=conversation_id,
            filename=filename,
            data=data,
            api_key=parser_api_key,
        )
        await _set_status(pk, "chunking", parsed_markdown_path=parse_result.markdown_key)

        # --- cleaning, structure extraction, and chunking ---
        cleaned_markdown = cleaner.clean_markdown(parse_result.markdown)
        normalized_intake = normalizer.normalize(
            raw_markdown=parse_result.markdown,
            cleaned_markdown=cleaned_markdown,
            source_filename=filename,
        )
        cleaned_markdown = normalized_intake.markdown
        cleaned_key = storage.cleaned_text_key(conversation_id, object_key)
        await storage.put_text(cleaned_key, cleaned_markdown)
        course_document = extractor.extract(
            cleaned_markdown,
            conversation_id=conversation_id,
            source_file_id=object_key,
            source_filename=filename,
            intake_metadata=normalized_intake.metadata,
            infer_plain_headings=not normalized_intake.normalized,
        )
        chunks = chunker.chunk_course_document(course_document, source_file_id=object_key)
        async with session_scope() as session:
            ingestion_options = await runtime_settings.resolve_options(session, queue_options)
        chunks = await question_generator.annotate_chunks(chunks, llm_options=ingestion_options)

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
            try:
                await _rebuild_learning_course_if_ready(ctx, conversation_id, llm_options=queue_options)
            except Exception:  # noqa: BLE001
                logger.exception("final course rebuild failed for %s; file remains ready", filename)
            return {"ok": True, "chunks": 0, "cleaned_key": cleaned_key}

        # --- embedding + upsert ---
        await _set_status(pk, "embedding")
        logger.info("embedding %s chunks for %s", len(chunks), filename)
        await vectors.ensure_collection(conversation_id)
        upserted = await vectors.upsert_chunks(conversation_id, chunks, file_id=object_key)
        logger.info("embedded and upserted %s chunks for %s", upserted, filename)

        await _set_status(pk, "ready", chunk_count=upserted)
        try:
            await _rebuild_learning_course_if_ready(ctx, conversation_id, llm_options=queue_options)
        except Exception:  # noqa: BLE001
            logger.exception("final course rebuild failed for %s; file remains ready", filename)
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
    await _recover_coursebuilder_jobs(ctx)


async def _recover_interrupted_uploads(ctx: dict[str, Any], settings: Settings) -> None:
    """Requeue files left mid-ingestion by a worker crash/restart."""

    redis = ctx.get("redis")
    if redis is None:
        logger.warning("cannot recover interrupted uploads: arq redis handle missing")
        return

    interrupted_statuses = ("parsing", "chunking", "extracting_concepts", "building_course", "embedding")
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


async def _recover_coursebuilder_jobs(ctx: dict[str, Any]) -> None:
    redis = ctx.get("redis")
    if redis is None:
        return
    running_statuses = (
        "queued",
        "analyzing",
        "generating_outline",
        "generating_chapters",
        "generating_lessons",
        "generating_quizzes",
        "validating",
    )
    try:
        from db.models import CourseBuilderCourseRecord

        async with session_scope() as session:
            service = get_coursebuilder_service()
            runtime_settings = get_runtime_settings_service()
            await service.ensure_schema(session)
            result = await session.execute(
                select(CourseBuilderCourseRecord)
                .where(CourseBuilderCourseRecord.status.in_(running_statuses))
                .order_by(CourseBuilderCourseRecord.created_at.asc())
            )
            courses = list(result.scalars().all())
            jobs: list[tuple[str, dict[str, Any], str | None]] = []
            for course in courses:
                if await _all_conversation_files_ready(course.conversation_id):
                    course.status = "queued"
                    generation_id = _coursebuilder_generation_id(course.generation_metadata)
                    jobs.append(
                        (
                            str(course.conversation_id),
                            runtime_settings.sanitize_client_options(
                                (course.generation_metadata or {}).get("llm_options")
                            ),
                            generation_id,
                        )
                    )
            await session.commit()
            for conversation_id, llm_options, generation_id in jobs:
                try:
                    await redis.enqueue_job(
                        "build_coursebuilder_course",
                        conversation_id,
                        llm_options,
                        generation_id,
                        _job_id=coursebuilder_job_id(conversation_id, generation_id),
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.exception("recovered CourseBuilder job enqueue failed")
                    async with session_scope() as fail_session:
                        await get_coursebuilder_service().mark_course_failed(
                            fail_session,
                            uuid.UUID(conversation_id),
                            f"CourseBuilder recovery queue failed: {exc}",
                        )
        if jobs:
            logger.warning("requeued %s interrupted CourseBuilder jobs", len(jobs))
    except Exception:  # noqa: BLE001
        logger.exception("CourseBuilder job recovery failed")


def _coursebuilder_generation_id(metadata: dict[str, Any] | None) -> str | None:
    value = (metadata or {}).get("generation_id")
    return str(value) if value else None


async def shutdown(ctx: dict[str, Any]) -> None:
    await get_vector_service().aclose()


class WorkerSettings:
    functions = [ingest_file, build_coursebuilder_course]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_jobs = get_settings().ingestion_max_jobs
    job_timeout = get_settings().ingestion_job_timeout_s
