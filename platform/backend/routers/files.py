from __future__ import annotations

import json
import logging
from typing import Any
import uuid

from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Conversation, CourseDocumentRecord, UploadedFile as UploadedFileModel
from db.session import get_db
from schemas.file import FileRetryRequest, UploadedFileList, UploadedFileRead
from services.coursebuilder_jobs import coursebuilder_job_id
from services.coursebuilder_service import get_coursebuilder_service
from services.runtime_settings_service import get_runtime_settings_service
from services.storage_service import get_storage


router = APIRouter(prefix="/api/conversations/{conversation_id}/files", tags=["files"])
logger = logging.getLogger(__name__)


def get_arq(request: Request) -> Any:
    pool = getattr(request.app.state, "arq_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="ingestion queue not initialized")
    return pool


@router.post("", response_model=UploadedFileRead, status_code=status.HTTP_201_CREATED)
async def upload_file(
    conversation_id: uuid.UUID,
    upload: UploadFile = File(...),
    llm_options_json: str | None = Form(default=None),
    session: AsyncSession = Depends(get_db),
    arq: Any = Depends(get_arq),
) -> UploadedFileModel:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    if not upload.filename:
        raise HTTPException(status_code=400, detail="filename is required")

    storage = get_storage()
    await storage.ensure_bucket()
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty upload")

    object_key = storage.upload_key(conversation_id, upload.filename)
    await storage.put_bytes(
        object_key,
        data,
        content_type=upload.content_type or "application/octet-stream",
    )

    record = UploadedFileModel(
        conversation_id=conversation_id,
        filename=upload.filename,
        file_id=object_key,
        status="uploaded",
    )
    session.add(record)
    await session.flush()
    await session.refresh(record)
    await session.commit()

    await arq.enqueue_job("ingest_file", str(record.id), _parse_llm_options(llm_options_json))
    return record


@router.get("", response_model=UploadedFileList)
async def list_files(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> UploadedFileList:
    total = await session.scalar(
        select(func.count())
        .select_from(UploadedFileModel)
        .where(UploadedFileModel.conversation_id == conversation_id)
    )
    result = await session.execute(
        select(UploadedFileModel)
        .where(UploadedFileModel.conversation_id == conversation_id)
        .order_by(UploadedFileModel.created_at.desc())
    )
    items = [UploadedFileRead.model_validate(f) for f in result.scalars().all()]
    return UploadedFileList(items=items, total=int(total or 0))


@router.get("/{file_pk}", response_model=UploadedFileRead)
async def get_file(
    conversation_id: uuid.UUID,
    file_pk: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> UploadedFileModel:
    record = await _load_file(session, conversation_id, file_pk)
    return record


@router.post("/{file_pk}/retry", response_model=UploadedFileRead)
async def retry_file(
    conversation_id: uuid.UUID,
    file_pk: uuid.UUID,
    body: FileRetryRequest | None = Body(default=None),
    session: AsyncSession = Depends(get_db),
    arq: Any = Depends(get_arq),
) -> UploadedFileModel:
    record = await _load_file(session, conversation_id, file_pk)
    if record.status != "failed":
        raise HTTPException(status_code=409, detail="only failed files can be retried")

    storage = get_storage()
    from services.vector_service import get_vector_service

    vectors = get_vector_service()
    document_result = await session.execute(
        select(CourseDocumentRecord).where(CourseDocumentRecord.uploaded_file_id == record.id)
    )
    document = document_result.scalar_one_or_none()

    await vectors.delete_by_file(conversation_id, record.file_id)
    if record.parsed_markdown_path:
        await storage.delete(record.parsed_markdown_path)
    if document and document.cleaned_text_path:
        await storage.delete(document.cleaned_text_path)
    if document is not None:
        await session.delete(document)

    record.status = "uploaded"
    record.error = None
    record.chunk_count = 0
    record.parsed_markdown_path = None

    await session.flush()
    await session.refresh(record)
    await session.commit()
    await arq.enqueue_job("ingest_file", str(record.id), _validate_llm_options(body.options if body else None))
    return record


@router.delete("/{file_pk}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    conversation_id: uuid.UUID,
    file_pk: uuid.UUID,
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> None:
    record = await _load_file(session, conversation_id, file_pk)
    coursebuilder_service = get_coursebuilder_service()
    existing_course = await coursebuilder_service.current_course(session, conversation_id)
    coursebuilder_options = (
        (existing_course.generation_metadata or {}).get("llm_options")
        if existing_course is not None
        else None
    )

    storage = get_storage()
    from services.vector_service import get_vector_service

    vectors = get_vector_service()
    document_result = await session.execute(
        select(CourseDocumentRecord).where(CourseDocumentRecord.uploaded_file_id == record.id)
    )
    document = document_result.scalar_one_or_none()

    await vectors.delete_by_file(conversation_id, record.file_id)
    await storage.delete(record.file_id)
    if record.parsed_markdown_path:
        await storage.delete(record.parsed_markdown_path)
    if document and document.cleaned_text_path:
        await storage.delete(document.cleaned_text_path)

    await session.delete(record)
    await session.flush()

    from services.concept_inventory_service import get_concept_inventory_service
    from services.learning_map_service import get_learning_map_service

    try:
        await get_concept_inventory_service().rebuild_concepts(session, conversation_id)
        await get_learning_map_service().rebuild_map(session, conversation_id)
    except Exception:  # noqa: BLE001
        logger.exception("learning inventory rebuild failed after deleting file %s", file_pk)

    await _sync_coursebuilder_after_file_delete(
        session,
        conversation_id,
        request,
        llm_options=coursebuilder_options if isinstance(coursebuilder_options, dict) else None,
    )


async def _load_file(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    file_pk: uuid.UUID,
) -> UploadedFileModel:
    record = await session.get(UploadedFileModel, file_pk)
    if record is None or record.conversation_id != conversation_id:
        raise HTTPException(status_code=404, detail="file not found")
    return record


def _parse_llm_options(raw: str | None) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="invalid llm_options_json") from exc
    return _validate_llm_options(parsed)


def _validate_llm_options(parsed: Any) -> dict[str, Any] | None:
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="llm_options_json must be an object")
    return get_runtime_settings_service().sanitize_client_options(parsed)


async def _sync_coursebuilder_after_file_delete(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    request: Request,
    *,
    llm_options: dict[str, Any] | None,
) -> None:
    service = get_coursebuilder_service()
    result = await session.execute(
        select(UploadedFileModel).where(UploadedFileModel.conversation_id == conversation_id)
    )
    remaining_files = list(result.scalars().all())
    if not remaining_files:
        await service.clear_course(session, conversation_id)
        return

    if any(file.status != "ready" for file in remaining_files):
        return

    runtime_settings = get_runtime_settings_service()
    queue_options = runtime_settings.sanitize_client_options(llm_options)
    resolved_options = await runtime_settings.resolve_options(session, queue_options)
    course = await service.queue_course(
        session,
        conversation_id,
        llm_options=resolved_options,
        restart_queued=True,
    )
    generation_id = str((course.generation_metadata or {}).get("generation_id") or "") or None
    await session.commit()
    arq = getattr(request.app.state, "arq_pool", None)
    if arq is None:
        logger.warning("coursebuilder queue unavailable after file delete; rebuilding inline")
        await service.generate_course(
            session,
            conversation_id,
            llm_options=resolved_options,
            course=course,
        )
        return

    try:
        await arq.enqueue_job(
            "build_coursebuilder_course",
            str(conversation_id),
            queue_options,
            generation_id,
            _job_id=coursebuilder_job_id(conversation_id, generation_id),
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("coursebuilder enqueue failed after deleting file %s", conversation_id)
        await service.mark_course_failed(
            session,
            conversation_id,
            f"Course rebuild queue failed after deleting a file: {exc}",
            course_id=course.id,
        )
        await session.commit()
