from __future__ import annotations

import json
import logging
from typing import Any
import uuid

from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Conversation, CourseDocumentRecord, UploadedFile as UploadedFileModel
from db.session import get_db
from schemas.file import UploadedFileList, UploadedFileRead
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


@router.delete("/{file_pk}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(
    conversation_id: uuid.UUID,
    file_pk: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> None:
    record = await _load_file(session, conversation_id, file_pk)

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
    if parsed is None:
        return None
    if not isinstance(parsed, dict):
        raise HTTPException(status_code=400, detail="llm_options_json must be an object")
    llm = parsed.get("llm")
    if llm is not None and not isinstance(llm, dict):
        raise HTTPException(status_code=400, detail="llm option must be an object")
    return parsed
