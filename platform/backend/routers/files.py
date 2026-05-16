from __future__ import annotations

import uuid

from arq import ArqRedis
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Conversation, CourseDocumentRecord, UploadedFile as UploadedFileModel
from db.session import get_db
from schemas.file import UploadedFileList, UploadedFileRead
from services.storage_service import get_storage
from services.vector_service import get_vector_service


router = APIRouter(prefix="/api/conversations/{conversation_id}/files", tags=["files"])


def get_arq(request: Request) -> ArqRedis:
    pool = getattr(request.app.state, "arq_pool", None)
    if pool is None:
        raise HTTPException(status_code=503, detail="ingestion queue not initialized")
    return pool


@router.post("", response_model=UploadedFileRead, status_code=status.HTTP_201_CREATED)
async def upload_file(
    conversation_id: uuid.UUID,
    upload: UploadFile = File(...),
    session: AsyncSession = Depends(get_db),
    arq: ArqRedis = Depends(get_arq),
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

    await arq.enqueue_job("ingest_file", str(record.id))
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


async def _load_file(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    file_pk: uuid.UUID,
) -> UploadedFileModel:
    record = await session.get(UploadedFileModel, file_pk)
    if record is None or record.conversation_id != conversation_id:
        raise HTTPException(status_code=404, detail="file not found")
    return record
