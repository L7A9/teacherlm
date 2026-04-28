from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Conversation, Message
from db.session import get_db
from schemas.conversation import (
    ConversationCreate,
    ConversationList,
    ConversationRead,
    ConversationUpdate,
)
from schemas.message import MessageList, MessageRead
from services.storage_service import get_storage
from services.vector_service import get_vector_service


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.post("", response_model=ConversationRead, status_code=status.HTTP_201_CREATED)
async def create_conversation(
    body: ConversationCreate,
    session: AsyncSession = Depends(get_db),
) -> Conversation:
    conversation = Conversation(title=body.title or "Untitled conversation")
    session.add(conversation)
    await session.flush()
    await session.refresh(conversation)
    return conversation


@router.get("", response_model=ConversationList)
async def list_conversations(
    session: AsyncSession = Depends(get_db),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> ConversationList:
    total = await session.scalar(select(func.count()).select_from(Conversation))
    result = await session.execute(
        select(Conversation)
        .order_by(Conversation.updated_at.desc())
        .limit(limit)
        .offset(offset)
    )
    items = [ConversationRead.model_validate(c) for c in result.scalars().all()]
    return ConversationList(items=items, total=int(total or 0))


@router.get("/{conversation_id}", response_model=ConversationRead)
async def get_conversation(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> Conversation:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conversation


@router.patch("/{conversation_id}", response_model=ConversationRead)
async def update_conversation(
    conversation_id: uuid.UUID,
    body: ConversationUpdate,
    session: AsyncSession = Depends(get_db),
) -> Conversation:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    conversation.title = body.title
    await session.flush()
    await session.refresh(conversation)
    return conversation


@router.get("/{conversation_id}/messages", response_model=MessageList)
async def list_messages(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
    limit: int = Query(default=200, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> MessageList:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    total = await session.scalar(
        select(func.count())
        .select_from(Message)
        .where(Message.conversation_id == conversation_id)
    )
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.asc(), Message.id.asc())
        .limit(limit)
        .offset(offset)
    )
    items = [MessageRead.model_validate(m) for m in result.scalars().all()]
    await _refresh_artifact_urls(items)
    return MessageList(items=items, total=int(total or 0))


async def _refresh_artifact_urls(messages: list[MessageRead]) -> None:
    """Re-sign MinIO presigned URLs for any artifact whose object key we know.

    Generators store the object key alongside the URL (see GeneratorArtifact);
    presigned URLs expire (default 1h) but the object survives container
    restarts, so we mint a fresh URL each time history is loaded.
    Artifacts without a key (older rows, or non-MinIO artifacts like the
    mindmap generator's static URLs) pass through untouched.
    """
    storage = get_storage()
    for message in messages:
        for artifact in message.artifacts:
            if artifact.key:
                try:
                    artifact.url = await storage.presigned_get_url(artifact.key)
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "failed to refresh artifact URL for key=%s", artifact.key
                    )


@router.delete("/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_conversation(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> None:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    await session.delete(conversation)
    await session.flush()
    # Drop the per-conversation Qdrant collection so embeddings don't linger.
    await get_vector_service().delete_collection(conversation_id)
