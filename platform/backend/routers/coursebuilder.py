from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from db.models import Conversation
from db.session import get_db, session_scope
from schemas.coursebuilder import (
    CourseBuilderGenerateRequest,
    CourseBuilderQuizSubmitRequest,
    CourseBuilderQuizSubmitResponse,
    CourseBuilderRead,
)
from services.coursebuilder_service import get_coursebuilder_service


router = APIRouter(prefix="/api/conversations", tags=["coursebuilder"])


@router.get("/{conversation_id}/coursebuilder", response_model=CourseBuilderRead)
async def get_coursebuilder(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> CourseBuilderRead:
    await _require_conversation(session, conversation_id)
    return await get_coursebuilder_service().get_course(session, conversation_id)


@router.post("/{conversation_id}/coursebuilder/generate", response_model=CourseBuilderRead)
async def generate_coursebuilder(
    conversation_id: uuid.UUID,
    body: CourseBuilderGenerateRequest,
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> CourseBuilderRead:
    await _require_conversation(session, conversation_id)
    service = get_coursebuilder_service()
    total, pending = await service.file_counts(session, conversation_id)
    if not total:
        raise HTTPException(status_code=409, detail="Upload course files before generating a course.")
    if pending:
        raise HTTPException(status_code=409, detail="Course generation waits until every uploaded file is ready.")
    course = await service.queue_course(session, conversation_id, llm_options=body.options)
    await _enqueue(request, conversation_id, body.options)
    await session.flush()
    return await service.get_course(session, conversation_id)


@router.post("/{conversation_id}/coursebuilder/rebuild", response_model=CourseBuilderRead)
async def rebuild_coursebuilder(
    conversation_id: uuid.UUID,
    body: CourseBuilderGenerateRequest,
    request: Request,
    session: AsyncSession = Depends(get_db),
) -> CourseBuilderRead:
    await _require_conversation(session, conversation_id)
    service = get_coursebuilder_service()
    total, pending = await service.file_counts(session, conversation_id)
    if not total:
        raise HTTPException(status_code=409, detail="Upload course files before rebuilding a course.")
    if pending:
        raise HTTPException(status_code=409, detail="Course rebuild waits until every uploaded file is ready.")
    course = await service.queue_course(session, conversation_id, llm_options=body.options)
    await _enqueue(request, conversation_id, body.options)
    await session.flush()
    return await service.get_course(session, conversation_id)


@router.get("/{conversation_id}/coursebuilder/events")
async def stream_coursebuilder_events(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> EventSourceResponse:
    await _require_conversation(session, conversation_id)
    return EventSourceResponse(
        _event_stream(conversation_id),
        media_type="text/event-stream",
    )


@router.post(
    "/{conversation_id}/coursebuilder/chapters/{chapter_id}/quiz/submit",
    response_model=CourseBuilderQuizSubmitResponse,
)
async def submit_coursebuilder_quiz(
    conversation_id: uuid.UUID,
    chapter_id: uuid.UUID,
    body: CourseBuilderQuizSubmitRequest,
    session: AsyncSession = Depends(get_db),
) -> CourseBuilderQuizSubmitResponse:
    await _require_conversation(session, conversation_id)
    try:
        return await get_coursebuilder_service().submit_quiz(
            session,
            conversation_id,
            chapter_id,
            {answer.question_id: answer.answer for answer in body.answers},
        )
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _event_stream(conversation_id: uuid.UUID) -> AsyncIterator[dict[str, str]]:
    last_event_count = -1
    while True:
        async with session_scope() as session:
            course = await get_coursebuilder_service().get_course(session, conversation_id)
        payload = course.model_dump(mode="json")
        event_count = len(course.progress_events)
        if event_count != last_event_count:
            yield {"event": "snapshot", "data": json.dumps(payload, default=str)}
            last_event_count = event_count
        if course.status in {"ready", "failed"}:
            return
        await asyncio.sleep(1.0)


async def _enqueue(
    request: Request,
    conversation_id: uuid.UUID,
    options: dict[str, Any] | None,
) -> None:
    arq = getattr(request.app.state, "arq_pool", None)
    if arq is None:
        raise HTTPException(status_code=503, detail="course generation queue not initialized")
    await arq.enqueue_job("build_coursebuilder_course", str(conversation_id), options or {})


async def _require_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> None:
    if await session.get(Conversation, conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
