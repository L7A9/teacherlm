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
from services.coursebuilder_jobs import coursebuilder_job_id
from services.coursebuilder_service import get_coursebuilder_service
from services.runtime_settings_service import get_runtime_settings_service


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
    _require_queue(request)
    runtime_settings = get_runtime_settings_service()
    queue_options = runtime_settings.sanitize_client_options(body.options)
    resolved_options = await runtime_settings.resolve_options(session, queue_options)
    course = await service.queue_course(
        session,
        conversation_id,
        llm_options=resolved_options,
        restart_queued=True,
    )
    generation_id = _generation_id(course.generation_metadata)
    await session.commit()
    try:
        await _enqueue(request, conversation_id, queue_options, generation_id)
    except Exception as exc:  # noqa: BLE001
        await service.mark_course_failed(
            session,
            conversation_id,
            f"Course generation queue failed: {exc}",
            course_id=course.id,
        )
        await session.commit()
        raise HTTPException(status_code=503, detail="course generation queue failed") from exc
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
    _require_queue(request)
    runtime_settings = get_runtime_settings_service()
    queue_options = runtime_settings.sanitize_client_options(body.options)
    resolved_options = await runtime_settings.resolve_options(session, queue_options)
    course = await service.queue_course(
        session,
        conversation_id,
        llm_options=resolved_options,
        restart_queued=True,
    )
    generation_id = _generation_id(course.generation_metadata)
    await session.commit()
    try:
        await _enqueue(request, conversation_id, queue_options, generation_id)
    except Exception as exc:  # noqa: BLE001
        await service.mark_course_failed(
            session,
            conversation_id,
            f"Course rebuild queue failed: {exc}",
            course_id=course.id,
        )
        await session.commit()
        raise HTTPException(status_code=503, detail="course rebuild queue failed") from exc
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
    last_cursor: tuple[str, str, str, str] | None = None
    while True:
        async with session_scope() as session:
            course = await get_coursebuilder_service().get_course(session, conversation_id)
        payload = course.model_dump(mode="json")
        cursor = _snapshot_cursor(course)
        if cursor != last_cursor:
            yield {"event": "snapshot", "data": json.dumps(payload, default=str)}
            last_cursor = cursor
        if course.status in {"ready", "failed"}:
            return
        await asyncio.sleep(1.0)


async def _enqueue(
    request: Request,
    conversation_id: uuid.UUID,
    options: dict[str, Any] | None,
    generation_id: str | None,
) -> None:
    arq = getattr(request.app.state, "arq_pool", None)
    if arq is None:
        raise HTTPException(status_code=503, detail="course generation queue not initialized")
    await arq.enqueue_job(
        "build_coursebuilder_course",
        str(conversation_id),
        options or {},
        generation_id,
        _job_id=coursebuilder_job_id(conversation_id, generation_id),
    )


def _require_queue(request: Request) -> None:
    if getattr(request.app.state, "arq_pool", None) is None:
        raise HTTPException(status_code=503, detail="course generation queue not initialized")


def _snapshot_cursor(course: CourseBuilderRead) -> tuple[str, str, str, str]:
    latest_event_id = str(course.progress_events[-1].id) if course.progress_events else ""
    generation_id = str((course.generation_metadata or {}).get("generation_id") or "")
    return (str(course.id or ""), str(course.status), generation_id, latest_event_id)


def _generation_id(metadata: dict[str, Any] | None) -> str | None:
    value = (metadata or {}).get("generation_id")
    return str(value) if value else None


async def _require_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> None:
    if await session.get(Conversation, conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
