from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from teacherlm_core.llm.language import reset_current_language, set_current_language

from db.models import Conversation, UploadedFile
from db.session import get_db
from schemas.course_player import (
    ChapterQuizSubmitRequest,
    ChapterQuizSubmitResponse,
    CoursePlayerRead,
    CoursePlayerUnlockResponse,
)
from services.concept_inventory_service import get_concept_inventory_service
from services.course_player_service import get_course_player_service
from services.knowledge_graph_service import get_knowledge_graph_service
from services.learner_tracker import get_learner_tracker
from services.learning_map_service import get_learning_map_service


router = APIRouter(prefix="/api/conversations", tags=["course-player"])


@router.get("/{conversation_id}/course-player", response_model=CoursePlayerRead)
async def get_course_player(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> CoursePlayerRead:
    await _require_conversation(session, conversation_id)
    waiting = await _waiting_for_files_response(session, conversation_id)
    if waiting is not None:
        return waiting

    service = get_course_player_service()
    course = await service.get_course(session, conversation_id)
    if not course.chapters:
        course = await _rebuild_full_course(session, conversation_id)
    return course


@router.post("/{conversation_id}/course-player/rebuild", response_model=CoursePlayerRead)
async def rebuild_course_player(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> CoursePlayerRead:
    await _require_conversation(session, conversation_id)
    waiting = await _waiting_for_files_response(session, conversation_id)
    if waiting is not None:
        return waiting
    return await _rebuild_full_course(session, conversation_id)


@router.post(
    "/{conversation_id}/course-player/chapters/{chapter_id}/unlock",
    response_model=CoursePlayerUnlockResponse,
)
async def unlock_course_chapter(
    conversation_id: uuid.UUID,
    chapter_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> CoursePlayerUnlockResponse:
    await _require_conversation(session, conversation_id)
    try:
        return await get_course_player_service().unlock_chapter(session, conversation_id, chapter_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/{conversation_id}/course-player/chapters/{chapter_id}/quiz/submit",
    response_model=ChapterQuizSubmitResponse,
)
async def submit_chapter_quiz(
    conversation_id: uuid.UUID,
    chapter_id: uuid.UUID,
    body: ChapterQuizSubmitRequest,
    session: AsyncSession = Depends(get_db),
) -> ChapterQuizSubmitResponse:
    await _require_conversation(session, conversation_id)
    token = set_current_language(_language_from_options(body.options))
    try:
        return await get_course_player_service().submit_chapter_quiz(
            session,
            conversation_id,
            chapter_id,
            {item.check_id: item.answer for item in body.answers},
            llm_options=body.options,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        reset_current_language(token)


async def _require_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> None:
    if await session.get(Conversation, conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")


async def _waiting_for_files_response(
    session: AsyncSession,
    conversation_id: uuid.UUID,
) -> CoursePlayerRead | None:
    result = await session.execute(
        select(UploadedFile).where(UploadedFile.conversation_id == conversation_id)
    )
    files = list(result.scalars().all())
    if not files:
        return None

    pending = [file for file in files if file.status != "ready"]
    if not pending:
        return None

    state = await get_learner_tracker().load_state(session, conversation_id)
    return CoursePlayerRead(
        conversation_id=conversation_id,
        chapters=[],
        learner_state=state,
        course_status="waiting_for_files",
        pending_file_count=len(pending),
        total_file_count=len(files),
    )


async def _rebuild_full_course(
    session: AsyncSession,
    conversation_id: uuid.UUID,
) -> CoursePlayerRead:
    await get_concept_inventory_service().rebuild_concepts(session, conversation_id)
    await get_learning_map_service().rebuild_map(session, conversation_id)
    await get_knowledge_graph_service().rebuild_graph(session, conversation_id)
    return await get_course_player_service().rebuild_course(session, conversation_id)


def _language_from_options(options: dict | None) -> str | None:
    if not isinstance(options, dict):
        return None
    language = str(options.get("language") or "").strip().lower()
    return language if language and language not in {"auto", "__auto__"} else None
