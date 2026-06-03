from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from teacherlm_core.llm.language import reset_current_language, set_current_language

from db.models import Conversation
from db.session import get_db
from schemas.knowledge_check import (
    KnowledgeCheckStartRequest,
    KnowledgeCheckStartResponse,
    KnowledgeCheckSubmitRequest,
    KnowledgeCheckSubmitResponse,
    QuizAttemptRequest,
    QuizAttemptResponse,
)
from services.knowledge_assessment_service import get_knowledge_assessment_service
from services.runtime_settings_service import get_runtime_settings_service


router = APIRouter(prefix="/api/conversations", tags=["knowledge-checks"])


@router.post("/{conversation_id}/knowledge-checks/start", response_model=KnowledgeCheckStartResponse)
async def start_knowledge_checks(
    conversation_id: uuid.UUID,
    body: KnowledgeCheckStartRequest,
    session: AsyncSession = Depends(get_db),
) -> KnowledgeCheckStartResponse:
    await _require_conversation(session, conversation_id)
    service = get_knowledge_assessment_service()
    resolved_options = await get_runtime_settings_service().resolve_options(session, body.options)
    token = set_current_language(_language_from_options(resolved_options))
    try:
        response = await service.start_checks(
            session,
            conversation_id,
            concept_id=body.concept_id,
            phase_id=body.phase_id,
            objective_id=body.objective_id,
            count=body.count,
            question_types=body.question_types,
            llm_options=resolved_options,
        )
    finally:
        reset_current_language(token)
    if not response.checks:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No course concepts are available for knowledge checks yet.",
        )
    return response


@router.post(
    "/{conversation_id}/knowledge-checks/{check_id}/submit",
    response_model=KnowledgeCheckSubmitResponse,
)
async def submit_knowledge_check(
    conversation_id: uuid.UUID,
    check_id: uuid.UUID,
    body: KnowledgeCheckSubmitRequest,
    session: AsyncSession = Depends(get_db),
) -> KnowledgeCheckSubmitResponse:
    await _require_conversation(session, conversation_id)
    service = get_knowledge_assessment_service()
    resolved_options = await get_runtime_settings_service().resolve_options(session, body.options)
    token = set_current_language(_language_from_options(resolved_options))
    try:
        result, learner_state = await service.submit_check(
            session,
            conversation_id,
            check_id,
            body.answer,
            llm_options=resolved_options,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        reset_current_language(token)
    return KnowledgeCheckSubmitResponse(result=result, learner_state=learner_state)


@router.post("/{conversation_id}/quiz-attempts", response_model=QuizAttemptResponse)
async def submit_quiz_attempt(
    conversation_id: uuid.UUID,
    body: QuizAttemptRequest,
    session: AsyncSession = Depends(get_db),
) -> QuizAttemptResponse:
    await _require_conversation(session, conversation_id)
    answers = {item.question_index: item.answer for item in body.answers}
    resolved_options = await get_runtime_settings_service().resolve_options(session, body.options)
    token = set_current_language(_language_from_options(resolved_options))
    try:
        return await get_knowledge_assessment_service().submit_quiz_attempt(
            session,
            conversation_id,
            body.questions,
            answers,
            llm_options=resolved_options,
        )
    finally:
        reset_current_language(token)


async def _require_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> None:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")


def _language_from_options(options: dict | None) -> str | None:
    if not isinstance(options, dict):
        return None
    language = str(options.get("language") or "").strip().lower()
    return language if language and language not in {"auto", "__auto__"} else None
