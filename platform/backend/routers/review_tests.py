from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from teacherlm_core.llm.language import reset_current_language, set_current_language

from db.models import Conversation
from db.session import get_db
from schemas.review_test import (
    ReviewTestActionResponse,
    ReviewTestStartRequest,
    ReviewTestStartResponse,
    ReviewTestStatusResponse,
    ReviewTestSubmitRequest,
    ReviewTestSubmitResponse,
)
from services.review_test_service import get_review_test_service
from services.runtime_settings_service import get_runtime_settings_service


router = APIRouter(prefix="/api/conversations", tags=["review-tests"])


@router.get("/{conversation_id}/review-tests/status", response_model=ReviewTestStatusResponse)
async def review_test_status(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> ReviewTestStatusResponse:
    await _require_conversation(session, conversation_id)
    return await get_review_test_service().status(session, conversation_id)


@router.post("/{conversation_id}/review-tests/start", response_model=ReviewTestStartResponse)
async def start_review_test(
    conversation_id: uuid.UUID,
    body: ReviewTestStartRequest,
    session: AsyncSession = Depends(get_db),
) -> ReviewTestStartResponse:
    await _require_conversation(session, conversation_id)
    resolved_options = await get_runtime_settings_service().resolve_options(session, body.options)
    token = set_current_language(_language_from_options(resolved_options))
    try:
        response = await get_review_test_service().start_review(
            session,
            conversation_id,
            llm_options=resolved_options,
        )
    except LookupError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    finally:
        reset_current_language(token)
    if not response.checks:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No review questions could be generated from the recent discussion.",
        )
    return response


@router.post(
    "/{conversation_id}/review-tests/{window_id}/submit",
    response_model=ReviewTestSubmitResponse,
)
async def submit_review_test(
    conversation_id: uuid.UUID,
    window_id: uuid.UUID,
    body: ReviewTestSubmitRequest,
    session: AsyncSession = Depends(get_db),
) -> ReviewTestSubmitResponse:
    await _require_conversation(session, conversation_id)
    resolved_options = await get_runtime_settings_service().resolve_options(session, body.options)
    token = set_current_language(_language_from_options(resolved_options))
    try:
        return await get_review_test_service().submit_review(
            session,
            conversation_id,
            window_id,
            {item.check_id: item.answer for item in body.answers},
            llm_options=resolved_options,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    finally:
        reset_current_language(token)


@router.post(
    "/{conversation_id}/review-tests/{window_id}/snooze",
    response_model=ReviewTestActionResponse,
)
async def snooze_review_test(
    conversation_id: uuid.UUID,
    window_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> ReviewTestActionResponse:
    await _require_conversation(session, conversation_id)
    try:
        return await get_review_test_service().snooze(session, conversation_id, window_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post(
    "/{conversation_id}/review-tests/{window_id}/dismiss",
    response_model=ReviewTestActionResponse,
)
async def dismiss_review_test(
    conversation_id: uuid.UUID,
    window_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> ReviewTestActionResponse:
    await _require_conversation(session, conversation_id)
    try:
        return await get_review_test_service().dismiss(session, conversation_id, window_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


async def _require_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> None:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")


def _language_from_options(options: dict | None) -> str | None:
    if not isinstance(options, dict):
        return None
    language = str(options.get("language") or "").strip().lower()
    return language if language and language not in {"auto", "__auto__"} else None
