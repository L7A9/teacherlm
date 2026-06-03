from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from teacherlm_core.schemas.chunk import Chunk
from teacherlm_core.schemas.generator_io import GeneratorInput, LearnerUpdates
from teacherlm_core.schemas.learner_state import LearnerState

from db.models import Conversation, Message
from db.session import get_db, session_scope
from dispatcher.registry import GeneratorEntry, GeneratorNotFound
from dispatcher.router import GeneratorRouter, get_router
from schemas.message import ChatRequest
from services.interaction_router import InteractionDecision, get_interaction_router
from services.review_test_service import get_review_test_service
from services.learner_tracker import get_learner_tracker
from services.runtime_settings_service import get_runtime_settings_service


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["chat"])


CHAT_HISTORY_LIMIT = 20


@router.post("/{conversation_id}/chat")
async def chat(
    conversation_id: uuid.UUID,
    body: ChatRequest,
    session: AsyncSession = Depends(get_db),
    grouter: GeneratorRouter = Depends(get_router),
) -> EventSourceResponse:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    try:
        entry = grouter.resolve_chat_default()
    except GeneratorNotFound as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    # Load prior history BEFORE appending this turn so the generator sees
    # `user_message` once (not duplicated in chat_history).
    chat_history = await _load_history(session, conversation_id, limit=CHAT_HISTORY_LIMIT)
    learner_state = await get_learner_tracker().load_state(session, conversation_id)
    learner_state_dict = learner_state.model_dump()
    request_options = await get_runtime_settings_service().resolve_options(session, body.options)

    route = await get_interaction_router().route(
        conversation_id=conversation_id,
        user_message=body.user_message,
        chat_history=chat_history,
        learner_state=learner_state_dict,
        options=request_options,
    )

    # Persist the user turn up-front so a failure mid-stream still leaves it.
    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=body.user_message,
    )
    session.add(user_message)
    await session.flush()
    user_message_id = user_message.id
    await session.commit()

    if route.action != "retrieve":
        return EventSourceResponse(
            _stream_direct_chat(conversation_id, route),
            media_type="text/event-stream",
        )

    from services.retrieval_orchestrator import get_retrieval_orchestrator

    retrieval_query = route.retrieval_query.strip() or body.user_message
    context_chunks = await get_retrieval_orchestrator().retrieve_for(
        output_type="text",
        query=retrieval_query,
        conversation_id=conversation_id,
    )

    payload = GeneratorInput(
        conversation_id=str(conversation_id),
        user_message=body.user_message,
        context_chunks=context_chunks,
        learner_state=learner_state,
        chat_history=chat_history,
        options=request_options,
    )

    return EventSourceResponse(
        _stream_chat(conversation_id, entry, payload, grouter, user_message_id=user_message_id),
        media_type="text/event-stream",
    )


async def _stream_direct_chat(
    conversation_id: uuid.UUID,
    route: InteractionDecision,
) -> AsyncIterator[dict[str, Any]]:
    """Stream a router-authored reply for turns that do not need retrieval."""
    response = route.response.strip() or _direct_reply_fallback(route.action)
    done_payload = {
        "response": response,
        "generator_id": "teacher_gen",
        "output_type": "text",
        "artifacts": [],
        "sources": [],
        "learner_updates": LearnerUpdates().model_dump(),
        "metadata": {
            "interaction_router": True,
            "mode": route.action,
        },
    }

    yield {"event": "token", "data": json.dumps({"delta": response}, default=str)}
    yield {"event": "sources", "data": json.dumps([], default=str)}
    learner_state = await _persist_assistant_turn(
        conversation_id=conversation_id,
        generator_id="teacher_gen",
        output_type="text",
        collected_text=[response],
        sources=[],
        artifacts=[],
        done_payload=done_payload,
    )
    done_payload["learner_state"] = learner_state.model_dump()
    yield {"event": "done", "data": json.dumps(done_payload, default=str)}


async def _stream_chat(
    conversation_id: uuid.UUID,
    entry: GeneratorEntry,
    payload: GeneratorInput,
    grouter: GeneratorRouter,
    *,
    user_message_id: uuid.UUID,
) -> AsyncIterator[dict[str, Any]]:
    """Proxy generator events to the client and persist the assistant turn on done."""
    collected_text: list[str] = []
    sources: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    done_payload: dict[str, Any] | None = None
    persisted = False

    try:
        async for event in grouter.dispatch_stream(entry, payload):
            data = event.data

            if event.event == "chunk":
                text = _extract_text(data)
                if text:
                    collected_text.append(text)
            elif event.event == "sources" and isinstance(data, list):
                sources = [s for s in data if isinstance(s, dict)]
            elif event.event == "artifact" and isinstance(data, dict):
                artifacts.append(data)
            elif event.event == "done" and isinstance(data, dict):
                done_payload = data
                learner_state = await _persist_assistant_turn(
                    conversation_id=conversation_id,
                    generator_id=entry.id,
                    output_type="text",
                    collected_text=collected_text,
                    sources=sources,
                    artifacts=artifacts,
                    done_payload=done_payload,
                    answered_course_question_user_message_id=user_message_id,
                    answered_course_question_text=payload.user_message,
                )
                done_payload = {**done_payload, "learner_state": learner_state.model_dump()}
                data = done_payload
                persisted = True

            yield {"event": event.event, "data": json.dumps(data, default=str)}

            if event.event == "error":
                return
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat stream failed")
        yield {"event": "error", "data": json.dumps({"message": str(exc)})}
        return

    if not persisted:
        await _persist_assistant_turn(
            conversation_id=conversation_id,
            generator_id=entry.id,
            output_type="text",
            collected_text=collected_text,
            sources=sources,
            artifacts=artifacts,
            done_payload=done_payload,
            answered_course_question_user_message_id=user_message_id,
            answered_course_question_text=payload.user_message,
        )


async def _persist_assistant_turn(
    *,
    conversation_id: uuid.UUID,
    generator_id: str,
    output_type: str,
    collected_text: list[str],
    sources: list[dict[str, Any]],
    artifacts: list[dict[str, Any]],
    done_payload: dict[str, Any] | None,
    answered_course_question_user_message_id: uuid.UUID | None = None,
    answered_course_question_text: str | None = None,
) -> LearnerState:
    done = done_payload or {}
    response_text = done.get("response") or "".join(collected_text)
    final_sources = done.get("sources") if isinstance(done.get("sources"), list) else sources
    final_artifacts = (
        done.get("artifacts") if isinstance(done.get("artifacts"), list) else artifacts
    )
    updates_raw = done.get("learner_updates") or {}
    updates = LearnerUpdates.model_validate(updates_raw) if isinstance(updates_raw, dict) else LearnerUpdates()

    async with session_scope() as bg_session:
        assistant_message = Message(
            conversation_id=conversation_id,
            role="assistant",
            content=response_text,
            generator_id=generator_id,
            output_type=output_type,
            sources=list(final_sources or []),
            artifacts=list(final_artifacts or []),
        )
        bg_session.add(assistant_message)
        await bg_session.flush()
        learner_state = await get_learner_tracker().apply_updates(
            bg_session,
            conversation_id,
            updates,
            allow_mastery_updates=False,
        )
        source_chunk_ids = _source_chunk_ids(final_sources or sources)
        if (
            answered_course_question_user_message_id is not None
            and response_text.strip()
            and source_chunk_ids
            and _looks_like_learning_question(answered_course_question_text or "")
        ):
            await get_review_test_service().record_answered_course_question(
                bg_session,
                conversation_id,
                user_message_id=answered_course_question_user_message_id,
                assistant_message_id=assistant_message.id,
                source_chunk_ids=source_chunk_ids,
                learner_updates=updates,
            )
        return learner_state


async def _load_history(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    *,
    limit: int,
) -> list[dict[str, str]]:
    result = await session.execute(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.created_at.desc())
        .limit(limit)
    )
    messages = list(result.scalars().all())[::-1]
    return [{"role": m.role, "content": m.content} for m in messages]


def _direct_reply_fallback(action: str) -> str:
    if action == "outside_files":
        return (
            "That seems outside the uploaded course files, so I can't answer it "
            "from your sources. Ask me about the course material and I'll help."
        )
    return "I'm here with you. Tell me what you'd like to work on from your course materials."


def _extract_text(data: Any) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in ("text", "delta", "content", "chunk"):
            value = data.get(key)
            if isinstance(value, str):
                return value
    return ""


def _source_chunk_ids(sources: list[dict[str, Any]]) -> list[str]:
    ids: list[str] = []
    seen: set[str] = set()
    for source in sources:
        raw = source.get("chunk_id")
        if raw is None:
            continue
        text = str(raw)
        if text and text not in seen:
            seen.add(text)
            ids.append(text)
    return ids


def _looks_like_learning_question(message: str) -> bool:
    text = " ".join(message.casefold().split())
    if len(text) < 4:
        return False
    if text in {"hi", "hello", "hey", "thanks", "thank you", "ok", "okay"}:
        return False
    if "?" in text:
        return True
    starters = (
        "explain",
        "teach",
        "show",
        "compare",
        "summarize",
        "résume",
        "resume",
        "explique",
        "montre",
        "compare",
        "pourquoi",
        "comment",
        "what",
        "why",
        "how",
        "can you",
        "could you",
    )
    return text.startswith(starters)
