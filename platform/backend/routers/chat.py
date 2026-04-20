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
from services.learner_tracker import get_learner_tracker
from services.retrieval_orchestrator import get_retrieval_orchestrator


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

    # Persist the user turn up-front so a failure mid-stream still leaves it.
    session.add(
        Message(
            conversation_id=conversation_id,
            role="user",
            content=body.user_message,
        )
    )
    await session.flush()
    context_chunks = await get_retrieval_orchestrator().retrieve_for(
        output_type="text",
        query=body.user_message,
        conversation_id=conversation_id,
    )

    payload = GeneratorInput(
        conversation_id=str(conversation_id),
        user_message=body.user_message,
        context_chunks=context_chunks,
        learner_state=learner_state,
        chat_history=chat_history,
        options=body.options,
    )

    return EventSourceResponse(
        _stream_chat(conversation_id, entry, payload, grouter),
        media_type="text/event-stream",
    )


async def _stream_chat(
    conversation_id: uuid.UUID,
    entry: GeneratorEntry,
    payload: GeneratorInput,
    grouter: GeneratorRouter,
) -> AsyncIterator[dict[str, Any]]:
    """Proxy generator events to the client and persist the assistant turn on done."""
    collected_text: list[str] = []
    sources: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    done_payload: dict[str, Any] | None = None

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

            yield {"event": event.event, "data": json.dumps(data, default=str)}
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat stream failed")
        yield {"event": "error", "data": json.dumps({"message": str(exc)})}
        return

    await _persist_assistant_turn(
        conversation_id=conversation_id,
        generator_id=entry.id,
        output_type="text",
        collected_text=collected_text,
        sources=sources,
        artifacts=artifacts,
        done_payload=done_payload,
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
) -> None:
    done = done_payload or {}
    response_text = done.get("response") or "".join(collected_text)
    final_sources = done.get("sources") if isinstance(done.get("sources"), list) else sources
    final_artifacts = (
        done.get("artifacts") if isinstance(done.get("artifacts"), list) else artifacts
    )
    updates_raw = done.get("learner_updates") or {}
    updates = LearnerUpdates.model_validate(updates_raw) if isinstance(updates_raw, dict) else LearnerUpdates()

    async with session_scope() as bg_session:
        bg_session.add(
            Message(
                conversation_id=conversation_id,
                role="assistant",
                content=response_text,
                generator_id=generator_id,
                output_type=output_type,
                sources=list(final_sources or []),
                artifacts=list(final_artifacts or []),
            )
        )
        await get_learner_tracker().apply_updates(bg_session, conversation_id, updates)


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


def _extract_text(data: Any) -> str:
    if isinstance(data, str):
        return data
    if isinstance(data, dict):
        for key in ("text", "delta", "content", "chunk"):
            value = data.get(key)
            if isinstance(value, str):
                return value
    return ""
