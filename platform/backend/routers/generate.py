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

from teacherlm_core.schemas.generator_io import GeneratorInput, LearnerUpdates

from db.models import Conversation, Message
from db.session import get_db, session_scope
from dispatcher.registry import GeneratorEntry, GeneratorNotFound
from dispatcher.router import GeneratorRouter, get_router
from schemas.message import GenerateRequest
from services.learner_tracker import get_learner_tracker
from services.retrieval_orchestrator import get_retrieval_orchestrator


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/conversations", tags=["generate"])


CHAT_HISTORY_LIMIT = 20


_SYNTH_PROMPTS: dict[str, str] = {
    "quiz": "Generate a quiz{topic}.",
    "report": "Generate a study report{topic}.",
    "presentation": "Generate a presentation{topic}.",
    "podcast": "Generate a podcast{topic}.",
    "chart": "Generate a chart{topic}.",
    "mindmap": "Generate a mind map{topic}.",
    "text": "Explain this topic{topic}.",
}


def _synthesize_prompt(output_type: str, topic: str | None) -> str:
    template = _SYNTH_PROMPTS.get(output_type, f"Generate a {output_type}{{topic}}.")
    suffix = f" about {topic}" if topic else ""
    return template.format(topic=suffix)


@router.post("/{conversation_id}/generate")
async def generate(
    conversation_id: uuid.UUID,
    body: GenerateRequest,
    session: AsyncSession = Depends(get_db),
    grouter: GeneratorRouter = Depends(get_router),
) -> EventSourceResponse:
    conversation = await session.get(Conversation, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")

    try:
        entry = grouter.resolve_for_output(body.output_type)
    except GeneratorNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    synth_message = _synthesize_prompt(body.output_type, body.topic)
    # No topic = no query: the orchestrator falls back to a broad corpus
    # sample. Using the synth message ("Generate a quiz.") as a query
    # produces near-empty retrieval pools because nothing in the source
    # files matches that phrase.
    retrieval_query = body.topic or ""

    chat_history = await _load_history(session, conversation_id, limit=CHAT_HISTORY_LIMIT)
    learner_state = await get_learner_tracker().load_state(session, conversation_id)

    session.add(
        Message(
            conversation_id=conversation_id,
            role="user",
            content=synth_message,
        )
    )
    await session.flush()

    context_chunks = await get_retrieval_orchestrator().retrieve_for(
        output_type=body.output_type,
        query=retrieval_query,
        conversation_id=conversation_id,
    )

    payload = GeneratorInput(
        conversation_id=str(conversation_id),
        user_message=synth_message,
        context_chunks=context_chunks,
        learner_state=learner_state,
        chat_history=chat_history,
        options={**body.options, "topic": body.topic} if body.topic else dict(body.options),
    )

    return EventSourceResponse(
        _stream_generate(conversation_id, entry, body.output_type, payload, grouter),
        media_type="text/event-stream",
    )


async def _stream_generate(
    conversation_id: uuid.UUID,
    entry: GeneratorEntry,
    output_type: str,
    payload: GeneratorInput,
    grouter: GeneratorRouter,
) -> AsyncIterator[dict[str, Any]]:
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

            if event.event == "error":
                return
    except Exception as exc:  # noqa: BLE001
        logger.exception("generate stream failed")
        yield {"event": "error", "data": json.dumps({"message": str(exc)})}
        return

    await _persist_assistant_turn(
        conversation_id=conversation_id,
        generator_id=entry.id,
        output_type=output_type,
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
    updates = (
        LearnerUpdates.model_validate(updates_raw)
        if isinstance(updates_raw, dict)
        else LearnerUpdates()
    )

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
