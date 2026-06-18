from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from teacherlm_core.schemas import GeneratorInput, GeneratorOutput

from local_api.db import get_store
from local_api.routers._sse import sse
from local_api.schemas import ChatRequest
from local_api.services.generators import get_generator_service
from local_api.services.learner import get_learner_service
from local_api.services.retrieval import get_retrieval_service

router = APIRouter(prefix="/api/conversations/{conversation_id}", tags=["chat"])


@router.post("/chat")
async def chat(conversation_id: str, payload: ChatRequest) -> StreamingResponse:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return StreamingResponse(_chat_stream(conversation_id, payload), media_type="text/event-stream")


async def _chat_stream(conversation_id: str, payload: ChatRequest) -> AsyncIterator[str]:
    get_store().add_message(conversation_id, "user", payload.message, output_type="text")
    history = [
        {"role": row["role"], "content": row["content"]}
        for row in get_store().list_messages(conversation_id)[-12:]
    ]
    context_chunks = await get_retrieval_service().retrieve_for(
        conversation_id=conversation_id,
        user_message=payload.message,
        output_type="text",
        source_file_ids=payload.source_file_ids,
        options=payload.options,
    )
    generator_input = GeneratorInput(
        conversation_id=conversation_id,
        user_message=payload.message,
        context_chunks=context_chunks,
        learner_state=get_learner_service().load(conversation_id),
        chat_history=history,
        options=payload.options,
    )
    manifest = get_generator_service().chat_default()
    final: GeneratorOutput | None = None
    try:
        async for event in get_generator_service().run(manifest, generator_input):
            if event["event"] == "done":
                final = GeneratorOutput.model_validate(event["data"])
            yield sse(event["event"], event["data"])
    except Exception as exc:  # noqa: BLE001
        yield sse("error", {"message": str(exc)})
        return
    if final is not None:
        get_store().add_message(
            conversation_id,
            "assistant",
            final.response,
            output_type=final.output_type,
            artifacts=[artifact.model_dump() for artifact in final.artifacts],
            sources=[source.model_dump() for source in final.sources],
            metadata=final.metadata,
        )
        get_learner_service().apply_updates(conversation_id, final.learner_updates)

