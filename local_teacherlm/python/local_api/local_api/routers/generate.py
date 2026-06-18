from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from teacherlm_core.schemas import GeneratorInput, GeneratorOutput

from local_api.db import get_store
from local_api.routers._sse import sse
from local_api.schemas import GenerateRequest
from local_api.services.generators import get_generator_service
from local_api.services.learner import get_learner_service
from local_api.services.retrieval import get_retrieval_service

router = APIRouter(prefix="/api/conversations/{conversation_id}", tags=["generate"])


@router.post("/generate")
async def generate(conversation_id: str, payload: GenerateRequest) -> StreamingResponse:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return StreamingResponse(_generate_stream(conversation_id, payload), media_type="text/event-stream")


async def _generate_stream(conversation_id: str, payload: GenerateRequest) -> AsyncIterator[str]:
    prompt = payload.prompt or f"Generate {payload.output_type}"
    get_store().add_message(conversation_id, "user", prompt, output_type=payload.output_type)
    try:
        manifest = get_generator_service().manifest_for_output(payload.output_type)
    except KeyError as exc:
        yield sse("error", {"message": str(exc)})
        return
    context_chunks = await get_retrieval_service().retrieve_for(
        conversation_id=conversation_id,
        user_message=prompt,
        output_type=payload.output_type,
        source_file_ids=payload.source_file_ids,
        options=payload.options,
    )
    generator_input = GeneratorInput(
        conversation_id=conversation_id,
        user_message=prompt,
        context_chunks=context_chunks,
        learner_state=get_learner_service().load(conversation_id),
        chat_history=[
            {"role": row["role"], "content": row["content"]}
            for row in get_store().list_messages(conversation_id)[-12:]
        ],
        options=payload.options,
    )
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

