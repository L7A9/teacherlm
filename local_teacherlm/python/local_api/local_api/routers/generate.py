from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from teacherlm_core.schemas import GeneratorInput, GeneratorOutput

from local_api.db import get_store, new_id
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
    source_file_ids = list(dict.fromkeys(payload.source_file_ids))
    generation_options = dict(payload.options)
    if payload.output_type in {"mindmap", "quiz"}:
        if not source_file_ids:
            output_label = "mind map" if payload.output_type == "mindmap" else "quiz"
            yield sse("error", {"message": f"Select at least one ready source file before generating the {output_label}."})
            return
        ready_file_ids = {
            str(row["id"])
            for row in get_store().list_files(conversation_id)
            if str(row.get("status") or "") == "ready"
        }
        invalid_file_ids = [file_id for file_id in source_file_ids if file_id not in ready_file_ids]
        if invalid_file_ids:
            yield sse(
                "error",
                {
                    "message": f"The {payload.output_type} can only be generated from ready files checked in this conversation.",
                    "invalid_source_file_ids": invalid_file_ids,
                },
            )
            return
    if payload.output_type == "mindmap":
        generation_options.update(
            {
                "generation_mode": "full_rebuild",
                "generation_run_id": new_id("mindmap_run"),
                "rebuild_from_scratch": True,
                "source_file_ids_snapshot": source_file_ids,
            }
        )
    elif payload.output_type == "quiz":
        generation_options.update(
            {
                "generation_mode": "fresh_quiz",
                "generation_run_id": new_id("quiz_run"),
                "fresh_generation": True,
                "rebuild_from_scratch": True,
                "retrieval_mode": "full_selected_files_with_graph",
                "include_knowledge_graph": True,
                "source_file_ids_snapshot": source_file_ids,
            }
        )
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
        source_file_ids=source_file_ids,
        options=generation_options,
    )
    chat_history = [] if payload.output_type in {"mindmap", "quiz"} else [
        {"role": row["role"], "content": row["content"]}
        for row in get_store().list_messages(conversation_id)[-12:]
    ]
    generator_input = GeneratorInput(
        conversation_id=conversation_id,
        user_message=prompt,
        context_chunks=context_chunks,
        learner_state=get_learner_service().load(conversation_id),
        chat_history=chat_history,
        options=generation_options,
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

