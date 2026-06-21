from fastapi import APIRouter, BackgroundTasks, HTTPException

from local_api.db import get_store
from local_api.schemas import CourseQuizSubmission
from local_api.services.coursebuilder import get_coursebuilder_service

router = APIRouter(prefix="/api/conversations/{conversation_id}/coursebuilder", tags=["coursebuilder"])


@router.get("")
async def get_coursebuilder(conversation_id: str) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return get_coursebuilder_service().get_or_build(conversation_id)


@router.get("/plan")
async def get_coursebuilder_plan(conversation_id: str) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return get_coursebuilder_service().get_plan(conversation_id)


@router.post("/rebuild", status_code=202)
async def rebuild_coursebuilder(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    improved_quality: bool = False,
) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    service = get_coursebuilder_service()
    current = service.get_or_build(conversation_id)
    if current.get("status") in {"empty", "waiting_for_files"}:
        return current
    stopped = current.get("status") == "stopped"
    force = not stopped or bool(current.get("metadata", {}).get("resume_requires_fresh_plan"))
    cancel_event = service.begin_build(conversation_id, force=force)
    queued = service.mark_build_queued(
        conversation_id,
        resuming=stopped and not force,
        improved_quality=improved_quality,
    )
    background_tasks.add_task(
        service.rebuild_async,
        conversation_id,
        force=force,
        improved_quality=improved_quality,
        _cancel_event=cancel_event,
    )
    return queued


@router.post("/stop")
async def stop_coursebuilder(conversation_id: str) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return get_coursebuilder_service().stop_build(conversation_id)


@router.post("/lessons/{lesson_id}/complete")
async def complete_course_lesson(conversation_id: str, lesson_id: str) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    try:
        return get_coursebuilder_service().mark_lesson_complete(conversation_id, lesson_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/quizzes/{quiz_id}/submit")
async def submit_course_quiz(conversation_id: str, quiz_id: str, payload: CourseQuizSubmission) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    try:
        return get_coursebuilder_service().submit_quiz(
            conversation_id,
            quiz_id,
            [answer.model_dump() for answer in payload.answers],
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
