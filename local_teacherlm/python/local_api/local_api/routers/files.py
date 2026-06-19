from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, Response, UploadFile, status

from local_api.db import get_store
from local_api.services.coursebuilder import get_coursebuilder_service
from local_api.services.ingestion import get_ingestion_service

router = APIRouter(prefix="/api/conversations/{conversation_id}/files", tags=["files"])


@router.get("")
async def list_files(conversation_id: str) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    files = get_store().list_files(conversation_id)
    return {"files": files, "items": files, "total": len(files)}


@router.post("")
async def upload_file(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(default=None),
    upload: UploadFile | None = File(default=None),
    llm_options_json: str | None = Form(default=None),
) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    selected = upload or file
    if selected is None:
        raise HTTPException(status_code=400, detail="file is required")
    try:
        record = await get_ingestion_service().ingest_upload(conversation_id, selected)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(get_ingestion_service().process_upload, record["id"])
    return record


@router.post("/batch")
async def upload_file_batch(
    conversation_id: str,
    background_tasks: BackgroundTasks,
    uploads: list[UploadFile] = File(...),
) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    if not uploads:
        raise HTTPException(status_code=400, detail="at least one file is required")
    try:
        records = [
            await get_ingestion_service().ingest_upload(conversation_id, upload)
            for upload in uploads
        ]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    background_tasks.add_task(
        get_ingestion_service().process_upload_batch,
        [record["id"] for record in records],
    )
    return {"files": records, "items": records, "total": len(records)}


@router.get("/{file_id}")
async def get_file(conversation_id: str, file_id: str) -> dict:
    record = get_store().get_file_for_conversation(conversation_id, file_id)
    if record is None:
        raise HTTPException(status_code=404, detail="file not found")
    return record


@router.post("/{file_id}/retry")
async def retry_file(conversation_id: str, file_id: str, background_tasks: BackgroundTasks) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    try:
        record = await get_ingestion_service().retry_upload(conversation_id, file_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail="file not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    background_tasks.add_task(get_ingestion_service().process_upload, record["id"])
    return record


@router.delete("/{file_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_file(conversation_id: str, file_id: str, background_tasks: BackgroundTasks) -> Response:
    if get_store().delete_uploaded_file(conversation_id, file_id) is None:
        raise HTTPException(status_code=404, detail="file not found")
    service = get_coursebuilder_service()
    service.invalidate_plan(conversation_id, "source file deleted")
    remaining = get_store().list_files(conversation_id)
    if remaining and all(file["status"] == "ready" for file in remaining):
        background_tasks.add_task(service.replan_and_rebuild_async, conversation_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
