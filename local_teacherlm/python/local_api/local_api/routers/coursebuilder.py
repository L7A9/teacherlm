from fastapi import APIRouter, HTTPException

from local_api.db import get_store
from local_api.services.coursebuilder import get_coursebuilder_service

router = APIRouter(prefix="/api/conversations/{conversation_id}/coursebuilder", tags=["coursebuilder"])


@router.get("")
async def get_coursebuilder(conversation_id: str) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return get_coursebuilder_service().get_or_build(conversation_id)


@router.post("/rebuild")
async def rebuild_coursebuilder(conversation_id: str) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return get_coursebuilder_service().rebuild(conversation_id)
