from fastapi import APIRouter, HTTPException, Response

from local_api.db import get_store
from local_api.schemas import ConversationCreate, ConversationUpdate
from local_api.services.learner import get_learner_service

router = APIRouter(prefix="/api/conversations", tags=["conversations"])


@router.get("")
async def list_conversations() -> dict:
    return {"conversations": get_store().list_conversations()}


@router.post("")
async def create_conversation(payload: ConversationCreate) -> dict:
    return get_store().create_conversation(payload.title)


@router.get("/{conversation_id}")
async def get_conversation(conversation_id: str) -> dict:
    conversation = get_store().get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return conversation


@router.patch("/{conversation_id}")
async def update_conversation(conversation_id: str, payload: ConversationUpdate) -> dict:
    conversation = get_store().get_conversation(conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    title = payload.title.strip() if payload.title is not None else None
    if title == "":
        raise HTTPException(status_code=400, detail="title cannot be empty")
    updated = get_store().update_conversation(conversation_id, title=title)
    if updated is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return updated


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str) -> Response:
    deleted = get_store().delete_conversation(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="conversation not found")
    return Response(status_code=204)


@router.get("/{conversation_id}/messages")
async def list_messages(conversation_id: str) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    return {"messages": get_store().list_messages(conversation_id)}


@router.get("/{conversation_id}/learner-state")
async def learner_state(conversation_id: str) -> dict:
    return get_learner_service().load(conversation_id).model_dump()


@router.get("/{conversation_id}/artifacts")
async def list_artifacts(conversation_id: str) -> dict:
    artifacts = []
    for row in get_store().list_artifacts(conversation_id):
        artifacts.append(
            {
                "type": row["type"],
                "url": f"teacherlm-local://artifact/{row['id']}",
                "filename": row["filename"],
                "key": row["id"],
                "mime_type": row["mime_type"],
                "created_at": row["created_at"],
            }
        )
    return {"artifacts": artifacts}
