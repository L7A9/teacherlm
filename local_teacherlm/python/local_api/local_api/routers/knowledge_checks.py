from fastapi import APIRouter

from local_api.db import get_store

router = APIRouter(prefix="/api/conversations/{conversation_id}/knowledge-checks", tags=["knowledge-checks"])


@router.get("")
async def list_knowledge_checks(conversation_id: str) -> dict:
    rows = get_store().query(
        "SELECT * FROM knowledge_checks WHERE conversation_id = ? ORDER BY created_at DESC",
        (conversation_id,),
    )
    return {"checks": rows}

