from fastapi import APIRouter

from local_api.db import get_store

router = APIRouter(prefix="/api/conversations/{conversation_id}/review-tests", tags=["review-tests"])


@router.get("")
async def list_review_tests(conversation_id: str) -> dict:
    rows = get_store().query(
        "SELECT * FROM review_windows WHERE conversation_id = ? ORDER BY created_at DESC",
        (conversation_id,),
    )
    return {"review_windows": rows}

