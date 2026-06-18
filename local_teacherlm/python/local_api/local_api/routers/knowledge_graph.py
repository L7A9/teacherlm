from fastapi import APIRouter

from local_api.services.knowledge_graph import get_knowledge_graph_service

router = APIRouter(prefix="/api/conversations/{conversation_id}/knowledge-graph", tags=["knowledge-graph"])


@router.get("")
async def knowledge_graph(conversation_id: str) -> dict:
    return get_knowledge_graph_service().get_graph(conversation_id)
