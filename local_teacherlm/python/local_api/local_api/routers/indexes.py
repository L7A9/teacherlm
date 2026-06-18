from fastapi import APIRouter, HTTPException

from local_api.db import get_store
from local_api.services.knowledge_graph import get_knowledge_graph_service
from local_api.services.vector_service import get_vector_service

router = APIRouter(prefix="/api/conversations/{conversation_id}/indexes", tags=["indexes"])


@router.post("/rebuild")
async def rebuild_indexes(conversation_id: str) -> dict:
    if get_store().get_conversation(conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    vector_status = await get_vector_service().rebuild_conversation(conversation_id)
    graph = get_knowledge_graph_service().rebuild_graph(conversation_id)
    return {
        "ok": True,
        "conversation_id": conversation_id,
        "index_status": {
            **vector_status,
            "graph_node_count": graph["node_count"],
            "graph_edge_count": graph["edge_count"],
        },
    }
