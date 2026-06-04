from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import Conversation
from db.session import get_db
from schemas.knowledge_graph import KnowledgeGraphRead, KnowledgeGraphRebuildRequest, RemediationPath
from services.knowledge_graph_service import get_knowledge_graph_service
from services.runtime_settings_service import get_runtime_settings_service


router = APIRouter(prefix="/api/conversations", tags=["knowledge-graph"])


@router.get("/{conversation_id}/knowledge-graph", response_model=KnowledgeGraphRead)
async def get_knowledge_graph(
    conversation_id: uuid.UUID,
    session: AsyncSession = Depends(get_db),
) -> KnowledgeGraphRead:
    await _require_conversation(session, conversation_id)
    graph = await get_knowledge_graph_service().get_graph(session, conversation_id)
    if graph.node_count == 0:
        graph = await get_knowledge_graph_service().rebuild_graph(session, conversation_id)
    return graph


@router.post("/{conversation_id}/knowledge-graph/rebuild", response_model=KnowledgeGraphRead)
async def rebuild_knowledge_graph(
    conversation_id: uuid.UUID,
    body: KnowledgeGraphRebuildRequest | None = None,
    session: AsyncSession = Depends(get_db),
) -> KnowledgeGraphRead:
    await _require_conversation(session, conversation_id)
    resolved_options = await get_runtime_settings_service().resolve_options(
        session,
        body.options if body else None,
    )
    return await get_knowledge_graph_service().rebuild_graph(
        session,
        conversation_id,
        llm_options=resolved_options,
    )


@router.get("/{conversation_id}/knowledge-graph/remediation", response_model=RemediationPath)
async def get_graph_remediation(
    conversation_id: uuid.UUID,
    concept_id: uuid.UUID = Query(...),
    session: AsyncSession = Depends(get_db),
) -> RemediationPath:
    await _require_conversation(session, conversation_id)
    path = await get_knowledge_graph_service().remediation_for_concept(
        session,
        conversation_id,
        concept_id,
    )
    if path is None:
        raise HTTPException(status_code=404, detail="no graph remediation found")
    return path


async def _require_conversation(session: AsyncSession, conversation_id: uuid.UUID) -> None:
    if await session.get(Conversation, conversation_id) is None:
        raise HTTPException(status_code=404, detail="conversation not found")
