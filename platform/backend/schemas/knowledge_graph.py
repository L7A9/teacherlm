from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class KnowledgeGraphNodeRead(BaseModel):
    id: uuid.UUID
    node_type: str
    label: str
    description: str = ""
    ref_id: str | None = None
    source_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphEdgeRead(BaseModel):
    id: uuid.UUID
    source_node_id: uuid.UUID
    target_node_id: uuid.UUID
    relation_type: str
    confidence: float
    source_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class KnowledgeGraphRead(BaseModel):
    conversation_id: uuid.UUID
    nodes: list[KnowledgeGraphNodeRead] = Field(default_factory=list)
    edges: list[KnowledgeGraphEdgeRead] = Field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0


class KnowledgeGraphRebuildRequest(BaseModel):
    options: dict[str, Any] = Field(default_factory=dict)


class RemediationStep(BaseModel):
    concept_id: uuid.UUID | None = None
    concept_name: str
    mastery: float = 0.0
    reason: str
    source_chunk_ids: list[str] = Field(default_factory=list)


class RemediationPath(BaseModel):
    target_concept_id: uuid.UUID
    target_concept_name: str
    steps: list[RemediationStep] = Field(default_factory=list)
    source: str = "knowledge_graph"
