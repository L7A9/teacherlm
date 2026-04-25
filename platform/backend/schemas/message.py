from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


Role = Literal["user", "assistant", "system"]
OutputType = Literal[
    "text",
    "quiz",
    "report",
    "presentation",
    "flashcards",
    "chart",
    "podcast",
    "mindmap",
]


class Artifact(BaseModel):
    type: str
    url: str
    filename: str | None = None


class Source(BaseModel):
    text: str
    source: str
    score: float
    chunk_id: str | None = None


class MessageRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    role: Role
    content: str
    generator_id: str | None = None
    output_type: OutputType | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    sources: list[Source] = Field(default_factory=list)
    created_at: datetime


class MessageList(BaseModel):
    items: list[MessageRead]
    total: int


class ChatRequest(BaseModel):
    """Body for POST /api/conversations/{id}/chat (SSE)."""
    user_message: str = Field(min_length=1)
    options: dict[str, Any] = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    """Body for POST /api/conversations/{id}/generate (SSE)."""
    output_type: OutputType
    options: dict[str, Any] = Field(default_factory=dict)
    topic: str | None = None
