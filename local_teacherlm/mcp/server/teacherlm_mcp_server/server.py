from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx


DEFAULT_LOCAL_API = "http://127.0.0.1:8765/api"


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    input_schema: dict[str, Any]
    permissions: tuple[str, ...]


TOOLS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name="teacherlm.list_conversations",
        description="List local course workspaces approved for this connected tool.",
        input_schema={"type": "object", "properties": {}},
        permissions=("read_context",),
    ),
    ToolDefinition(
        name="teacherlm.list_sources",
        description="List uploaded source files for a conversation.",
        input_schema={
            "type": "object",
            "required": ["conversation_id"],
            "properties": {"conversation_id": {"type": "string"}},
        },
        permissions=("read_context",),
    ),
    ToolDefinition(
        name="teacherlm.retrieve_context",
        description="Retrieve grounded chunks for a query and output type.",
        input_schema={
            "type": "object",
            "required": ["conversation_id", "query", "output_type"],
            "properties": {
                "conversation_id": {"type": "string"},
                "query": {"type": "string"},
                "output_type": {"type": "string"},
                "source_file_ids": {"type": "array", "items": {"type": "string"}},
            },
        },
        permissions=("read_context",),
    ),
    ToolDefinition(
        name="teacherlm.get_learner_state",
        description="Return scoped learner state for a conversation.",
        input_schema={
            "type": "object",
            "required": ["conversation_id"],
            "properties": {"conversation_id": {"type": "string"}},
        },
        permissions=("read_learner_state",),
    ),
    ToolDefinition(
        name="teacherlm.create_artifact",
        description="Store an artifact through TeacherLM's local artifact store.",
        input_schema={
            "type": "object",
            "required": ["conversation_id", "type", "filename", "content"],
            "properties": {
                "conversation_id": {"type": "string"},
                "type": {"type": "string"},
                "filename": {"type": "string"},
                "content": {"type": "string"},
                "mime_type": {"type": "string"},
            },
        },
        permissions=("write_artifacts",),
    ),
)


def list_tool_contracts() -> list[dict[str, Any]]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "input_schema": tool.input_schema,
            "permissions": list(tool.permissions),
        }
        for tool in TOOLS
    ]


async def call_local_api(path: str, *, method: str = "GET", payload: dict[str, Any] | None = None) -> Any:
    async with httpx.AsyncClient(base_url=DEFAULT_LOCAL_API, timeout=30.0) as client:
        response = await client.request(method, path, json=payload)
        response.raise_for_status()
        return response.json()

