from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from dispatcher.registry import (
    GeneratorEntry,
    GeneratorNotFound,
    GeneratorRegistry,
    get_registry,
)


router = APIRouter(prefix="/api/generators", tags=["generators"])


class GeneratorView(BaseModel):
    id: str
    name: str | None
    output_type: str
    icon: str | None
    description: str | None
    is_chat_default: bool
    enabled: bool


class GeneratorListResponse(BaseModel):
    items: list[GeneratorView]


def _to_view(entry: GeneratorEntry) -> GeneratorView:
    return GeneratorView(
        id=entry.id,
        name=entry.name,
        output_type=entry.output_type,
        icon=entry.icon,
        description=entry.description,
        is_chat_default=entry.is_chat_default,
        enabled=entry.enabled,
    )


@router.get("", response_model=GeneratorListResponse)
async def list_generators(
    registry: GeneratorRegistry = Depends(get_registry),
    include_disabled: bool = Query(default=False),
) -> GeneratorListResponse:
    entries = registry.all(only_enabled=not include_disabled)
    return GeneratorListResponse(items=[_to_view(e) for e in entries])


@router.get("/{generator_id}", response_model=GeneratorView)
async def get_generator(
    generator_id: str,
    registry: GeneratorRegistry = Depends(get_registry),
) -> GeneratorView:
    try:
        entry = registry.get(generator_id)
    except GeneratorNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return _to_view(entry)
