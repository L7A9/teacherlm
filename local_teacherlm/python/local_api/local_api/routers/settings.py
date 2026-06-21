from fastapi import APIRouter, HTTPException

from local_api.schemas import (
    CourseBuilderSettingsUpdate,
    GeneratorSettingsUpdate,
    ParserSettingsUpdate,
    ProviderPatch,
    ProviderWrite,
    RetrievalSettingsUpdate,
)
from local_api.services.settings import get_settings_service

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/runtime")
async def runtime_settings() -> dict:
    return get_settings_service().runtime_settings().model_dump()


@router.get("/llm-providers")
async def list_providers() -> dict:
    return {"providers": [provider.model_dump() for provider in get_settings_service().list_providers()]}


@router.post("/llm-providers")
async def create_provider(payload: ProviderWrite) -> dict:
    return get_settings_service().create_provider(payload).model_dump()


@router.patch("/llm-providers/{provider_id}")
async def update_provider(provider_id: str, payload: ProviderPatch) -> dict:
    try:
        return get_settings_service().update_provider(provider_id, payload).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="provider not found") from exc


@router.delete("/llm-providers/{provider_id}")
async def delete_provider(provider_id: str) -> dict:
    try:
        get_settings_service().delete_provider(provider_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="provider not found") from exc
    return {"ok": True}


@router.post("/llm-providers/{provider_id}/test")
async def test_provider(provider_id: str) -> dict:
    try:
        return (await get_settings_service().test_provider(provider_id)).model_dump()
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="provider not found") from exc


@router.get("/parse")
async def parser_settings() -> dict:
    return get_settings_service().get_parser_settings().model_dump()


@router.patch("/parse")
async def update_parser_settings(payload: ParserSettingsUpdate) -> dict:
    return get_settings_service().update_parser_settings(payload).model_dump()


@router.get("/coursebuilder")
async def coursebuilder_settings() -> dict:
    return get_settings_service().get_coursebuilder_settings().model_dump()


@router.patch("/coursebuilder")
async def update_coursebuilder_settings(payload: CourseBuilderSettingsUpdate) -> dict:
    return get_settings_service().update_coursebuilder_settings(payload).model_dump()


@router.get("/generators")
async def generator_settings() -> dict:
    return get_settings_service().get_generator_settings().model_dump()


@router.patch("/generators")
async def update_generator_settings(payload: GeneratorSettingsUpdate) -> dict:
    return get_settings_service().update_generator_settings(payload).model_dump()


@router.get("/retrieval")
async def retrieval_settings() -> dict:
    return get_settings_service().get_retrieval_settings().model_dump()


@router.patch("/retrieval")
async def update_retrieval_settings(payload: RetrievalSettingsUpdate) -> dict:
    return get_settings_service().update_retrieval_settings(payload).model_dump()
