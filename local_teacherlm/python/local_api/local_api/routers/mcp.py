from fastapi import APIRouter

from local_api.db import get_store
from local_api.services.generators import get_generator_service

router = APIRouter(prefix="/api/mcp", tags=["mcp"])


@router.get("/generators")
async def mcp_generators() -> dict:
    manifests = [
        manifest.model_dump()
        for manifest in get_generator_service().list_manifests()
        if manifest.transport.startswith("mcp")
    ]
    return {"generators": manifests}


@router.get("/permissions")
async def mcp_permissions() -> dict:
    rows = get_store().query("SELECT * FROM connected_external_agents ORDER BY display_name ASC")
    return {"agents": rows}

