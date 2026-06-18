from fastapi import APIRouter

from local_api.services.generators import get_generator_service

router = APIRouter(prefix="/api/generators", tags=["generators"])


@router.get("")
async def list_generators() -> dict:
    return {
        "generators": [
            manifest.model_dump() for manifest in get_generator_service().list_manifests()
        ]
    }

