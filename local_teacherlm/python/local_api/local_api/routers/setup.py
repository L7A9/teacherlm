from fastapi import APIRouter, status

from local_api.services.runtime_setup import get_runtime_setup_service


router = APIRouter(prefix="/api/setup", tags=["setup"])


@router.get("")
async def setup_status() -> dict:
    return await get_runtime_setup_service().status()


@router.post("", status_code=status.HTTP_202_ACCEPTED)
async def start_setup() -> dict:
    return await get_runtime_setup_service().start()
