from fastapi import APIRouter

from local_api.config import get_settings

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health")
async def health() -> dict:
    settings = get_settings()
    return {
        "ok": True,
        "app": settings.app_name,
        "data_dir": str(settings.data_dir),
        "db_path": str(settings.db_path),
    }

