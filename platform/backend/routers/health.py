from __future__ import annotations

import logging

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from config import get_settings
from db.session import get_session_factory
from services.storage_service import get_storage
from services.vector_service import get_vector_service


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/health", tags=["health"])


class LivenessResponse(BaseModel):
    status: str
    app: str
    environment: str


class ReadinessCheck(BaseModel):
    ok: bool
    error: str | None = None


class ReadinessResponse(BaseModel):
    ready: bool
    checks: dict[str, ReadinessCheck]


@router.get("", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    settings = get_settings()
    return LivenessResponse(
        status="ok",
        app=settings.app_name,
        environment=settings.environment,
    )


@router.get("/ready", response_model=ReadinessResponse)
async def readiness() -> ReadinessResponse:
    checks: dict[str, ReadinessCheck] = {
        "database": await _check_database(),
        "minio": await _check_minio(),
        "qdrant": await _check_qdrant(),
    }
    ready = all(c.ok for c in checks.values())
    return ReadinessResponse(ready=ready, checks=checks)


async def _check_database() -> ReadinessCheck:
    try:
        factory = get_session_factory()
        async with factory() as session:
            await session.execute(text("SELECT 1"))
        return ReadinessCheck(ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("database readiness failed: %s", exc)
        return ReadinessCheck(ok=False, error=f"{type(exc).__name__}: {exc}")


async def _check_minio() -> ReadinessCheck:
    try:
        await get_storage().ensure_bucket()
        return ReadinessCheck(ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("minio readiness failed: %s", exc)
        return ReadinessCheck(ok=False, error=f"{type(exc).__name__}: {exc}")


async def _check_qdrant() -> ReadinessCheck:
    try:
        await get_vector_service()._client.get_collections()
        return ReadinessCheck(ok=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qdrant readiness failed: %s", exc)
        return ReadinessCheck(ok=False, error=f"{type(exc).__name__}: {exc}")
