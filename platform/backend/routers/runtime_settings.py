from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from db.session import get_db
from schemas.runtime_settings import RuntimeSettingsRead, RuntimeSettingsUpdate
from services.runtime_settings_service import RuntimeSettingsError, get_runtime_settings_service


router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.get("/runtime", response_model=RuntimeSettingsRead)
async def get_runtime_settings(
    session: AsyncSession = Depends(get_db),
) -> RuntimeSettingsRead:
    return await get_runtime_settings_service().get_public_settings(session)


@router.patch("/runtime", response_model=RuntimeSettingsRead)
async def update_runtime_settings(
    body: RuntimeSettingsUpdate,
    session: AsyncSession = Depends(get_db),
) -> RuntimeSettingsRead:
    try:
        return await get_runtime_settings_service().update_settings(session, body)
    except RuntimeSettingsError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
