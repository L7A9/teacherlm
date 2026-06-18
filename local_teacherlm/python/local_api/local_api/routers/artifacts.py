from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from local_api.db import get_store
from local_api.services.artifacts import get_artifact_service

router = APIRouter(prefix="/api/artifacts", tags=["artifacts"])


@router.get("/{artifact_id}")
async def get_artifact(artifact_id: str) -> FileResponse:
    row = get_store().get_artifact(artifact_id)
    path = get_artifact_service().path_for(artifact_id)
    if row is None or path is None or not path.exists():
        raise HTTPException(status_code=404, detail="artifact not found")
    return FileResponse(path, media_type=row["mime_type"], filename=row["filename"])

