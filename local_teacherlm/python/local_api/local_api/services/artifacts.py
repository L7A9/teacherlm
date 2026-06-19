from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from teacherlm_core.schemas.generator_io import GeneratorArtifact

from local_api.config import get_settings
from local_api.db import get_store, new_id


_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class ArtifactService:
    def create_artifact(
        self,
        conversation_id: str,
        artifact_type: str,
        filename: str,
        payload: bytes | bytearray | memoryview | str | dict[str, Any] | list[Any],
        *,
        mime_type: str = "application/octet-stream",
        source_message_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GeneratorArtifact:
        artifact_id = new_id("artifact")
        safe_filename = _SAFE_NAME_RE.sub("_", filename).strip("_") or f"{artifact_type}.bin"
        local_key = f"artifacts/{_artifact_dir(artifact_type)}/{artifact_id}_{safe_filename}"
        path = get_settings().data_dir / local_key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(_to_bytes(payload))
        get_store().insert_artifact(
            artifact_id=artifact_id,
            conversation_id=conversation_id,
            artifact_type=artifact_type,
            filename=safe_filename,
            local_key=local_key,
            mime_type=mime_type,
            source_message_id=source_message_id,
            metadata=metadata,
        )
        return GeneratorArtifact(
            type=artifact_type,
            url=f"teacherlm-local://artifact/{artifact_id}",
            filename=safe_filename,
            key=artifact_id,
        )

    def path_for(self, artifact_id: str) -> Path | None:
        row = get_store().get_artifact(artifact_id)
        if row is None:
            return None
        return get_settings().data_dir / row["local_key"]


def _to_bytes(payload: bytes | bytearray | memoryview | str | dict[str, Any] | list[Any]) -> bytes:
    if isinstance(payload, (bytes, bytearray, memoryview)):
        return bytes(payload)
    if isinstance(payload, str):
        return payload.encode("utf-8")
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def _artifact_dir(artifact_type: str) -> str:
    return {
        "quiz": "quizzes",
        "mindmap": "mindmaps",
        "html": "mindmaps",
        "podcast": "podcasts",
        "transcript": "podcasts",
        "presentation": "presentations",
        "report": "reports",
        "chart": "charts",
    }.get(artifact_type, "reports")


_artifact_service: ArtifactService | None = None


def get_artifact_service() -> ArtifactService:
    global _artifact_service
    if _artifact_service is None:
        _artifact_service = ArtifactService()
    return _artifact_service

