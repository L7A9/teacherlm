from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


FileStatus = Literal["uploaded", "parsing", "chunking", "embedding", "ready", "failed"]


class UploadedFileRead(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    conversation_id: uuid.UUID
    filename: str
    file_id: str
    status: FileStatus
    chunk_count: int
    parsed_markdown_path: str | None = None
    error: str | None = None
    created_at: datetime


class UploadedFileList(BaseModel):
    items: list[UploadedFileRead]
    total: int
