from __future__ import annotations

import uuid


def new_coursebuilder_generation_id() -> str:
    return uuid.uuid4().hex


def coursebuilder_job_id(conversation_id: uuid.UUID | str, generation_id: str | None) -> str:
    suffix = str(generation_id or "manual")
    return f"coursebuilder:{conversation_id}:{suffix}"
