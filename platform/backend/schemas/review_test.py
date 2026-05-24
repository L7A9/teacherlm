from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field
from teacherlm_core.schemas.learner_state import LearnerState

from schemas.knowledge_check import KnowledgeCheckQuestion, KnowledgeCheckResult


class ReviewWindowSummary(BaseModel):
    id: uuid.UUID
    status: str
    answered_count: int
    due_count: int
    snooze_until_count: int | None = None
    concept_ids: list[uuid.UUID] = Field(default_factory=list)
    objective_ids: list[uuid.UUID] = Field(default_factory=list)
    phase_ids: list[uuid.UUID] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    generated_check_ids: list[uuid.UUID] = Field(default_factory=list)


class ReviewTestStatusResponse(BaseModel):
    answered_count: int
    pending_count: int
    due: bool
    window: ReviewWindowSummary | None = None
    learner_state: LearnerState | None = None


class ReviewTestStartRequest(BaseModel):
    options: dict[str, Any] = Field(default_factory=dict)


class ReviewTestStartResponse(BaseModel):
    window: ReviewWindowSummary
    checks: list[KnowledgeCheckQuestion]
    learner_state: LearnerState


class ReviewTestAnswer(BaseModel):
    check_id: uuid.UUID
    answer: Any


class ReviewTestSubmitRequest(BaseModel):
    answers: list[ReviewTestAnswer] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class ReviewTestSubmitResponse(BaseModel):
    window: ReviewWindowSummary
    results: list[KnowledgeCheckResult]
    learner_state: LearnerState


class ReviewTestActionResponse(BaseModel):
    window: ReviewWindowSummary
    answered_count: int
    due: bool
