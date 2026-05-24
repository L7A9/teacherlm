from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field
from teacherlm_core.schemas.learner_state import LearnerState


QuestionType = Literal["mcq", "true_false", "fill_blank", "short_answer"]
BloomLevel = Literal["remember", "understand", "apply", "analyze"]


class KnowledgeCheckStartRequest(BaseModel):
    concept_id: uuid.UUID | None = None
    phase_id: uuid.UUID | None = None
    objective_id: uuid.UUID | None = None
    count: int = Field(default=1, ge=1, le=5)
    question_types: list[QuestionType] | None = None
    options: dict[str, Any] = Field(default_factory=dict)


class KnowledgeCheckQuestion(BaseModel):
    id: uuid.UUID
    concept_id: uuid.UUID
    concept_name: str
    phase_id: uuid.UUID | None = None
    objective_id: uuid.UUID | None = None
    question_type: QuestionType
    bloom_level: BloomLevel
    prompt: str
    options: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class KnowledgeCheckStartResponse(BaseModel):
    checks: list[KnowledgeCheckQuestion]
    learner_state: LearnerState


class KnowledgeCheckSubmitRequest(BaseModel):
    answer: Any
    options: dict[str, Any] = Field(default_factory=dict)


class KnowledgeCheckResult(BaseModel):
    check_id: uuid.UUID
    concept_id: uuid.UUID
    concept_name: str
    question_index: int | None = None
    score: float
    is_correct: bool
    feedback: str
    evidence_strength: str
    mastery_delta: float
    remediation_paths: list[dict[str, Any]] = Field(default_factory=list)


class KnowledgeCheckSubmitResponse(BaseModel):
    result: KnowledgeCheckResult
    learner_state: LearnerState


class QuizAttemptQuestion(BaseModel):
    type: QuestionType
    bloom_level: BloomLevel | str | None = None
    question: str
    options: list[str] = Field(default_factory=list)
    correct_index: int | None = None
    answer: str | bool | None = None
    accepted_answers: list[str] = Field(default_factory=list)
    explanation: str = ""
    concept_id: uuid.UUID | None = None
    concept: str | None = None
    phase_id: uuid.UUID | None = None
    objective_id: uuid.UUID | None = None
    source_chunk_id: str | None = None


class QuizAttemptAnswer(BaseModel):
    question_index: int = Field(ge=0)
    answer: Any


class QuizAttemptRequest(BaseModel):
    questions: list[QuizAttemptQuestion] = Field(default_factory=list)
    answers: list[QuizAttemptAnswer] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class QuizAttemptResponse(BaseModel):
    results: list[KnowledgeCheckResult]
    learner_state: LearnerState
    score: float
    total: int
