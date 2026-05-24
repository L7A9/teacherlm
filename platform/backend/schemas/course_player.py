from __future__ import annotations

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field
from teacherlm_core.schemas.learner_state import LearnerState

from schemas.knowledge_check import KnowledgeCheckQuestion, KnowledgeCheckResult


BlockType = Literal["definition", "explanation", "example", "procedure", "formula", "summary", "quiz"]


class CourseLessonBlockRead(BaseModel):
    id: uuid.UUID
    lesson_id: uuid.UUID
    block_type: BlockType | str
    title: str
    content: str
    order_index: int
    source_chunk_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CourseLessonRead(BaseModel):
    id: uuid.UUID
    chapter_id: uuid.UUID
    objective_id: uuid.UUID | None = None
    title: str
    summary: str
    order_index: int
    concept_ids: list[uuid.UUID] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    prerequisite_concept_ids: list[uuid.UUID] = Field(default_factory=list)
    next_concept_ids: list[uuid.UUID] = Field(default_factory=list)
    related_example_ids: list[uuid.UUID] = Field(default_factory=list)
    remediation_objective_ids: list[uuid.UUID] = Field(default_factory=list)
    graph_hints: dict[str, Any] = Field(default_factory=dict)
    blocks: list[CourseLessonBlockRead] = Field(default_factory=list)


class ChapterQuizRead(BaseModel):
    id: uuid.UUID
    chapter_id: uuid.UUID
    pass_score: float
    question_ids: list[uuid.UUID] = Field(default_factory=list)
    questions: list[KnowledgeCheckQuestion] = Field(default_factory=list)


class CourseChapterRead(BaseModel):
    id: uuid.UUID
    phase_id: uuid.UUID | None = None
    title: str
    summary: str
    order_index: int
    objective_ids: list[uuid.UUID] = Field(default_factory=list)
    concept_ids: list[uuid.UUID] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    state: Literal["locked", "available", "completed"]
    best_score: float
    attempts: int
    soft_lock_overridden: bool
    progress: float
    lessons: list[CourseLessonRead] = Field(default_factory=list)
    quiz: ChapterQuizRead | None = None


class CoursePlayerRead(BaseModel):
    conversation_id: uuid.UUID
    chapters: list[CourseChapterRead] = Field(default_factory=list)
    learner_state: LearnerState
    course_status: Literal["waiting_for_files", "ready"] = "ready"
    pending_file_count: int = 0
    total_file_count: int = 0


class CoursePlayerUnlockResponse(BaseModel):
    chapter: CourseChapterRead
    learner_state: LearnerState


class ChapterQuizAnswer(BaseModel):
    check_id: uuid.UUID
    answer: Any


class ChapterQuizSubmitRequest(BaseModel):
    answers: list[ChapterQuizAnswer] = Field(default_factory=list)
    options: dict[str, Any] = Field(default_factory=dict)


class ChapterQuizSubmitResponse(BaseModel):
    chapter: CourseChapterRead
    results: list[KnowledgeCheckResult]
    score: float
    passed: bool
    learner_state: LearnerState
