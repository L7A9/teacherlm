from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


CourseBuilderStatus = Literal[
    "queued",
    "analyzing",
    "generating_outline",
    "generating_chapters",
    "generating_lessons",
    "generating_quizzes",
    "validating",
    "ready",
    "failed",
]


class CourseBuilderCitation(BaseModel):
    chunk_id: str
    source: str = ""
    page_start: int | None = None
    page_end: int | None = None
    section: str = ""
    snippet: str = ""


class CourseBuilderLessonBlockRead(BaseModel):
    id: uuid.UUID
    lesson_id: uuid.UUID
    block_type: str
    title: str
    content: str
    order_index: int
    data_json: dict[str, Any] = Field(default_factory=dict)
    source_citations: list[CourseBuilderCitation] = Field(default_factory=list)
    validation_status: str


class CourseBuilderLessonRead(BaseModel):
    id: uuid.UUID
    chapter_id: uuid.UUID
    title: str
    order_index: int
    learning_objectives: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    support_status: str
    blocks: list[CourseBuilderLessonBlockRead] = Field(default_factory=list)


class CourseBuilderQuizQuestionRead(BaseModel):
    id: uuid.UUID
    quiz_id: uuid.UUID
    chapter_id: uuid.UUID
    question_type: str
    prompt: str
    options: list[str] = Field(default_factory=list)
    explanation: str = ""
    order_index: int
    source_citations: list[CourseBuilderCitation] = Field(default_factory=list)


class CourseBuilderQuizRead(BaseModel):
    id: uuid.UUID
    chapter_id: uuid.UUID
    pass_score: float
    question_count: int
    source_chunk_ids: list[str] = Field(default_factory=list)
    questions: list[CourseBuilderQuizQuestionRead] = Field(default_factory=list)


class CourseBuilderChapterRead(BaseModel):
    id: uuid.UUID
    course_id: uuid.UUID
    title: str
    description: str
    order_index: int
    summary: str
    source_chunk_ids: list[str] = Field(default_factory=list)
    is_locked: bool
    unlock_rule: dict[str, Any] = Field(default_factory=dict)
    best_score: float = 0.0
    attempts: int = 0
    completed: bool = False
    lessons: list[CourseBuilderLessonRead] = Field(default_factory=list)
    quiz: CourseBuilderQuizRead | None = None


class CourseBuilderProgressEventRead(BaseModel):
    id: uuid.UUID
    conversation_id: uuid.UUID
    course_id: uuid.UUID | None = None
    stage: str
    message: str
    percent: float
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class CourseBuilderRead(BaseModel):
    id: uuid.UUID | None = None
    conversation_id: uuid.UUID
    title: str = ""
    description: str = ""
    learning_objectives: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    status: CourseBuilderStatus | str = "queued"
    language: str | None = None
    error: str | None = None
    generation_metadata: dict[str, Any] = Field(default_factory=dict)
    chapters: list[CourseBuilderChapterRead] = Field(default_factory=list)
    progress_events: list[CourseBuilderProgressEventRead] = Field(default_factory=list)
    pending_file_count: int = 0
    total_file_count: int = 0


class CourseBuilderGenerateRequest(BaseModel):
    options: dict[str, Any] = Field(default_factory=dict)


class CourseBuilderQuizAnswer(BaseModel):
    question_id: uuid.UUID
    answer: str | int


class CourseBuilderQuizSubmitRequest(BaseModel):
    answers: list[CourseBuilderQuizAnswer] = Field(default_factory=list)


class CourseBuilderQuizResult(BaseModel):
    question_id: uuid.UUID
    is_correct: bool
    correct_index: int
    selected_index: int | None = None
    feedback: str


class CourseBuilderQuizSubmitResponse(BaseModel):
    chapter: CourseBuilderChapterRead
    score: float
    passed: bool
    results: list[CourseBuilderQuizResult]
    course: CourseBuilderRead
