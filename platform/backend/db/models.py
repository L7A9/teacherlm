from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_uuid() -> uuid.UUID:
    return uuid.uuid4()


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    title: Mapped[str] = mapped_column(String(512), default="Untitled conversation")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    messages: Mapped[list[Message]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    files: Mapped[list[UploadedFile]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )
    course_documents: Mapped[list[CourseDocumentRecord]] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
    )
    learner_state: Mapped[LearnerStateRecord | None] = relationship(
        back_populates="conversation",
        cascade="all, delete-orphan",
        uselist=False,
    )


class AppRuntimeSettingsRecord(Base):
    __tablename__ = "app_runtime_settings"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, default="global")
    llm_enabled: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    llm_provider: Mapped[str] = mapped_column(String(32), default="ollama", nullable=False)
    llm_model: Mapped[str] = mapped_column(String(256), default="", nullable=False)
    llm_base_url: Mapped[str] = mapped_column(String(1024), default="", nullable=False)
    llm_api_key_encrypted: Mapped[str | None] = mapped_column(Text)
    llama_cloud_api_key_encrypted: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role: Mapped[str] = mapped_column(String(32), nullable=False)  # user | assistant | system
    content: Mapped[str] = mapped_column(Text, nullable=False)
    generator_id: Mapped[str | None] = mapped_column(String(128))
    output_type: Mapped[str | None] = mapped_column(String(64))
    artifacts: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class AnsweredCourseQuestionRecord(Base):
    __tablename__ = "answered_course_questions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    assistant_message_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("messages.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    review_window_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("learning_review_windows.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    concept_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    objective_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    phase_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    question_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class LearningReviewWindowRecord(Base):
    __tablename__ = "learning_review_windows"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(32), default="pending", nullable=False, index=True)
    answered_question_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    user_message_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    assistant_message_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    concept_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    objective_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    phase_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    generated_check_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    answer_count: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    due_count: Mapped[int] = mapped_column(Integer, default=10, nullable=False)
    snooze_until_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    review_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class UploadedFile(Base):
    __tablename__ = "uploaded_files"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_id: Mapped[str] = mapped_column(String(256), nullable=False)  # MinIO object key
    status: Mapped[str] = mapped_column(String(32), default="uploaded", nullable=False)
    # statuses: uploaded | parsing | chunking | extracting_concepts | building_course | embedding | ready | failed
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    parsed_markdown_path: Mapped[str | None] = mapped_column(String(1024))
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="files")
    course_document: Mapped[CourseDocumentRecord | None] = relationship(
        back_populates="uploaded_file",
        cascade="all, delete-orphan",
        uselist=False,
    )


class CourseDocumentRecord(Base):
    __tablename__ = "course_documents"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    uploaded_file_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("uploaded_files.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )
    source_file_id: Mapped[str] = mapped_column(String(1024), nullable=False)
    source_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    raw_markdown_path: Mapped[str | None] = mapped_column(String(1024))
    cleaned_text_path: Mapped[str | None] = mapped_column(String(1024))
    text_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    course_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    conversation: Mapped[Conversation] = relationship(back_populates="course_documents")
    uploaded_file: Mapped[UploadedFile] = relationship(back_populates="course_document")
    sections: Mapped[list[CourseSectionRecord]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="CourseSectionRecord.order_index",
    )
    chunks: Mapped[list[SearchChunkRecord]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan",
        order_by="SearchChunkRecord.chunk_index",
    )


class CourseSectionRecord(Base):
    __tablename__ = "course_sections"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    parent_section_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    level: Mapped[int] = mapped_column(Integer, nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    heading_path: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, nullable=False)
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    key_concepts: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    equations: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    tables: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    timeline_events: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    section_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    document: Mapped[CourseDocumentRecord] = relationship(back_populates="sections")
    chunks: Mapped[list[SearchChunkRecord]] = relationship(
        back_populates="section",
        cascade="all, delete-orphan",
        order_by="SearchChunkRecord.chunk_index",
    )


class SearchChunkRecord(Base):
    __tablename__ = "search_chunks"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    section_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_sections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    source_file_id: Mapped[str] = mapped_column(String(1024), nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    prev_chunk_id: Mapped[str | None] = mapped_column(String(64))
    next_chunk_id: Mapped[str | None] = mapped_column(String(64))
    page_start: Mapped[int | None] = mapped_column(Integer)
    page_end: Mapped[int | None] = mapped_column(Integer)
    heading_path: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    document: Mapped[CourseDocumentRecord] = relationship(back_populates="chunks")
    section: Mapped[CourseSectionRecord] = relationship(back_populates="chunks")


class CourseConceptRecord(Base):
    __tablename__ = "course_concepts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    canonical_key: Mapped[str] = mapped_column(String(256), nullable=False)
    canonical_name: Mapped[str] = mapped_column(String(256), nullable=False)
    aliases: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    bloom_level: Mapped[str] = mapped_column(String(32), default="understand", nullable=False)
    importance: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    source_file_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_section_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    concept_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CourseLearningPhaseRecord(Base):
    __tablename__ = "course_learning_phases"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    phase_key: Mapped[str] = mapped_column(String(256), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    source_file_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_section_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    phase_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CourseLearningObjectiveRecord(Base):
    __tablename__ = "course_learning_objectives"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    phase_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_learning_phases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    objective_key: Mapped[str] = mapped_column(String(256), nullable=False)
    objective_text: Mapped[str] = mapped_column(Text, nullable=False)
    bloom_level: Mapped[str] = mapped_column(String(32), default="understand", nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    concept_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_file_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_section_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    objective_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CourseKnowledgeNodeRecord(Base):
    __tablename__ = "course_knowledge_nodes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    node_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    node_key: Mapped[str] = mapped_column(String(384), nullable=False)
    label: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    ref_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    node_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CourseKnowledgeEdgeRecord(Base):
    __tablename__ = "course_knowledge_edges"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    source_node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_knowledge_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    target_node_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_knowledge_nodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    relation_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    confidence: Mapped[float] = mapped_column(Float, default=0.6, nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    edge_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CourseGraphRebuildRecord(Base):
    __tablename__ = "course_graph_rebuilds"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status: Mapped[str] = mapped_column(String(32), default="completed", nullable=False, index=True)
    node_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    edge_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    rebuild_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class CourseChapterRecord(Base):
    __tablename__ = "course_chapters"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    phase_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_learning_phases.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    chapter_key: Mapped[str] = mapped_column(String(256), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    objective_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    concept_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    state_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CourseLessonRecord(Base):
    __tablename__ = "course_lessons"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_chapters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    objective_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_learning_objectives.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    lesson_key: Mapped[str] = mapped_column(String(256), nullable=False)
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    concept_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    lesson_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CourseLessonBlockRecord(Base):
    __tablename__ = "course_lesson_blocks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    lesson_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_lessons.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    block_type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    block_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class ChapterQuizRecord(Base):
    __tablename__ = "chapter_quizzes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_chapters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    pass_score: Mapped[float] = mapped_column(Float, default=0.7, nullable=False)
    quiz_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class ChapterAttemptRecord(Base):
    __tablename__ = "chapter_attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_chapters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    quiz_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("chapter_quizzes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    answers: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    results: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    attempt_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class CourseBuilderCourseRecord(Base):
    __tablename__ = "coursebuilder_courses"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    learning_objectives: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    prerequisites: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    status: Mapped[str] = mapped_column(String(32), default="queued", nullable=False, index=True)
    language: Mapped[str | None] = mapped_column(String(32), nullable=True)
    generation_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CourseBuilderChapterRecord(Base):
    __tablename__ = "coursebuilder_chapters"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coursebuilder_courses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    description: Mapped[str] = mapped_column(Text, default="", nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    summary: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    is_locked: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    unlock_rule: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CourseBuilderLessonRecord(Base):
    __tablename__ = "coursebuilder_lessons"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coursebuilder_chapters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coursebuilder_courses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(512), nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    learning_objectives: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    support_status: Mapped[str] = mapped_column(String(32), default="supported", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CourseBuilderLessonBlockRecord(Base):
    __tablename__ = "coursebuilder_lesson_blocks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    lesson_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coursebuilder_lessons.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    block_type: Mapped[str] = mapped_column(String(32), nullable=False)
    title: Mapped[str] = mapped_column(String(512), default="", nullable=False)
    content: Mapped[str] = mapped_column(Text, default="", nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    data_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    source_citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    validation_status: Mapped[str] = mapped_column(String(32), default="supported", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CourseBuilderQuizRecord(Base):
    __tablename__ = "coursebuilder_quizzes"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coursebuilder_chapters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coursebuilder_courses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    pass_score: Mapped[float] = mapped_column(Float, default=0.7, nullable=False)
    question_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )


class CourseBuilderQuizQuestionRecord(Base):
    __tablename__ = "coursebuilder_quiz_questions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    quiz_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coursebuilder_quizzes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coursebuilder_chapters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_type: Mapped[str] = mapped_column(String(32), default="mcq", nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    answer_key: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    explanation: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source_citations: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    order_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class CourseBuilderChapterAttemptRecord(Base):
    __tablename__ = "coursebuilder_chapter_attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    course_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coursebuilder_courses.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chapter_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coursebuilder_chapters.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    quiz_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coursebuilder_quizzes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    passed: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    answers: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    feedback: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class CourseBuilderProgressEventRecord(Base):
    __tablename__ = "coursebuilder_progress_events"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    course_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("coursebuilder_courses.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    stage: Mapped[str] = mapped_column(String(64), nullable=False)
    message: Mapped[str] = mapped_column(Text, default="", nullable=False)
    percent: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    event_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class KnowledgeCheckRecord(Base):
    __tablename__ = "knowledge_checks"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    concept_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_concepts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    question_type: Mapped[str] = mapped_column(String(32), nullable=False)
    bloom_level: Mapped[str] = mapped_column(String(32), default="understand", nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    options: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    answer_key: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    rubric: Mapped[str] = mapped_column(Text, default="", nullable=False)
    source_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, default=list, nullable=False)
    check_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class KnowledgeAttemptRecord(Base):
    __tablename__ = "knowledge_attempts"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=_new_uuid)
    check_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("knowledge_checks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    concept_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("course_concepts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    answer: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    score: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    is_correct: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    feedback: Mapped[str] = mapped_column(Text, default="", nullable=False)
    evidence_strength: Mapped[str] = mapped_column(String(32), default="weak", nullable=False)
    mastery_delta: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    attempt_metadata: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)


class LearnerStateRecord(Base):
    __tablename__ = "learner_states"

    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        primary_key=True,
    )
    state_json: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False
    )

    conversation: Mapped[Conversation] = relationship(back_populates="learner_state")
