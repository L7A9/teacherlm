from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
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
    # statuses: uploaded | parsing | chunking | embedding | ready | failed
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
