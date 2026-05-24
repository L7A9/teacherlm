"""coursebuilder agent

Revision ID: 0010_coursebuilder
Revises: 0009_course_knowledge_graph
Create Date: 2026-05-24
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0010_coursebuilder"
down_revision: str | None = "0009_course_knowledge_graph"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json_default(value: str) -> sa.TextClause:
    return sa.text(f"'{value}'::jsonb")


def upgrade() -> None:
    op.create_table(
        "coursebuilder_courses",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("learning_objectives", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("prerequisites", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="queued"),
        sa.Column("language", sa.String(length=32), nullable=True),
        sa.Column("generation_metadata", postgresql.JSONB(), nullable=False, server_default=_json_default("{}")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_coursebuilder_courses_conversation_id", "coursebuilder_courses", ["conversation_id"])
    op.create_index("ix_coursebuilder_courses_status", "coursebuilder_courses", ["status"])

    op.create_table(
        "coursebuilder_chapters",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("course_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("coursebuilder_courses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("is_locked", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("unlock_rule", postgresql.JSONB(), nullable=False, server_default=_json_default("{}")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_coursebuilder_chapters_course_id", "coursebuilder_chapters", ["course_id"])
    op.create_index("ix_coursebuilder_chapters_conversation_id", "coursebuilder_chapters", ["conversation_id"])

    op.create_table(
        "coursebuilder_lessons",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("chapter_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("coursebuilder_chapters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("course_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("coursebuilder_courses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("learning_objectives", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("support_status", sa.String(length=32), nullable=False, server_default="supported"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_coursebuilder_lessons_chapter_id", "coursebuilder_lessons", ["chapter_id"])
    op.create_index("ix_coursebuilder_lessons_course_id", "coursebuilder_lessons", ["course_id"])
    op.create_index("ix_coursebuilder_lessons_conversation_id", "coursebuilder_lessons", ["conversation_id"])

    op.create_table(
        "coursebuilder_lesson_blocks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("lesson_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("coursebuilder_lessons.id", ondelete="CASCADE"), nullable=False),
        sa.Column("block_type", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("data_json", postgresql.JSONB(), nullable=False, server_default=_json_default("{}")),
        sa.Column("source_citations", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("validation_status", sa.String(length=32), nullable=False, server_default="supported"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_coursebuilder_lesson_blocks_lesson_id", "coursebuilder_lesson_blocks", ["lesson_id"])

    op.create_table(
        "coursebuilder_quizzes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("chapter_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("coursebuilder_chapters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("course_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("coursebuilder_courses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("pass_score", sa.Float(), nullable=False, server_default="0.7"),
        sa.Column("question_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_coursebuilder_quizzes_chapter_id", "coursebuilder_quizzes", ["chapter_id"])
    op.create_index("ix_coursebuilder_quizzes_course_id", "coursebuilder_quizzes", ["course_id"])

    op.create_table(
        "coursebuilder_quiz_questions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("quiz_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("coursebuilder_quizzes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chapter_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("coursebuilder_chapters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("question_type", sa.String(length=32), nullable=False, server_default="mcq"),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("options", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("answer_key", postgresql.JSONB(), nullable=False, server_default=_json_default("{}")),
        sa.Column("explanation", sa.Text(), nullable=False, server_default=""),
        sa.Column("source_citations", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_coursebuilder_quiz_questions_quiz_id", "coursebuilder_quiz_questions", ["quiz_id"])
    op.create_index("ix_coursebuilder_quiz_questions_chapter_id", "coursebuilder_quiz_questions", ["chapter_id"])

    op.create_table(
        "coursebuilder_chapter_attempts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("course_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("coursebuilder_courses.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chapter_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("coursebuilder_chapters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("quiz_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("coursebuilder_quizzes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("passed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("answers", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("feedback", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_coursebuilder_chapter_attempts_conversation_id", "coursebuilder_chapter_attempts", ["conversation_id"])
    op.create_index("ix_coursebuilder_chapter_attempts_course_id", "coursebuilder_chapter_attempts", ["course_id"])
    op.create_index("ix_coursebuilder_chapter_attempts_chapter_id", "coursebuilder_chapter_attempts", ["chapter_id"])
    op.create_index("ix_coursebuilder_chapter_attempts_quiz_id", "coursebuilder_chapter_attempts", ["quiz_id"])

    op.create_table(
        "coursebuilder_progress_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("course_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("coursebuilder_courses.id", ondelete="CASCADE"), nullable=True),
        sa.Column("stage", sa.String(length=64), nullable=False),
        sa.Column("message", sa.Text(), nullable=False, server_default=""),
        sa.Column("percent", sa.Float(), nullable=False, server_default="0"),
        sa.Column("event_metadata", postgresql.JSONB(), nullable=False, server_default=_json_default("{}")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_coursebuilder_progress_events_conversation_id", "coursebuilder_progress_events", ["conversation_id"])
    op.create_index("ix_coursebuilder_progress_events_course_id", "coursebuilder_progress_events", ["course_id"])


def downgrade() -> None:
    for index, table in [
        ("ix_coursebuilder_progress_events_course_id", "coursebuilder_progress_events"),
        ("ix_coursebuilder_progress_events_conversation_id", "coursebuilder_progress_events"),
        ("ix_coursebuilder_chapter_attempts_quiz_id", "coursebuilder_chapter_attempts"),
        ("ix_coursebuilder_chapter_attempts_chapter_id", "coursebuilder_chapter_attempts"),
        ("ix_coursebuilder_chapter_attempts_course_id", "coursebuilder_chapter_attempts"),
        ("ix_coursebuilder_chapter_attempts_conversation_id", "coursebuilder_chapter_attempts"),
        ("ix_coursebuilder_quiz_questions_chapter_id", "coursebuilder_quiz_questions"),
        ("ix_coursebuilder_quiz_questions_quiz_id", "coursebuilder_quiz_questions"),
        ("ix_coursebuilder_quizzes_course_id", "coursebuilder_quizzes"),
        ("ix_coursebuilder_quizzes_chapter_id", "coursebuilder_quizzes"),
        ("ix_coursebuilder_lesson_blocks_lesson_id", "coursebuilder_lesson_blocks"),
        ("ix_coursebuilder_lessons_conversation_id", "coursebuilder_lessons"),
        ("ix_coursebuilder_lessons_course_id", "coursebuilder_lessons"),
        ("ix_coursebuilder_lessons_chapter_id", "coursebuilder_lessons"),
        ("ix_coursebuilder_chapters_conversation_id", "coursebuilder_chapters"),
        ("ix_coursebuilder_chapters_course_id", "coursebuilder_chapters"),
        ("ix_coursebuilder_courses_status", "coursebuilder_courses"),
        ("ix_coursebuilder_courses_conversation_id", "coursebuilder_courses"),
    ]:
        op.drop_index(index, table_name=table)
    for table in [
        "coursebuilder_progress_events",
        "coursebuilder_chapter_attempts",
        "coursebuilder_quiz_questions",
        "coursebuilder_quizzes",
        "coursebuilder_lesson_blocks",
        "coursebuilder_lessons",
        "coursebuilder_chapters",
        "coursebuilder_courses",
    ]:
        op.drop_table(table)
