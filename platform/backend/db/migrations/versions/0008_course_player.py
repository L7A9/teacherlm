"""course player

Revision ID: 0008_course_player
Revises: 0007_discussion_reviews
Create Date: 2026-05-23
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0008_course_player"
down_revision: str | None = "0007_discussion_reviews"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json_default(value: str) -> sa.TextClause:
    return sa.text(f"'{value}'::jsonb")


def upgrade() -> None:
    op.create_table(
        "course_chapters",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("phase_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("course_learning_phases.id", ondelete="SET NULL"), nullable=True),
        sa.Column("chapter_key", sa.String(length=256), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("objective_ids", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("concept_ids", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("state_metadata", postgresql.JSONB(), nullable=False, server_default=_json_default("{}")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_course_chapters_conversation_id", "course_chapters", ["conversation_id"])
    op.create_index("ix_course_chapters_phase_id", "course_chapters", ["phase_id"])
    op.create_index("uq_course_chapters_conversation_key", "course_chapters", ["conversation_id", "chapter_key"], unique=True)

    op.create_table(
        "course_lessons",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chapter_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("course_chapters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("objective_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("course_learning_objectives.id", ondelete="SET NULL"), nullable=True),
        sa.Column("lesson_key", sa.String(length=256), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("concept_ids", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("lesson_metadata", postgresql.JSONB(), nullable=False, server_default=_json_default("{}")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_course_lessons_conversation_id", "course_lessons", ["conversation_id"])
    op.create_index("ix_course_lessons_chapter_id", "course_lessons", ["chapter_id"])
    op.create_index("ix_course_lessons_objective_id", "course_lessons", ["objective_id"])
    op.create_index("uq_course_lessons_conversation_key", "course_lessons", ["conversation_id", "lesson_key"], unique=True)

    op.create_table(
        "course_lesson_blocks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("lesson_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("course_lessons.id", ondelete="CASCADE"), nullable=False),
        sa.Column("block_type", sa.String(length=32), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False, server_default=""),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("block_metadata", postgresql.JSONB(), nullable=False, server_default=_json_default("{}")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_course_lesson_blocks_conversation_id", "course_lesson_blocks", ["conversation_id"])
    op.create_index("ix_course_lesson_blocks_lesson_id", "course_lesson_blocks", ["lesson_id"])

    op.create_table(
        "chapter_quizzes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chapter_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("course_chapters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("question_ids", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("pass_score", sa.Float(), nullable=False, server_default="0.7"),
        sa.Column("quiz_metadata", postgresql.JSONB(), nullable=False, server_default=_json_default("{}")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chapter_quizzes_conversation_id", "chapter_quizzes", ["conversation_id"])
    op.create_index("ix_chapter_quizzes_chapter_id", "chapter_quizzes", ["chapter_id"])

    op.create_table(
        "chapter_attempts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("chapter_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("course_chapters.id", ondelete="CASCADE"), nullable=False),
        sa.Column("quiz_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("chapter_quizzes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("passed", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("answers", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("results", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("attempt_metadata", postgresql.JSONB(), nullable=False, server_default=_json_default("{}")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_chapter_attempts_conversation_id", "chapter_attempts", ["conversation_id"])
    op.create_index("ix_chapter_attempts_chapter_id", "chapter_attempts", ["chapter_id"])
    op.create_index("ix_chapter_attempts_quiz_id", "chapter_attempts", ["quiz_id"])


def downgrade() -> None:
    op.drop_index("ix_chapter_attempts_quiz_id", table_name="chapter_attempts")
    op.drop_index("ix_chapter_attempts_chapter_id", table_name="chapter_attempts")
    op.drop_index("ix_chapter_attempts_conversation_id", table_name="chapter_attempts")
    op.drop_table("chapter_attempts")
    op.drop_index("ix_chapter_quizzes_chapter_id", table_name="chapter_quizzes")
    op.drop_index("ix_chapter_quizzes_conversation_id", table_name="chapter_quizzes")
    op.drop_table("chapter_quizzes")
    op.drop_index("ix_course_lesson_blocks_lesson_id", table_name="course_lesson_blocks")
    op.drop_index("ix_course_lesson_blocks_conversation_id", table_name="course_lesson_blocks")
    op.drop_table("course_lesson_blocks")
    op.drop_index("uq_course_lessons_conversation_key", table_name="course_lessons")
    op.drop_index("ix_course_lessons_objective_id", table_name="course_lessons")
    op.drop_index("ix_course_lessons_chapter_id", table_name="course_lessons")
    op.drop_index("ix_course_lessons_conversation_id", table_name="course_lessons")
    op.drop_table("course_lessons")
    op.drop_index("uq_course_chapters_conversation_key", table_name="course_chapters")
    op.drop_index("ix_course_chapters_phase_id", table_name="course_chapters")
    op.drop_index("ix_course_chapters_conversation_id", table_name="course_chapters")
    op.drop_table("course_chapters")
