"""discussion review windows

Revision ID: 0007_discussion_reviews
Revises: 0006_learning_map
Create Date: 2026-05-23
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0007_discussion_reviews"
down_revision: str | None = "0006_learning_map"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "learning_review_windows",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="pending"),
        sa.Column("answered_question_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("user_message_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("assistant_message_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("concept_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("objective_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("phase_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("generated_check_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("answer_count", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("due_count", sa.Integer(), nullable=False, server_default="10"),
        sa.Column("snooze_until_count", sa.Integer(), nullable=True),
        sa.Column("review_metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_learning_review_windows_conversation_id", "learning_review_windows", ["conversation_id"])
    op.create_index("ix_learning_review_windows_status", "learning_review_windows", ["status"])

    op.create_table(
        "answered_course_questions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "user_message_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "assistant_message_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("messages.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "review_window_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("learning_review_windows.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("concept_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("objective_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("phase_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("question_metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_answered_course_questions_conversation_id", "answered_course_questions", ["conversation_id"])
    op.create_index("ix_answered_course_questions_user_message_id", "answered_course_questions", ["user_message_id"])
    op.create_index("ix_answered_course_questions_assistant_message_id", "answered_course_questions", ["assistant_message_id"])
    op.create_index("ix_answered_course_questions_review_window_id", "answered_course_questions", ["review_window_id"])


def downgrade() -> None:
    op.drop_index("ix_answered_course_questions_review_window_id", table_name="answered_course_questions")
    op.drop_index("ix_answered_course_questions_assistant_message_id", table_name="answered_course_questions")
    op.drop_index("ix_answered_course_questions_user_message_id", table_name="answered_course_questions")
    op.drop_index("ix_answered_course_questions_conversation_id", table_name="answered_course_questions")
    op.drop_table("answered_course_questions")
    op.drop_index("ix_learning_review_windows_status", table_name="learning_review_windows")
    op.drop_index("ix_learning_review_windows_conversation_id", table_name="learning_review_windows")
    op.drop_table("learning_review_windows")
