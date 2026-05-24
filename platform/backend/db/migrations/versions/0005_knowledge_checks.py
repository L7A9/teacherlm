"""knowledge checks and attempts

Revision ID: 0005_knowledge_checks
Revises: 0004_course_concepts
Create Date: 2026-05-21
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0005_knowledge_checks"
down_revision: str | None = "0004_course_concepts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "knowledge_checks",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "concept_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("course_concepts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("question_type", sa.String(length=32), nullable=False),
        sa.Column("bloom_level", sa.String(length=32), nullable=False, server_default="understand"),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("options", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("answer_key", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("rubric", sa.Text(), nullable=False, server_default=""),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("check_metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_knowledge_checks_conversation_id", "knowledge_checks", ["conversation_id"])
    op.create_index("ix_knowledge_checks_concept_id", "knowledge_checks", ["concept_id"])

    op.create_table(
        "knowledge_attempts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "check_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("knowledge_checks.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "concept_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("course_concepts.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("answer", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("score", sa.Float(), nullable=False, server_default="0"),
        sa.Column("is_correct", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("feedback", sa.Text(), nullable=False, server_default=""),
        sa.Column("evidence_strength", sa.String(length=32), nullable=False, server_default="weak"),
        sa.Column("mastery_delta", sa.Float(), nullable=False, server_default="0"),
        sa.Column("attempt_metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_knowledge_attempts_check_id", "knowledge_attempts", ["check_id"])
    op.create_index("ix_knowledge_attempts_conversation_id", "knowledge_attempts", ["conversation_id"])
    op.create_index("ix_knowledge_attempts_concept_id", "knowledge_attempts", ["concept_id"])


def downgrade() -> None:
    op.drop_index("ix_knowledge_attempts_concept_id", table_name="knowledge_attempts")
    op.drop_index("ix_knowledge_attempts_conversation_id", table_name="knowledge_attempts")
    op.drop_index("ix_knowledge_attempts_check_id", table_name="knowledge_attempts")
    op.drop_table("knowledge_attempts")
    op.drop_index("ix_knowledge_checks_concept_id", table_name="knowledge_checks")
    op.drop_index("ix_knowledge_checks_conversation_id", table_name="knowledge_checks")
    op.drop_table("knowledge_checks")
