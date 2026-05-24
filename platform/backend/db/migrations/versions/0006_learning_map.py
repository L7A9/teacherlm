"""course learning map

Revision ID: 0006_learning_map
Revises: 0005_knowledge_checks
Create Date: 2026-05-23
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0006_learning_map"
down_revision: str | None = "0005_knowledge_checks"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "course_learning_phases",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("phase_key", sa.String(length=256), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("source_file_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("source_section_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("phase_metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_course_learning_phases_conversation_id", "course_learning_phases", ["conversation_id"])
    op.create_index(
        "uq_course_learning_phases_conversation_key",
        "course_learning_phases",
        ["conversation_id", "phase_key"],
        unique=True,
    )

    op.create_table(
        "course_learning_objectives",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "phase_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("course_learning_phases.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("objective_key", sa.String(length=256), nullable=False),
        sa.Column("objective_text", sa.Text(), nullable=False),
        sa.Column("bloom_level", sa.String(length=32), nullable=False, server_default="understand"),
        sa.Column("order_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("concept_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("source_file_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("source_section_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("objective_metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_course_learning_objectives_conversation_id", "course_learning_objectives", ["conversation_id"])
    op.create_index("ix_course_learning_objectives_phase_id", "course_learning_objectives", ["phase_id"])
    op.create_index(
        "uq_course_learning_objectives_conversation_key",
        "course_learning_objectives",
        ["conversation_id", "objective_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_course_learning_objectives_conversation_key", table_name="course_learning_objectives")
    op.drop_index("ix_course_learning_objectives_phase_id", table_name="course_learning_objectives")
    op.drop_index("ix_course_learning_objectives_conversation_id", table_name="course_learning_objectives")
    op.drop_table("course_learning_objectives")
    op.drop_index("uq_course_learning_phases_conversation_key", table_name="course_learning_phases")
    op.drop_index("ix_course_learning_phases_conversation_id", table_name="course_learning_phases")
    op.drop_table("course_learning_phases")
