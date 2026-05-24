"""canonical course concepts

Revision ID: 0004_course_concepts
Revises: 0002_course_content
Create Date: 2026-05-20
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0004_course_concepts"
down_revision: str | None = "0002_course_content"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "course_concepts",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("canonical_key", sa.String(length=256), nullable=False),
        sa.Column("canonical_name", sa.String(length=256), nullable=False),
        sa.Column("aliases", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("bloom_level", sa.String(length=32), nullable=False, server_default="understand"),
        sa.Column("importance", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("source_file_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("source_section_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("concept_metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_course_concepts_conversation_id", "course_concepts", ["conversation_id"])
    op.create_index(
        "uq_course_concepts_conversation_key",
        "course_concepts",
        ["conversation_id", "canonical_key"],
        unique=True,
    )


def downgrade() -> None:
    op.drop_index("uq_course_concepts_conversation_key", table_name="course_concepts")
    op.drop_index("ix_course_concepts_conversation_id", table_name="course_concepts")
    op.drop_table("course_concepts")
