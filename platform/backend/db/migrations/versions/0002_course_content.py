"""course content model

Revision ID: 0002_course_content
Revises: 0001_initial
Create Date: 2026-05-13
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0002_course_content"
down_revision: str | None = "0003_add_uploaded_file_summary"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "course_documents",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "uploaded_file_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("uploaded_files.id", ondelete="CASCADE"),
            nullable=False,
            unique=True,
        ),
        sa.Column("source_file_id", sa.String(length=1024), nullable=False),
        sa.Column("source_filename", sa.String(length=512), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("raw_markdown_path", sa.String(length=1024), nullable=True),
        sa.Column("cleaned_text_path", sa.String(length=1024), nullable=True),
        sa.Column("text_hash", sa.String(length=64), nullable=False),
        sa.Column("course_metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_course_documents_conversation_id", "course_documents", ["conversation_id"])
    op.create_index("ix_course_documents_uploaded_file_id", "course_documents", ["uploaded_file_id"])

    op.create_table(
        "course_sections",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("course_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("parent_section_id", postgresql.UUID(as_uuid=True), nullable=True),
        sa.Column("level", sa.Integer(), nullable=False),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("heading_path", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("order_index", sa.Integer(), nullable=False),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False, server_default=""),
        sa.Column("key_concepts", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("equations", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("tables", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("timeline_events", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("section_metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_course_sections_conversation_id", "course_sections", ["conversation_id"])
    op.create_index("ix_course_sections_document_id", "course_sections", ["document_id"])

    op.create_table(
        "search_chunks",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column(
            "conversation_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("conversations.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("course_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "section_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("course_sections.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("source_filename", sa.String(length=512), nullable=False),
        sa.Column("source_file_id", sa.String(length=1024), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False),
        sa.Column("token_count", sa.Integer(), nullable=False),
        sa.Column("prev_chunk_id", sa.String(length=64), nullable=True),
        sa.Column("next_chunk_id", sa.String(length=64), nullable=True),
        sa.Column("page_start", sa.Integer(), nullable=True),
        sa.Column("page_end", sa.Integer(), nullable=True),
        sa.Column("heading_path", postgresql.JSONB(), nullable=False, server_default=sa.text("'[]'::jsonb")),
        sa.Column("chunk_metadata", postgresql.JSONB(), nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_search_chunks_conversation_id", "search_chunks", ["conversation_id"])
    op.create_index("ix_search_chunks_document_id", "search_chunks", ["document_id"])
    op.create_index("ix_search_chunks_section_id", "search_chunks", ["section_id"])
    op.create_index("ix_search_chunks_source_file_id", "search_chunks", ["source_file_id"])


def downgrade() -> None:
    op.drop_index("ix_search_chunks_source_file_id", table_name="search_chunks")
    op.drop_index("ix_search_chunks_section_id", table_name="search_chunks")
    op.drop_index("ix_search_chunks_document_id", table_name="search_chunks")
    op.drop_index("ix_search_chunks_conversation_id", table_name="search_chunks")
    op.drop_table("search_chunks")
    op.drop_index("ix_course_sections_document_id", table_name="course_sections")
    op.drop_index("ix_course_sections_conversation_id", table_name="course_sections")
    op.drop_table("course_sections")
    op.drop_index("ix_course_documents_uploaded_file_id", table_name="course_documents")
    op.drop_index("ix_course_documents_conversation_id", table_name="course_documents")
    op.drop_table("course_documents")
