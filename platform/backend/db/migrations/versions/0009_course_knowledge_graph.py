"""course knowledge graph

Revision ID: 0009_course_knowledge_graph
Revises: 0008_course_player
Create Date: 2026-05-23
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


revision: str = "0009_course_knowledge_graph"
down_revision: str | None = "0008_course_player"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def _json_default(value: str) -> sa.TextClause:
    return sa.text(f"'{value}'::jsonb")


def upgrade() -> None:
    op.create_table(
        "course_knowledge_nodes",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("node_type", sa.String(length=32), nullable=False),
        sa.Column("node_key", sa.String(length=384), nullable=False),
        sa.Column("label", sa.String(length=512), nullable=False),
        sa.Column("description", sa.Text(), nullable=False, server_default=""),
        sa.Column("ref_id", sa.String(length=128), nullable=True),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("node_metadata", postgresql.JSONB(), nullable=False, server_default=_json_default("{}")),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_course_knowledge_nodes_conversation_id", "course_knowledge_nodes", ["conversation_id"])
    op.create_index("ix_course_knowledge_nodes_node_type", "course_knowledge_nodes", ["node_type"])
    op.create_index("ix_course_knowledge_nodes_ref_id", "course_knowledge_nodes", ["ref_id"])
    op.create_index("ix_course_knowledge_nodes_active", "course_knowledge_nodes", ["active"])
    op.create_index("uq_course_knowledge_nodes_conversation_key", "course_knowledge_nodes", ["conversation_id", "node_type", "node_key"], unique=True)

    op.create_table(
        "course_knowledge_edges",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("source_node_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("course_knowledge_nodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("target_node_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("course_knowledge_nodes.id", ondelete="CASCADE"), nullable=False),
        sa.Column("relation_type", sa.String(length=32), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.6"),
        sa.Column("source_chunk_ids", postgresql.JSONB(), nullable=False, server_default=_json_default("[]")),
        sa.Column("edge_metadata", postgresql.JSONB(), nullable=False, server_default=_json_default("{}")),
        sa.Column("active", sa.Boolean(), nullable=False, server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_course_knowledge_edges_conversation_id", "course_knowledge_edges", ["conversation_id"])
    op.create_index("ix_course_knowledge_edges_source_node_id", "course_knowledge_edges", ["source_node_id"])
    op.create_index("ix_course_knowledge_edges_target_node_id", "course_knowledge_edges", ["target_node_id"])
    op.create_index("ix_course_knowledge_edges_relation_type", "course_knowledge_edges", ["relation_type"])
    op.create_index("ix_course_knowledge_edges_active", "course_knowledge_edges", ["active"])
    op.create_index("uq_course_knowledge_edges_conversation_relation", "course_knowledge_edges", ["conversation_id", "source_node_id", "target_node_id", "relation_type"], unique=True)

    op.create_table(
        "course_graph_rebuilds",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("conversation_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="completed"),
        sa.Column("node_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("edge_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("rebuild_metadata", postgresql.JSONB(), nullable=False, server_default=_json_default("{}")),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_course_graph_rebuilds_conversation_id", "course_graph_rebuilds", ["conversation_id"])
    op.create_index("ix_course_graph_rebuilds_status", "course_graph_rebuilds", ["status"])


def downgrade() -> None:
    op.drop_index("ix_course_graph_rebuilds_status", table_name="course_graph_rebuilds")
    op.drop_index("ix_course_graph_rebuilds_conversation_id", table_name="course_graph_rebuilds")
    op.drop_table("course_graph_rebuilds")
    op.drop_index("uq_course_knowledge_edges_conversation_relation", table_name="course_knowledge_edges")
    op.drop_index("ix_course_knowledge_edges_active", table_name="course_knowledge_edges")
    op.drop_index("ix_course_knowledge_edges_relation_type", table_name="course_knowledge_edges")
    op.drop_index("ix_course_knowledge_edges_target_node_id", table_name="course_knowledge_edges")
    op.drop_index("ix_course_knowledge_edges_source_node_id", table_name="course_knowledge_edges")
    op.drop_index("ix_course_knowledge_edges_conversation_id", table_name="course_knowledge_edges")
    op.drop_table("course_knowledge_edges")
    op.drop_index("uq_course_knowledge_nodes_conversation_key", table_name="course_knowledge_nodes")
    op.drop_index("ix_course_knowledge_nodes_active", table_name="course_knowledge_nodes")
    op.drop_index("ix_course_knowledge_nodes_ref_id", table_name="course_knowledge_nodes")
    op.drop_index("ix_course_knowledge_nodes_node_type", table_name="course_knowledge_nodes")
    op.drop_index("ix_course_knowledge_nodes_conversation_id", table_name="course_knowledge_nodes")
    op.drop_table("course_knowledge_nodes")
