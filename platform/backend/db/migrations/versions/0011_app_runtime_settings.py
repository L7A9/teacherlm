"""app runtime settings

Revision ID: 0011_app_runtime_settings
Revises: 0010_coursebuilder
Create Date: 2026-06-03
"""
from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op


revision: str = "0011_app_runtime_settings"
down_revision: str | None = "0010_coursebuilder"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "app_runtime_settings",
        sa.Column("id", sa.String(length=64), primary_key=True),
        sa.Column("llm_enabled", sa.Boolean(), nullable=False, server_default=sa.text("false")),
        sa.Column("llm_provider", sa.String(length=32), nullable=False, server_default="ollama"),
        sa.Column("llm_model", sa.String(length=256), nullable=False, server_default=""),
        sa.Column("llm_base_url", sa.String(length=1024), nullable=False, server_default=""),
        sa.Column("llm_api_key_encrypted", sa.Text(), nullable=True),
        sa.Column("llama_cloud_api_key_encrypted", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("app_runtime_settings")
