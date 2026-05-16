"""compatibility marker for dev databases

Revision ID: 0003_add_uploaded_file_summary
Revises: 0001_initial
Create Date: 2026-05-13
"""
from __future__ import annotations

from collections.abc import Sequence


revision: str = "0003_add_uploaded_file_summary"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
