from __future__ import annotations

import uuid
from hashlib import sha256
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from teacherlm_core.schemas.chunk import Chunk as RetrievalChunk

from db.models import (
    CourseDocumentRecord,
    CourseSectionRecord,
    SearchChunkRecord,
)
from services.chunking_service import Chunk
from services.course_structure_service import CourseDocument, CourseSection


class CourseContentStore:
    """Postgres-backed course content model and searchable chunk metadata."""

    async def replace_document(
        self,
        session: AsyncSession,
        *,
        conversation_id: uuid.UUID,
        uploaded_file_id: uuid.UUID,
        source_file_id: str,
        source_filename: str,
        raw_markdown_path: str | None,
        cleaned_text_path: str | None,
        cleaned_text: str,
        document: CourseDocument,
        chunks: list[Chunk],
    ) -> None:
        await session.execute(
            delete(CourseDocumentRecord).where(
                CourseDocumentRecord.uploaded_file_id == uploaded_file_id
            )
        )
        record = CourseDocumentRecord(
            id=document.id,
            conversation_id=conversation_id,
            uploaded_file_id=uploaded_file_id,
            source_file_id=source_file_id,
            source_filename=source_filename,
            title=document.title,
            raw_markdown_path=raw_markdown_path,
            cleaned_text_path=cleaned_text_path,
            text_hash=sha256(cleaned_text.encode("utf-8")).hexdigest(),
            course_metadata=document.metadata,
        )
        session.add(record)
        session.add_all(
            self._section_record(conversation_id, document.id, section)
            for section in document.sections
        )
        session.add_all(
            self._chunk_record(
                conversation_id=conversation_id,
                document_id=document.id,
                source_file_id=source_file_id,
                chunk=chunk,
            )
            for chunk in chunks
        )
        await session.flush()

    async def get_documents(self, session: AsyncSession, conversation_id: uuid.UUID) -> list[CourseDocumentRecord]:
        result = await session.execute(
            select(CourseDocumentRecord)
            .where(CourseDocumentRecord.conversation_id == conversation_id)
            .order_by(CourseDocumentRecord.created_at, CourseDocumentRecord.title)
        )
        return list(result.scalars().all())

    async def get_sections(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        section_ids: list[str] | None = None,
    ) -> list[CourseSectionRecord]:
        stmt = (
            select(CourseSectionRecord)
            .where(CourseSectionRecord.conversation_id == conversation_id)
            .order_by(CourseSectionRecord.document_id, CourseSectionRecord.order_index)
        )
        if section_ids:
            stmt = stmt.where(CourseSectionRecord.id.in_([uuid.UUID(s) for s in section_ids]))
        result = await session.execute(stmt)
        return list(result.scalars().all())

    async def get_section(
        self,
        session: AsyncSession,
        section_id: uuid.UUID | str,
    ) -> CourseSectionRecord | None:
        return await session.get(CourseSectionRecord, uuid.UUID(str(section_id)))

    async def get_chunks(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        limit: int = 5000,
    ) -> list[RetrievalChunk]:
        result = await session.execute(
            select(SearchChunkRecord)
            .where(SearchChunkRecord.conversation_id == conversation_id)
            .order_by(SearchChunkRecord.document_id, SearchChunkRecord.chunk_index)
            .limit(limit)
        )
        return [record_to_chunk(record) for record in result.scalars().all()]

    async def get_neighbor_chunks(
        self,
        session: AsyncSession,
        chunk: RetrievalChunk,
        *,
        window: int,
    ) -> list[RetrievalChunk]:
        if window <= 0:
            return []
        metadata = chunk.metadata or {}
        document_id = metadata.get("document_id")
        chunk_index = _safe_int(metadata.get("chunk_index"), default=-1)
        if not document_id or chunk_index < 0:
            return []
        result = await session.execute(
            select(SearchChunkRecord)
            .where(
                SearchChunkRecord.document_id == uuid.UUID(str(document_id)),
                SearchChunkRecord.chunk_index >= max(0, chunk_index - window),
                SearchChunkRecord.chunk_index <= chunk_index + window,
                SearchChunkRecord.id != chunk.chunk_id,
            )
            .order_by(SearchChunkRecord.chunk_index)
        )
        return [record_to_chunk(record) for record in result.scalars().all()]

    async def get_chunks_by_section_ids(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        section_ids: list[str],
    ) -> list[RetrievalChunk]:
        if not section_ids:
            return []
        result = await session.execute(
            select(SearchChunkRecord)
            .where(
                SearchChunkRecord.conversation_id == conversation_id,
                SearchChunkRecord.section_id.in_([uuid.UUID(s) for s in section_ids]),
            )
            .order_by(SearchChunkRecord.document_id, SearchChunkRecord.chunk_index)
        )
        return [record_to_chunk(record) for record in result.scalars().all()]

    @staticmethod
    def _section_record(
        conversation_id: uuid.UUID,
        document_id: uuid.UUID,
        section: CourseSection,
    ) -> CourseSectionRecord:
        return CourseSectionRecord(
            id=section.id,
            conversation_id=conversation_id,
            document_id=document_id,
            parent_section_id=section.parent_id,
            level=section.level,
            title=section.title,
            heading_path=section.heading_path,
            order_index=section.order_index,
            page_start=section.page_start,
            page_end=section.page_end,
            text=section.text,
            summary=section.summary,
            key_concepts=section.key_concepts,
            equations=section.equations,
            tables=[{"rows": table.rows, "order_index": table.order_index} for table in section.tables],
            timeline_events=section.timeline_events,
            section_metadata={"extractor": "course-structure-v1"},
        )

    @staticmethod
    def _chunk_record(
        *,
        conversation_id: uuid.UUID,
        document_id: uuid.UUID,
        source_file_id: str,
        chunk: Chunk,
    ) -> SearchChunkRecord:
        heading_path = chunk.metadata.get("heading_path_list") or []
        if not isinstance(heading_path, list):
            heading_path = str(chunk.metadata.get("heading_path", "")).split(" > ")
        return SearchChunkRecord(
            id=chunk.chunk_id,
            conversation_id=conversation_id,
            document_id=document_id,
            section_id=uuid.UUID(str(chunk.section_id or chunk.metadata["section_id"])),
            source_filename=chunk.source,
            source_file_id=source_file_id,
            text=chunk.text,
            chunk_index=chunk.index,
            token_count=chunk.token_count,
            prev_chunk_id=_optional_str(chunk.metadata.get("prev_chunk_id")),
            next_chunk_id=_optional_str(chunk.metadata.get("next_chunk_id")),
            page_start=_optional_int(chunk.metadata.get("page_start")),
            page_end=_optional_int(chunk.metadata.get("page_end")),
            heading_path=[str(item) for item in heading_path if str(item)],
            chunk_metadata=dict(chunk.metadata),
        )


def record_to_chunk(record: SearchChunkRecord) -> RetrievalChunk:
    metadata: dict[str, Any] = dict(record.chunk_metadata or {})
    metadata.update(
        {
            "document_id": str(record.document_id),
            "section_id": str(record.section_id),
            "source_file_id": record.source_file_id,
            "chunk_index": record.chunk_index,
            "token_count": record.token_count,
            "prev_chunk_id": record.prev_chunk_id or "",
            "next_chunk_id": record.next_chunk_id or "",
            "heading_path_list": list(record.heading_path or []),
            "heading_path": " > ".join(record.heading_path or []),
        }
    )
    return RetrievalChunk(
        text=record.text,
        source=record.source_filename,
        score=0.0,
        chunk_id=record.id,
        metadata=metadata,
    )


def _optional_str(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


def _optional_int(value: object) -> int | None:
    try:
        return int(value) if value is not None and str(value) else None
    except (TypeError, ValueError):
        return None


def _safe_int(value: object, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


_store: CourseContentStore | None = None


def get_course_content_store() -> CourseContentStore:
    global _store
    if _store is None:
        _store = CourseContentStore()
    return _store
