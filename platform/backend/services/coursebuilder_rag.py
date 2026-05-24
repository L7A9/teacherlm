from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import SearchChunkRecord


class CourseBuilderRagService:
    async def load_chunks(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        limit: int = 1200,
    ) -> list[SearchChunkRecord]:
        result = await session.execute(
            select(SearchChunkRecord)
            .where(SearchChunkRecord.conversation_id == conversation_id)
            .order_by(SearchChunkRecord.document_id, SearchChunkRecord.chunk_index)
            .limit(limit)
        )
        return list(result.scalars().all())

    async def retrieve_lesson_chunks(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        query: str,
        *,
        fallback_chunks: list[SearchChunkRecord],
        top_k: int = 8,
    ) -> list[SearchChunkRecord]:
        by_id = {chunk.id: chunk for chunk in fallback_chunks}
        try:
            from services.vector_service import get_vector_service

            hits = await get_vector_service().search(conversation_id, query, top_k=top_k)
            matched = [by_id[hit.chunk_id] for hit in hits if hit.chunk_id in by_id]
            if matched:
                return matched[:top_k]
        except Exception:
            pass
        return fallback_chunks[:top_k]

    def citations_for(
        self,
        chunks: list[SearchChunkRecord],
        source_chunk_ids: list[str] | None = None,
    ) -> list[dict]:
        wanted = set(source_chunk_ids or [])
        selected = [chunk for chunk in chunks if not wanted or chunk.id in wanted]
        if not selected and chunks:
            selected = chunks[:2]
        citations: list[dict] = []
        seen: set[str] = set()
        for chunk in selected[:4]:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            section = " > ".join(chunk.heading_path or [])
            citations.append(
                {
                    "chunk_id": chunk.id,
                    "source": chunk.source_filename,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "section": section,
                    "snippet": " ".join(chunk.text.split())[:360],
                }
            )
        return citations


_service: CourseBuilderRagService | None = None


def get_coursebuilder_rag_service() -> CourseBuilderRagService:
    global _service
    if _service is None:
        _service = CourseBuilderRagService()
    return _service
