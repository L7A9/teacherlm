from __future__ import annotations

import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass, field
from hashlib import sha1

from config import Settings, get_settings
from services.course_structure_service import CourseDocument, CourseSection


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_PARAGRAPH_SPLIT = re.compile(r"\n{2,}")


def approx_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text.split()) * 1.3))


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    source: str
    index: int
    token_count: int
    document_id: uuid.UUID | None = None
    section_id: uuid.UUID | None = None
    metadata: dict[str, object] = field(default_factory=dict)


class ChunkingService:
    """Create stable search chunks from structured course sections."""

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        self._max_tokens = s.chunk_max_tokens
        self._overlap_tokens = s.chunk_overlap_tokens

    def chunk_course_document(
        self,
        document: CourseDocument,
        *,
        source_file_id: str,
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        for section in document.sections:
            chunks.extend(
                self._chunk_section(
                    document=document,
                    section=section,
                    source_file_id=source_file_id,
                    start_index=len(chunks),
                )
            )
        self._link_neighbors(chunks)
        return chunks

    def chunk_text(self, text: str, *, source: str) -> list[Chunk]:
        """Compatibility helper for tests and one-off local scripts."""
        from services.course_structure_service import get_course_structure_extractor

        document = get_course_structure_extractor().extract(
            text,
            conversation_id="local",
            source_file_id=source,
            source_filename=source,
        )
        return self.chunk_course_document(document, source_file_id=source)

    def _chunk_section(
        self,
        *,
        document: CourseDocument,
        section: CourseSection,
        source_file_id: str,
        start_index: int,
    ) -> list[Chunk]:
        units = _section_units(section.text)
        if not units:
            return []

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_tokens = 0
        section_chunk_index = 0

        for unit in units:
            unit_tokens = approx_tokens(unit)

            if unit_tokens >= self._max_tokens:
                if buffer:
                    chunks.append(
                        self._finalize(
                            document=document,
                            section=section,
                            source_file_id=source_file_id,
                            global_index=start_index + len(chunks),
                            section_chunk_index=section_chunk_index,
                            parts=buffer,
                            token_count=buffer_tokens,
                        )
                    )
                    section_chunk_index += 1
                    buffer, buffer_tokens = self._carry_overlap(buffer)
                chunks.append(
                    self._finalize(
                        document=document,
                        section=section,
                        source_file_id=source_file_id,
                        global_index=start_index + len(chunks),
                        section_chunk_index=section_chunk_index,
                        parts=[unit],
                        token_count=unit_tokens,
                    )
                )
                section_chunk_index += 1
                buffer, buffer_tokens = self._carry_overlap([unit])
                continue

            if buffer and buffer_tokens + unit_tokens > self._max_tokens:
                chunks.append(
                    self._finalize(
                        document=document,
                        section=section,
                        source_file_id=source_file_id,
                        global_index=start_index + len(chunks),
                        section_chunk_index=section_chunk_index,
                        parts=buffer,
                        token_count=buffer_tokens,
                    )
                )
                section_chunk_index += 1
                buffer, buffer_tokens = self._carry_overlap(buffer)

            buffer.append(unit)
            buffer_tokens += unit_tokens

        if buffer:
            chunks.append(
                self._finalize(
                    document=document,
                    section=section,
                    source_file_id=source_file_id,
                    global_index=start_index + len(chunks),
                    section_chunk_index=section_chunk_index,
                    parts=buffer,
                    token_count=buffer_tokens,
                )
            )

        return chunks

    def _finalize(
        self,
        *,
        document: CourseDocument,
        section: CourseSection,
        source_file_id: str,
        global_index: int,
        section_chunk_index: int,
        parts: list[str],
        token_count: int,
    ) -> Chunk:
        text = "\n\n".join(part.strip() for part in parts if part.strip()).strip()
        chunk_id = _chunk_id(source_file_id, str(section.id), section_chunk_index, text)
        heading_path = " > ".join(section.heading_path)
        return Chunk(
            chunk_id=chunk_id,
            text=text,
            source=document.source_filename,
            index=global_index,
            token_count=token_count,
            document_id=document.id,
            section_id=section.id,
            metadata={
                "chunker": "structured-section-v1",
                "document_id": str(document.id),
                "section_id": str(section.id),
                "parent_section_id": str(section.parent_id) if section.parent_id else "",
                "heading_path": heading_path,
                "heading_path_list": section.heading_path,
                "section_title": section.title,
                "section_index": section.order_index,
                "chunk_index": global_index,
                "section_chunk_index": section_chunk_index,
                "source_file_id": source_file_id,
                "source": document.source_filename,
                "section_summary": section.summary,
                "key_concepts": section.key_concepts,
                "equation_count": len(section.equations),
                "table_count": len(section.tables),
                "timeline_event_count": len(section.timeline_events),
            },
        )

    def _carry_overlap(self, prev: list[str]) -> tuple[list[str], int]:
        if self._overlap_tokens <= 0 or not prev:
            return [], 0

        carry: list[str] = []
        total = 0
        for unit in reversed(prev):
            tokens = approx_tokens(unit)
            if carry and total + tokens > self._overlap_tokens:
                break
            carry.insert(0, unit)
            total += tokens
            if total >= self._overlap_tokens:
                break
        return carry, total

    @staticmethod
    def _link_neighbors(chunks: list[Chunk]) -> None:
        for idx, chunk in enumerate(chunks):
            if idx > 0:
                chunk.metadata["prev_chunk_id"] = chunks[idx - 1].chunk_id
            if idx + 1 < len(chunks):
                chunk.metadata["next_chunk_id"] = chunks[idx + 1].chunk_id


def _section_units(text: str) -> list[str]:
    units: list[str] = []
    for paragraph in _PARAGRAPH_SPLIT.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if _looks_structured_block(paragraph):
            units.append(paragraph)
            continue
        for sentence in _SENTENCE_SPLIT.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                units.append(sentence)
    return units


def _looks_structured_block(paragraph: str) -> bool:
    lines = [line.strip() for line in paragraph.splitlines() if line.strip()]
    if len(lines) <= 1:
        return False
    if any(line.startswith("|") for line in lines):
        return True
    if any(line.startswith(("- ", "* ", "1. ")) for line in lines):
        return True
    return "$$" in paragraph or "\\[" in paragraph


def _chunk_id(source_file_id: str, section_id: str, section_chunk_index: int, text: str) -> str:
    fingerprint = sha1(text.encode("utf-8")).hexdigest()[:16]
    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"teacherlm:chunk:{source_file_id}:{section_id}:{section_chunk_index}:{fingerprint}",
        )
    )


def chunks_for_sections(chunks: Iterable[Chunk], section_ids: set[str]) -> list[Chunk]:
    return [
        chunk
        for chunk in chunks
        if str(chunk.section_id or chunk.metadata.get("section_id", "")) in section_ids
    ]


_chunker: ChunkingService | None = None


def get_chunker() -> ChunkingService:
    global _chunker
    if _chunker is None:
        _chunker = ChunkingService()
    return _chunker
