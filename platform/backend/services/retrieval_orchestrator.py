from __future__ import annotations

import asyncio
import logging
import re
import unicodedata
import uuid
from typing import Literal

from teacherlm_core.retrieval.reranker import CrossEncoderReranker
from teacherlm_core.schemas.chunk import Chunk

from config import Settings, get_settings
from services.course_context_service import (
    CourseContextService,
    RetrievalMode,
    get_course_context_service,
)


logger = logging.getLogger(__name__)


OutputType = Literal[
    "text",
    "quiz",
    "report",
    "presentation",
    "chart",
    "podcast",
    "mindmap",
]


_OUTPUT_TO_MODE: dict[str, RetrievalMode] = {
    "text": "semantic_topk",
    "chat": "semantic_topk",
    "quiz": "coverage_broad",
    "report": "topic_clusters",
    "presentation": "topic_clusters",
    "podcast": "narrative_arc",
    "chart": "relationship_dense",
    "diagram": "relationship_dense",
    "mindmap": "topic_clusters",
}

_BROAD_NO_TOPIC_OUTPUTS = {"quiz", "mindmap", "presentation", "podcast"}
_TOPIC_OUTPUTS_WITH_SECTION_CONTEXT = {"quiz", "podcast", "presentation"}


class RetrievalOrchestrator:
    """Selects measurable retrieval/context policies for each output type."""

    def __init__(
        self,
        settings: Settings | None = None,
        context_service: CourseContextService | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._context = context_service or get_course_context_service()
        self._reranker: CrossEncoderReranker | None = None

    async def warmup(self) -> None:
        if not self._settings.retrieval_rerank_enabled or not self._settings.retrieval_rerank_warmup_enabled:
            return
        if self._reranker is None:
            logger.info("warming retrieval reranker model %s", self._settings.retrieval_reranker_model)
            self._reranker = await asyncio.to_thread(
                CrossEncoderReranker,
                self._settings.retrieval_reranker_model,
            )

    def mode_for(self, output_type: str) -> RetrievalMode:
        return _OUTPUT_TO_MODE.get(output_type, "semantic_topk")

    async def retrieve_for(
        self,
        *,
        output_type: str,
        query: str,
        conversation_id: uuid.UUID | str,
        topic: str | None = None,
    ) -> list[Chunk]:
        if output_type in _BROAD_NO_TOPIC_OUTPUTS and not topic:
            return await self._context.get_generator_context(
                conversation_id=conversation_id,
                output_type=output_type,
                query=query,
                topic=topic,
            )

        if output_type in {"text", "chat"} and _is_course_overview_query(query):
            overview = _dedupe_chunks(
                [
                    *(await self._context.get_mindmap_course_context(conversation_id)),
                    *(await self._context.get_full_course_outline(conversation_id)),
                    *(await self._context.get_representative_course_context(conversation_id)),
                ]
            )
            if overview:
                return overview
            return await self.retrieve(
                mode="coverage_broad",
                query="",
                conversation_id=conversation_id,
            )

        mode = self.mode_for(output_type)
        chunks = await self.retrieve(
            mode=mode,
            query=topic or query,
            conversation_id=conversation_id,
        )

        if output_type in _TOPIC_OUTPUTS_WITH_SECTION_CONTEXT and (topic or query):
            sections = await self._section_summaries_for_hits(conversation_id, chunks)
            if output_type == "presentation":
                sections.extend(await self._context.get_equations(conversation_id))
                sections.extend(await self._context.get_tables(conversation_id))
            return _dedupe_chunks([*sections, *chunks])

        return chunks

    async def retrieve(
        self,
        *,
        mode: RetrievalMode,
        query: str,
        conversation_id: uuid.UUID | str,
        full_corpus: bool = False,
    ) -> list[Chunk]:
        if full_corpus:
            return await self._context.get_course_sections(conversation_id)

        chunks = await self._context.get_relevant_chunks(conversation_id, query, mode)
        chunks = await self._maybe_rerank(mode, query, chunks)
        return await self._maybe_expand_context(mode, conversation_id, chunks)

    async def _section_summaries_for_hits(
        self,
        conversation_id: uuid.UUID | str,
        hits: list[Chunk],
    ) -> list[Chunk]:
        section_ids = [
            str(chunk.metadata.get("section_id", "")).strip()
            for chunk in hits
            if chunk.metadata.get("section_id")
        ]
        if not section_ids:
            return []
        all_sections = await self._context.get_course_sections(conversation_id)
        wanted = set(section_ids)
        return [
            Chunk(
                text=chunk.text[: self._settings.retrieval_expansion_max_chars],
                source=chunk.source,
                score=1.0,
                chunk_id=f"topic-section:{chunk.metadata.get('section_id')}",
                metadata={**chunk.metadata, "context_type": "topic_section"},
            )
            for chunk in all_sections
            if str(chunk.metadata.get("section_id", "")) in wanted
        ]

    async def _maybe_rerank(
        self,
        mode: RetrievalMode,
        query: str,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        if (
            not self._settings.retrieval_rerank_enabled
            or mode not in set(self._settings.retrieval_rerank_modes)
            or not query.strip()
            or not chunks
        ):
            return chunks[: self._settings.retrieval_top_k]

        try:
            if self._reranker is None:
                logger.info("loading retrieval reranker model %s", self._settings.retrieval_reranker_model)
                self._reranker = await asyncio.to_thread(
                    CrossEncoderReranker,
                    self._settings.retrieval_reranker_model,
                )
            comparison_labels = _comparison_labels(chunks)
            if len(comparison_labels) >= 2:
                return await self._rerank_comparison_groups(query, chunks, comparison_labels)
            return await self._reranker.rerank(
                query,
                chunks,
                top_k=max(self._settings.retrieval_top_k, self._settings.retrieval_rerank_top_k),
            )
        except Exception:
            logger.exception("retrieval reranking failed; falling back to fused candidates")
            return chunks[: self._settings.retrieval_top_k]

    async def _rerank_comparison_groups(
        self,
        query: str,
        chunks: list[Chunk],
        labels: list[str],
    ) -> list[Chunk]:
        if self._reranker is None:
            return chunks[: self._settings.retrieval_top_k]

        per_group_top_k = max(2, self._settings.retrieval_top_k // len(labels) + 2)
        ranked_groups: dict[str, list[Chunk]] = {}
        for label in labels:
            group = [
                chunk
                for chunk in chunks
                if str(chunk.metadata.get("matched_query_term", "")) == label
            ]
            ranked_groups[label] = await self._reranker.rerank(
                query,
                group,
                top_k=per_group_top_k,
            )

        selected: list[Chunk] = []
        seen: set[str] = set()
        max_depth = max((len(group) for group in ranked_groups.values()), default=0)
        for index in range(max_depth):
            for label in labels:
                group = ranked_groups[label]
                if index >= len(group):
                    continue
                chunk = group[index]
                if chunk.chunk_id in seen:
                    continue
                seen.add(chunk.chunk_id)
                selected.append(chunk)
                if len(selected) >= self._settings.retrieval_top_k:
                    return selected

        if len(selected) < self._settings.retrieval_top_k:
            remainder = [
                chunk
                for chunk in chunks
                if chunk.chunk_id not in seen and not chunk.metadata.get("matched_query_term")
            ]
            selected.extend(remainder[: self._settings.retrieval_top_k - len(selected)])
        return selected[: self._settings.retrieval_top_k]

    async def _maybe_expand_context(
        self,
        mode: RetrievalMode,
        conversation_id: uuid.UUID | str,
        chunks: list[Chunk],
    ) -> list[Chunk]:
        if not self._settings.retrieval_context_expansion_enabled or not chunks:
            return chunks[: self._settings.retrieval_top_k]
        if mode == "coverage_broad":
            return chunks[: self._settings.retrieval_top_k]

        expanded: list[Chunk] = []
        for chunk in chunks[: self._settings.retrieval_top_k]:
            neighbors = await self._neighbors(chunk)
            parts = self._context_parts(chunk, neighbors)
            metadata = dict(chunk.metadata)
            metadata.update({"retrieval_expanded": True, "retrieval_mode": mode})
            expanded.append(
                Chunk(
                    text=self._compose_expanded_text(chunk, parts),
                    source=chunk.source,
                    score=chunk.score,
                    chunk_id=chunk.chunk_id,
                    metadata=metadata,
                )
            )
        return expanded

    async def _neighbors(self, chunk: Chunk) -> list[Chunk]:
        from db.session import session_scope
        from services.course_content_store import get_course_content_store

        async with session_scope() as session:
            return await get_course_content_store().get_neighbor_chunks(
                session,
                chunk,
                window=self._settings.retrieval_neighbor_window,
            )

    def _context_parts(self, chunk: Chunk, neighbors: list[Chunk]) -> list[str]:
        parts: list[str] = []
        heading_path = str(chunk.metadata.get("heading_path", "") or "").strip()
        if heading_path:
            parts.append(f"Section path: {heading_path}")

        summary = str(chunk.metadata.get("section_summary", "") or "").strip()
        if summary:
            parts.append(f"Section summary: {summary}")

        parts.extend(neighbor.text for neighbor in neighbors)
        return parts

    def _compose_expanded_text(self, chunk: Chunk, context_parts: list[str]) -> str:
        seen: set[str] = set()
        cleaned_parts: list[str] = []
        for part in [*context_parts, f"Focused chunk:\n{chunk.text}"]:
            normalized = " ".join(part.split())
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            cleaned_parts.append(part.strip())

        text = "\n\n".join(cleaned_parts)
        max_chars = max(1000, self._settings.retrieval_expansion_max_chars)
        if len(text) > max_chars:
            return text[:max_chars].rsplit(" ", 1)[0].strip()
        return text


def _dedupe_chunks(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    out: list[Chunk] = []
    for chunk in chunks:
        if chunk.chunk_id in seen:
            continue
        seen.add(chunk.chunk_id)
        out.append(chunk)
    return out


def _comparison_labels(chunks: list[Chunk]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for chunk in chunks:
        label = str(chunk.metadata.get("matched_query_term", "")).strip()
        if not label or label in seen:
            continue
        seen.add(label)
        labels.append(label)
    return labels


_COURSE_OVERVIEW_INTENTS = {
    "explain",
    "summarize",
    "summary",
    "overview",
    "teach",
    "review",
    "understand",
    "resume",
    "resumer",
    "explique",
    "expliquer",
    "apprendre",
    "comprendre",
    "presente",
    "presenter",
    "شرح",
    "لخص",
}

_COURSE_TARGET_TERMS = {
    "course",
    "class",
    "lesson",
    "lecture",
    "chapter",
    "module",
    "material",
    "materials",
    "document",
    "documents",
    "file",
    "files",
    "uploaded",
    "cours",
    "seance",
    "chapitre",
    "support",
    "supports",
    "document",
    "documents",
    "fichier",
    "fichiers",
}

_BROAD_PRONOUN_TARGETS = {
    "this",
    "that",
    "it",
    "these",
    "those",
    "ce",
    "cet",
    "cette",
    "ces",
    "ca",
    "cela",
}


def _is_course_overview_query(query: str) -> bool:
    """Detect vague course-wide chat requests that semantic top-k handles poorly."""

    normalized = _normalize_query(query)
    if not normalized:
        return False

    tokens = set(normalized.split())
    has_intent = bool(tokens & _COURSE_OVERVIEW_INTENTS)
    has_course_target = bool(tokens & _COURSE_TARGET_TERMS)
    has_broad_pronoun = bool(tokens & _BROAD_PRONOUN_TARGETS)

    if has_intent and has_course_target:
        return True
    if has_intent and has_broad_pronoun and len(tokens) <= 8:
        return True
    if "about" in tokens and has_course_target and has_broad_pronoun and len(tokens) <= 8:
        return True

    phrase_patterns = [
        r"\bwhat\s+(?:is\s+|s\s+)?(this|that)\s+(course|class|lecture|lesson)\s+about\b",
        r"\bwhat\s+are\s+these\s+(documents|files|materials)\s+about\b",
        r"\bde\s+quoi\s+parle\s+(ce|cet|cette)\s+(cours|document|chapitre)\b",
        r"\bc'?est\s+quoi\s+(ce|cet|cette)\s+(cours|document|chapitre)\b",
    ]
    return any(re.search(pattern, normalized) for pattern in phrase_patterns)


def _normalize_query(query: str) -> str:
    text = unicodedata.normalize("NFKD", query.casefold())
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w\s\u0600-\u06ff]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_orchestrator: RetrievalOrchestrator | None = None


def get_retrieval_orchestrator() -> RetrievalOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RetrievalOrchestrator()
    return _orchestrator
