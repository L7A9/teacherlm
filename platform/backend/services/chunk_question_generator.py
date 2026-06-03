from __future__ import annotations

import logging
from collections.abc import Sequence
from functools import lru_cache
from typing import Any

from pydantic import BaseModel, Field

from teacherlm_core.llm.ollama_client import OllamaClient
from teacherlm_core.llm.runtime import build_llm_client_kwargs

from config import Settings, get_settings
from services.chunking_service import Chunk


logger = logging.getLogger(__name__)


class ChunkQuestionItem(BaseModel):
    chunk_id: str
    questions: list[str] = Field(default_factory=list)


class ChunkQuestionBatch(BaseModel):
    chunks: list[ChunkQuestionItem] = Field(default_factory=list)


class ChunkQuestionGenerator:
    """Generate likely student questions for chunks during ingestion."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def annotate_chunks(
        self,
        chunks: Sequence[Chunk],
        *,
        llm_options: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        if not chunks or not self._settings.chunk_question_generation_enabled:
            return list(chunks)

        out = list(chunks)
        for batch in _batched(out, max(1, self._settings.chunk_question_batch_size)):
            try:
                try:
                    generated = await self._generate_batch(batch, llm_options=llm_options)
                except TypeError as exc:
                    if "unexpected keyword argument 'llm_options'" not in str(exc):
                        raise
                    generated = await self._generate_batch(batch)  # type: ignore[call-arg]
            except Exception:  # noqa: BLE001
                logger.exception("chunk question generation failed for a batch")
                continue

            by_id = {item.chunk_id: item.questions for item in generated.chunks}
            for chunk in batch:
                questions = _sanitize_questions(
                    by_id.get(chunk.chunk_id, []),
                    limit=max(1, self._settings.chunk_question_count),
                )
                if not questions:
                    continue
                chunk.metadata["generated_questions"] = questions
                chunk.metadata["question_generator"] = "llm-v1"
        return out

    async def _generate_batch(
        self,
        chunks: Sequence[Chunk],
        *,
        llm_options: dict[str, Any] | None,
    ) -> ChunkQuestionBatch:
        cfg = build_llm_client_kwargs(
            default_base_url=self._settings.ollama_host,
            default_model=self._settings.ollama_chat_model,
            options=(llm_options or {}).get("llm") if isinstance(llm_options, dict) else None,
        )
        client = OllamaClient(**cfg)  # type: ignore[arg-type]
        return await client.chat_structured(
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": self._format_batch(chunks)},
            ],
            schema=ChunkQuestionBatch,
            options={"temperature": 0.2, "num_predict": 1400, "max_tokens": 1400},
        )

    def _format_batch(self, chunks: Sequence[Chunk]) -> str:
        parts: list[str] = []
        max_chars = max(400, self._settings.chunk_question_max_chars)
        wanted = max(1, self._settings.chunk_question_count)
        for chunk in chunks:
            metadata = chunk.metadata or {}
            heading = str(metadata.get("heading_path", "")).strip()
            concepts = ", ".join(str(item) for item in metadata.get("key_concepts", [])[:8])
            text = " ".join(chunk.text.split())
            if len(text) > max_chars:
                text = text[:max_chars].rsplit(" ", 1)[0].strip()
            parts.append(
                "\n".join(
                    [
                        f"chunk_id: {chunk.chunk_id}",
                        f"source: {chunk.source}",
                        f"heading: {heading or '(none)'}",
                        f"key_concepts: {concepts or '(none)'}",
                        f"questions_to_generate: {wanted}",
                        "chunk_text:",
                        text,
                    ]
                )
            )
        return "\n\n---\n\n".join(parts)


def searchable_chunk_text(chunk: Chunk | Any) -> str:
    """Text used for retrieval indexing, not for final answer grounding."""
    metadata = getattr(chunk, "metadata", {}) or {}
    questions = _metadata_strings(metadata.get("generated_questions"))
    heading = str(metadata.get("heading_path", "") or "").strip()
    concepts = _metadata_strings(metadata.get("key_concepts"))
    parts = [str(getattr(chunk, "text", "") or "")]
    if heading:
        parts.append(f"Section path: {heading}")
    if concepts:
        parts.append("Key concepts: " + ", ".join(concepts))
    if questions:
        parts.append("Likely student questions:\n" + "\n".join(f"- {q}" for q in questions))
    return "\n\n".join(part for part in parts if part.strip())


def _sanitize_questions(values: Sequence[object], *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        text = " ".join(str(raw or "").split()).strip(" -")
        if not 8 <= len(text) <= 180:
            continue
        if not text.endswith("?"):
            text = f"{text}?"
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if len(out) >= limit:
            break
    return out


def _metadata_strings(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value or "").strip()
    return [text] if text else []


def _batched[T](items: Sequence[T], batch_size: int) -> list[list[T]]:
    return [
        list(items[index : index + batch_size])
        for index in range(0, len(items), batch_size)
    ]


_SYSTEM_PROMPT = """You generate retrieval helper questions for TeacherLM.

For each chunk, write likely questions a student could ask whose answer is
contained in that chunk. The questions are used only for search, so they must
be grounded in the chunk and must not introduce facts from outside it.

Rules:
- Return exactly one object per chunk_id.
- Generate the requested number of concise questions when possible.
- Include acronym, formula, comparison, definition, and "explain" phrasing when
  those are relevant to the chunk.
- Prefer the same language as the chunk when clear.
- Do not answer the questions.
- Return only JSON matching the schema."""


@lru_cache(maxsize=1)
def get_chunk_question_generator() -> ChunkQuestionGenerator:
    return ChunkQuestionGenerator()

