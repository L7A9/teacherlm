from __future__ import annotations

from teacherlm_core.schemas.chunk import Chunk

from ..schemas import (
    BloomLevel,
    FillBlank,
    MCQ,
    Question,
    QuestionKind,
    QuestionSlot,
    TrueFalse,
)
from .llm_service import LLMService, build_system_prompt


_PROMPT_FOR: dict[QuestionKind, str] = {
    "mcq": "mcq_generation.txt",
    "true_false": "true_false_generation.txt",
    "fill_blank": "fill_blank_generation.txt",
}

_SCHEMA_FOR: dict[QuestionKind, type] = {
    "mcq": MCQ,
    "true_false": TrueFalse,
    "fill_blank": FillBlank,
}


def _pick_chunk(
    concept_chunk_ids: list[str],
    all_chunks: list[Chunk],
) -> Chunk | None:
    """Prefer a chunk explicitly tagged for this concept; otherwise highest-scoring."""
    by_id = {c.chunk_id: c for c in all_chunks}
    for cid in concept_chunk_ids:
        if cid in by_id:
            return by_id[cid]
    if not all_chunks:
        return None
    return max(all_chunks, key=lambda c: c.score)


def _normalize_to_slot(question: Question, slot: QuestionSlot, chunk_id: str) -> Question:
    """Force concept / source_chunk_id / bloom_level to match the slot.

    The model tends to drift on these fields even with format= constraints, so
    we overwrite to keep the analytics + sourcing honest.
    """
    generated_concept = str(getattr(question, "concept", "") or "").strip()
    generated_text = f"{question.question} {question.explanation}".casefold()
    slot_concept = slot.concept
    if (
        generated_concept
        and generated_concept.casefold() in generated_text
        and slot.concept.casefold() not in generated_text
    ):
        slot_concept = generated_concept

    return question.model_copy(
        update={
            "concept": slot_concept,
            "source_chunk_id": chunk_id,
            "bloom_level": slot.bloom_level,
        }
    )


async def generate_one(
    slot: QuestionSlot,
    concept_chunk_ids: list[str],
    all_chunks: list[Chunk],
    llm: LLMService,
) -> Question | None:
    """Generate a single question for the given slot via ollama format=schema.

    Returns None if no chunk is available or generation fails after retries.
    """
    chunk = _pick_chunk(concept_chunk_ids, all_chunks)
    if chunk is None:
        return None

    prompt_name = _PROMPT_FOR[slot.kind]
    schema = _SCHEMA_FOR[slot.kind]

    system = build_system_prompt(
        prompt_name,
        concept=slot.concept,
        bloom_level=slot.bloom_level,
        chunk_id=chunk.chunk_id,
        chunk_text=chunk.text,
    )
    user = f'Write one {slot.kind} question for "{slot.concept}" at Bloom level "{slot.bloom_level}".'

    try:
        result = await llm.generate_structured(
            system=system,
            user_message=user,
            schema=schema,
        )
    except Exception:
        return None

    return _normalize_to_slot(result, slot, chunk.chunk_id)


async def generate_questions(
    slots: list[QuestionSlot],
    concept_to_chunk_ids: dict[str, list[str]],
    all_chunks: list[Chunk],
    llm: LLMService,
) -> list[Question]:
    """Generate one question per slot, sequentially.

    Sequential (not gathered) on purpose: keeps load on a single Ollama instance
    predictable, and per-slot prompts are independent so ordering doesn't matter.
    """
    out: list[Question] = []
    for slot in slots:
        question = await generate_one(
            slot=slot,
            concept_chunk_ids=concept_to_chunk_ids.get(slot.concept, []),
            all_chunks=all_chunks,
            llm=llm,
        )
        if question is not None:
            out.append(question)
    return out
