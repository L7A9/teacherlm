from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator

from teacherlm_core.llm.language import set_current_language
from teacherlm_core.llm.runtime import set_current_llm_options
from teacherlm_core.schemas.chunk import Chunk
from teacherlm_core.schemas.generator_io import (
    GeneratorArtifact,
    GeneratorInput,
    GeneratorOutput,
    LearnerUpdates,
)

from .config import get_settings
from .schemas import (
    ExtractedConcepts,
    MCQ,
    Question,
    QuestionKind,
    QuestionSlot,
    QuizOutput,
    QuizPlan,
    TrueFalse,
)
from .services.artifact_store import get_artifact_store
from .services.concept_extractor import extract_concepts
from .services.difficulty_adapter import plan_question_mix
from .services.distractor_engine import enhance_mcq_distractors
from .services.llm_service import LLMService, build_system_prompt, get_llm_service
from .services.quality_validator import (
    bloom_distribution,
    deduplicate_questions,
    is_valid,
    validate_questions,
)
from .services.question_generator import generate_questions


# Frontend sends human-friendly type names; map to the internal QuestionKind.
_FRONTEND_KIND_ALIASES: dict[str, QuestionKind] = {
    "mcq": "mcq",
    "multiple_choice": "mcq",
    "multi_choice": "mcq",
    "multichoice": "mcq",
    "true_false": "true_false",
    "truefalse": "true_false",
    "tf": "true_false",
}

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _resolve_question_count(options: dict) -> int:
    s = get_settings()
    raw = (
        options.get("question_count")
        or options.get("n_questions")
        or options.get("count")
        or s.default_question_count
    )
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = s.default_question_count
    return max(s.min_question_count, min(n, s.max_question_count))


def _resolve_allowed_kinds(options: dict) -> list[QuestionKind] | None:
    raw = options.get("question_types") or options.get("types") or options.get("kinds")
    if not raw:
        return None
    if isinstance(raw, str):
        raw = [raw]
    kinds: list[QuestionKind] = []
    for item in raw:
        key = str(item).strip().lower()
        kind = _FRONTEND_KIND_ALIASES.get(key)
        if kind and kind not in kinds:
            kinds.append(kind)
    if kinds:
        return kinds[:1]
    return ["mcq"]


def _resolve_title(options: dict, learner_concepts: list[str]) -> str:
    if topic := options.get("topic"):
        return f"Quiz: {topic}"
    if learner_concepts:
        return f"Quiz: {learner_concepts[0]}"
    return "Quiz"


def _concept_to_chunk_ids(extracted: ExtractedConcepts) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for level in ("remember", "understand", "apply", "analyze"):
        for card in getattr(extracted, level):
            out.setdefault(card.name, list(card.source_chunk_ids))
    return out


async def _enhance_mcqs(
    questions: list[Question],
    chunks: list[Chunk],
) -> list[Question]:
    out: list[Question] = []
    for q in questions:
        if isinstance(q, MCQ):
            out.append(await enhance_mcq_distractors(q, chunks))
        else:
            out.append(q)
    return out


def _repair_obvious_concept_labels(questions: list[Question]) -> list[Question]:
    return [
        question.model_copy(update={"concept": _clean_concept_label(question.concept)})
        for question in questions
    ]


def _clean_concept_label(value: str) -> str:
    label = " ".join(str(value or "").split()).strip(" -:;")
    return label[:90] if label else "Course concept"


def _attach_canonical_concept_ids(
    questions: list[Question],
    known_concepts: list,
) -> list[Question]:
    if not known_concepts:
        return questions

    by_label: dict[str, str] = {}
    for concept in known_concepts:
        concept_id = str(getattr(concept, "id", "") or "")
        labels = [
            str(getattr(concept, "name", "") or ""),
            *[str(alias) for alias in (getattr(concept, "aliases", []) or [])],
        ]
        for label in labels:
            key = _concept_key(label)
            if key and concept_id:
                by_label.setdefault(key, concept_id)

    out: list[Question] = []
    for question in questions:
        concept_id = by_label.get(_concept_key(question.concept))
        if concept_id:
            out.append(question.model_copy(update={"concept_id": concept_id}))
        else:
            out.append(question)
    return out


def _concept_key(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", str(value or "").casefold()))


async def _build_intro(
    *,
    llm: LLMService,
    learner_struggling: list[str],
    learner_understood: list[str],
    plan: QuizPlan,
    bloom_counts: dict[str, int],
) -> str:
    system = build_system_prompt(
        "adaptive_guidance.txt",
        struggling=", ".join(learner_struggling) or "(none yet)",
        understood=", ".join(learner_understood) or "(none yet)",
        total=plan.total,
        bloom_mix=", ".join(f"{k}:{v}" for k, v in bloom_counts.items() if v),
        slot_mix=", ".join(f"{k}:{v}" for k, v in plan.counts.items() if v),
    )
    try:
        return await llm.reply(system=system, user_message="Write the intro.")
    except Exception:
        return "Let's test what you know — quick check on what we've been working on."


def _top_up_with_grounded_questions(
    *,
    questions: list[Question],
    plan: QuizPlan,
    concept_to_chunk_ids: dict[str, list[str]],
    chunks: list[Chunk],
    target_count: int,
) -> list[Question]:
    if len(questions) >= target_count:
        return questions

    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    seen_questions = {_question_key(question.question) for question in questions}
    out = list(questions)

    for slot in plan.slots:
        if len(out) >= target_count:
            break

        chunk = _chunk_for_slot(slot, concept_to_chunk_ids, chunks_by_id, chunks)
        if chunk is None:
            continue

        fallback = _fallback_question_for_slot(slot, chunk, chunks)
        if fallback is None:
            continue

        key = _question_key(fallback.question)
        if key in seen_questions:
            continue
        if not is_valid(fallback, chunks_by_id):
            continue

        seen_questions.add(key)
        out.append(fallback)

    return out


def _chunk_for_slot(
    slot: QuestionSlot,
    concept_to_chunk_ids: dict[str, list[str]],
    chunks_by_id: dict[str, Chunk],
    chunks: list[Chunk],
) -> Chunk | None:
    for chunk_id in concept_to_chunk_ids.get(slot.concept, []):
        if chunk_id in chunks_by_id:
            return chunks_by_id[chunk_id]
    return chunks[0] if chunks else None


def _fallback_question_for_slot(
    slot: QuestionSlot,
    chunk: Chunk,
    chunks: list[Chunk],
) -> Question | None:
    if slot.kind == "mcq":
        mcq = _fallback_mcq(slot, chunk, chunks)
        if mcq is not None:
            return mcq
    return _fallback_true_false(slot, chunk)


def _fallback_mcq(
    slot: QuestionSlot,
    chunk: Chunk,
    chunks: list[Chunk],
) -> MCQ | None:
    correct = _best_statement(chunk, slot.concept)
    if correct is None:
        return None

    distractors = _distractor_statements(
        chunks,
        source_chunk_id=chunk.chunk_id,
        correct=correct,
    )
    if len(distractors) < 3:
        return None

    concept = _clean_concept_label(slot.concept)
    return MCQ(
        bloom_level=slot.bloom_level,
        question=f"Which statement best describes {concept}?",
        options=[correct, *distractors[:3]],
        correct_index=0,
        explanation=f"Right - that statement matches the lesson's explanation of {concept}.",
        concept=concept,
        source_chunk_id=chunk.chunk_id,
    )


def _fallback_true_false(slot: QuestionSlot, chunk: Chunk) -> TrueFalse | None:
    statement = _best_statement(chunk, slot.concept)
    if statement is None:
        return None

    concept = _clean_concept_label(slot.concept)
    return TrueFalse(
        bloom_level=slot.bloom_level,
        question=statement,
        answer=True,
        explanation="The lesson states this directly, so the statement is true.",
        concept=concept,
        source_chunk_id=chunk.chunk_id,
    )


def _best_statement(chunk: Chunk, concept: str) -> str | None:
    candidates = _statement_candidates(chunk)
    if not candidates:
        return None

    terms = [term for term in _WORD_RE.findall(concept.casefold()) if len(term) > 3]
    if not terms:
        return candidates[0]

    return max(
        candidates,
        key=lambda statement: sum(term in statement.casefold() for term in terms),
    )


def _distractor_statements(
    chunks: list[Chunk],
    *,
    source_chunk_id: str,
    correct: str,
) -> list[str]:
    out: list[str] = []
    seen = {_question_key(correct)}
    ordered_chunks = [
        *[chunk for chunk in chunks if chunk.chunk_id != source_chunk_id],
        *[chunk for chunk in chunks if chunk.chunk_id == source_chunk_id],
    ]
    for chunk in ordered_chunks:
        for statement in _statement_candidates(chunk):
            key = _question_key(statement)
            if key in seen:
                continue
            seen.add(key)
            out.append(statement)
            if len(out) >= 3:
                return out
    return out


def _statement_candidates(chunk: Chunk) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for piece in _SENTENCE_SPLIT_RE.split(chunk.text or ""):
        statement = _clean_statement(piece)
        if not statement:
            continue
        key = _question_key(statement)
        if key in seen:
            continue
        seen.add(key)
        out.append(statement)
    return out


def _clean_statement(value: str) -> str:
    text = " ".join(str(value or "").split()).strip(" -*#:;")
    text = re.sub(r"^\d+(?:\.\d+)*\s+", "", text).strip()
    if text.endswith("?") or len(text) < 24:
        return ""
    if len(text) > 180:
        text = text[:177].rsplit(" ", 1)[0].rstrip(",;:")
    if not text.endswith((".", "!")):
        text = f"{text}."
    return text


def _question_key(value: str) -> str:
    return " ".join(_WORD_RE.findall(str(value or "").casefold()))


async def run(inp: GeneratorInput) -> AsyncIterator[str]:
    settings = get_settings()

    options = dict(inp.options or {})
    set_current_llm_options(options)
    llm = get_llm_service()
    set_current_language(options.get("language"))
    n_target = _resolve_question_count(options)
    allowed_kinds = _resolve_allowed_kinds(options)
    learner = inp.learner_state

    yield _sse(
        "progress",
        {"stage": "extracting_concepts", "chunks": len(inp.context_chunks)},
    )

    extracted = await extract_concepts(inp.context_chunks, llm)
    total_concepts = sum(
        len(getattr(extracted, lvl)) for lvl in ("remember", "understand", "apply", "analyze")
    )
    yield _sse("progress", {"stage": "concepts_extracted", "count": total_concepts})

    concept_to_chunk_ids = _concept_to_chunk_ids(extracted)
    plan = plan_question_mix(
        learner_state=learner,
        extracted=extracted,
        n_total=n_target,
        allowed_kinds=allowed_kinds,
    )
    yield _sse(
        "progress",
        {"stage": "planned", "total": plan.total, "counts": plan.counts},
    )

    if plan.total == 0:
        empty_response = (
            "I couldn't pull enough quizzable material from your sources for this topic — "
            "try uploading more on it, or ask me to chat about it first."
        )
        yield _sse("token", {"delta": empty_response})
        yield _sse(
            "done",
            GeneratorOutput(
                response=empty_response,
                generator_id=settings.generator_id,
                output_type=settings.output_type,
                sources=inp.context_chunks,
                metadata={"reason": "no_concepts_extracted"},
            ).model_dump(),
        )
        return

    raw_questions = await generate_questions(
        slots=plan.slots,
        concept_to_chunk_ids=concept_to_chunk_ids,
        all_chunks=inp.context_chunks,
        llm=llm,
    )
    yield _sse("progress", {"stage": "generated", "count": len(raw_questions)})

    if settings.enhance_distractors:
        enhanced = await _enhance_mcqs(raw_questions, inp.context_chunks)
        yield _sse("progress", {"stage": "distractors_enhanced"})
    else:
        enhanced = raw_questions

    enhanced = _repair_obvious_concept_labels(enhanced)
    enhanced = _attach_canonical_concept_ids(enhanced, learner.known_concepts)
    validated, dropped = validate_questions(enhanced, inp.context_chunks)
    questions, duplicates = deduplicate_questions(validated)
    dropped.extend(duplicates)
    before_top_up = len(questions)
    questions = _top_up_with_grounded_questions(
        questions=questions,
        plan=plan,
        concept_to_chunk_ids=concept_to_chunk_ids,
        chunks=inp.context_chunks,
        target_count=n_target,
    )
    questions = _attach_canonical_concept_ids(questions, learner.known_concepts)
    top_up_count = len(questions) - before_top_up
    bloom_counts = bloom_distribution(questions)
    yield _sse(
        "progress",
        {
            "stage": "validated",
            "kept": len(questions),
            "dropped": len(dropped),
            "top_up": top_up_count,
        },
    )

    if not questions:
        empty_response = (
            "I found your sources, but I couldn't make quiz questions that passed "
            "the quality checks. Try selecting more source material, then generate again."
        )
        yield _sse("token", {"delta": empty_response})
        yield _sse(
            "done",
            GeneratorOutput(
                response=empty_response,
                generator_id=settings.generator_id,
                output_type=settings.output_type,
                sources=inp.context_chunks,
                metadata={
                    "reason": "no_questions_after_validation",
                    "plan": plan.model_dump(),
                    "dropped_questions": dropped,
                    "top_up_questions": top_up_count,
                },
            ).model_dump(),
        )
        return

    intro_message = await _build_intro(
        llm=llm,
        learner_struggling=learner.struggling_concepts,
        learner_understood=learner.understood_concepts,
        plan=plan,
        bloom_counts=bloom_counts,
    )

    quiz = QuizOutput(
        title=_resolve_title(options, learner.struggling_concepts),
        intro_message=intro_message,
        questions=questions,
        bloom_distribution=bloom_counts,
    )

    payload = quiz.model_dump_json(indent=2).encode("utf-8")
    store = get_artifact_store()
    try:
        key, url = await store.save_json(
            conversation_id=inp.conversation_id,
            filename="quiz.json",
            payload=payload,
        )
        artifact = GeneratorArtifact(
            type="quiz", url=url, filename="quiz.json", key=key
        )
        artifacts = [artifact]
    except Exception as exc:  # storage outage shouldn't kill the quiz
        artifacts = []
        yield _sse("progress", {"stage": "artifact_upload_failed", "error": str(exc)})

    response_text = f"{intro_message}\n\nSee the quiz below."
    yield _sse("token", {"delta": response_text})

    concepts_covered = sorted({q.concept for q in questions if q.concept})
    learner_updates = LearnerUpdates(concepts_covered=concepts_covered)

    output = GeneratorOutput(
        response=response_text,
        generator_id=settings.generator_id,
        output_type=settings.output_type,
        artifacts=artifacts,
        sources=inp.context_chunks,
        learner_updates=learner_updates,
        metadata={
            "quiz_data": quiz.model_dump(),
            "plan": plan.model_dump(),
            "bloom_distribution": bloom_counts,
            "dropped_questions": dropped,
            "top_up_questions": top_up_count,
        },
    )
    yield _sse("done", output.model_dump())
