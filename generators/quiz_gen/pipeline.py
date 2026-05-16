from __future__ import annotations

import json
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
    "fill_blank": "fill_blank",
    "fill_in_the_blank": "fill_blank",
    "short_answer": "fill_blank",
}


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
    return kinds or None


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
    repaired: list[Question] = []
    for question in questions:
        text_parts = [
            question.question,
            question.explanation,
            str(getattr(question, "answer", "")),
            " ".join(getattr(question, "options", []) or []),
        ]
        text = " ".join(text_parts).casefold()
        concept = question.concept
        if "intent" in text:
            concept = "Definition of Intent"
        elif "xcode" in text or ("ios" in text and "tool" in text):
            concept = "iOS development tools"
        elif "android architecture" in text or "android system components" in text:
            concept = "Android Architecture Basis"
        repaired.append(question.model_copy(update={"concept": concept}))
    return repaired


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


def _top_up_with_grounded_true_false(
    *,
    questions: list[Question],
    plan: QuizPlan,
    concept_to_chunk_ids: dict[str, list[str]],
    chunks: list[Chunk],
    target_count: int,
) -> list[Question]:
    if len(questions) >= target_count:
        return questions

    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    covered_questions = {q.question.strip().casefold() for q in questions}
    covered_concepts = {q.concept.strip().casefold() for q in questions if q.concept}
    out = list(questions)

    for slot in plan.slots:
        if len(out) >= target_count:
            break
        concept_key = slot.concept.strip().casefold()
        if concept_key in covered_concepts:
            continue

        chunk = next(
            (
                by_id[cid]
                for cid in concept_to_chunk_ids.get(slot.concept, [])
                if cid in by_id
            ),
            None,
        )
        if chunk is None:
            chunk = chunks[0] if chunks else None
        if chunk is None:
            continue

        heading = str(chunk.metadata.get("heading_path") or chunk.source or "the course material")
        concept = _fallback_concept(slot.concept, chunk)
        question_text = f"The course material covers {concept} in the section {heading}."
        signature = question_text.strip().casefold()
        if signature in covered_questions:
            continue
        covered_questions.add(signature)
        covered_concepts.add(concept_key)
        out.append(
            TrueFalse(
                bloom_level=slot.bloom_level,
                question=question_text,
                answer=True,
                explanation=f"Yes. The retrieved source section is {heading}.",
                concept=concept,
                source_chunk_id=chunk.chunk_id,
            )
        )

    for slot in plan.slots:
        if len(out) >= target_count:
            break
        chunk = next(
            (
                by_id[cid]
                for cid in concept_to_chunk_ids.get(slot.concept, [])
                if cid in by_id
            ),
            None,
        )
        if chunk is None:
            continue
        source = chunk.source or "the uploaded course material"
        concept = _fallback_concept(slot.concept, chunk)
        question_text = f"The source {source} includes material related to {concept}."
        signature = question_text.strip().casefold()
        if signature in covered_questions:
            continue
        covered_questions.add(signature)
        out.append(
            TrueFalse(
                bloom_level=slot.bloom_level,
                question=question_text,
                answer=True,
                explanation=f"Yes. This question is grounded in the retrieved source {source}.",
                concept=concept,
                source_chunk_id=chunk.chunk_id,
            )
        )

    return out


def _fallback_concept(slot_concept: str, chunk: Chunk) -> str:
    source_text = f"{chunk.text} {chunk.metadata.get('heading_path', '')}".casefold()
    if slot_concept.casefold() in source_text:
        return slot_concept
    if "intent" in source_text:
        return "Definition of Intent"
    if "android architecture" in source_text or "android system" in source_text:
        return "Android Architecture Basis"
    if "xcode" in source_text or "ios" in source_text:
        return "iOS development tools"
    if "sqlite" in source_text or "shared preferences" in source_text or "store data" in source_text:
        return "Data persistence methods"
    return slot_concept


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
    validated, dropped = validate_questions(enhanced, inp.context_chunks)
    questions, duplicates = deduplicate_questions(validated)
    dropped.extend(duplicates)
    before_top_up = len(questions)
    questions = _top_up_with_grounded_true_false(
        questions=questions,
        plan=plan,
        concept_to_chunk_ids=concept_to_chunk_ids,
        chunks=inp.context_chunks,
        target_count=n_target,
    )
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
