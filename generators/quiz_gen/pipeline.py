from __future__ import annotations

import json
from collections.abc import AsyncIterator

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
    QuizOutput,
    QuizPlan,
)
from .services.artifact_store import get_artifact_store
from .services.concept_extractor import extract_concepts
from .services.difficulty_adapter import plan_question_mix
from .services.distractor_engine import enhance_mcq_distractors
from .services.llm_service import LLMService, build_system_prompt, get_llm_service
from .services.quality_validator import bloom_distribution, validate_questions
from .services.question_generator import generate_questions


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _resolve_question_count(options: dict) -> int:
    s = get_settings()
    raw = options.get("n_questions") or options.get("count") or s.default_question_count
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = s.default_question_count
    return max(s.min_question_count, min(n, s.max_question_count))


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


async def run(inp: GeneratorInput) -> AsyncIterator[str]:
    settings = get_settings()
    llm = get_llm_service()

    options = dict(inp.options or {})
    n_target = _resolve_question_count(options)
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

    plan = plan_question_mix(
        learner_state=learner,
        extracted=extracted,
        n_total=n_target,
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
        concept_to_chunk_ids=_concept_to_chunk_ids(extracted),
        all_chunks=inp.context_chunks,
        llm=llm,
    )
    yield _sse("progress", {"stage": "generated", "count": len(raw_questions)})

    enhanced = await _enhance_mcqs(raw_questions, inp.context_chunks)
    yield _sse("progress", {"stage": "distractors_enhanced"})

    questions, dropped = validate_questions(enhanced)
    bloom_counts = bloom_distribution(questions)
    yield _sse(
        "progress",
        {"stage": "validated", "kept": len(questions), "dropped": len(dropped)},
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
        _, url = await store.save_json(
            conversation_id=inp.conversation_id,
            filename="quiz.json",
            payload=payload,
        )
        artifact = GeneratorArtifact(type="quiz", url=url, filename="quiz.json")
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
        },
    )
    yield _sse("done", output.model_dump())
