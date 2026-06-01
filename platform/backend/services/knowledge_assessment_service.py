from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from teacherlm_core.llm.ollama_client import OllamaClient
from teacherlm_core.llm.runtime import build_llm_client_kwargs

from config import Settings, get_settings
from db.models import (
    CourseConceptRecord,
    CourseLearningObjectiveRecord,
    CourseLearningPhaseRecord,
    KnowledgeAttemptRecord,
    KnowledgeCheckRecord,
    SearchChunkRecord,
)
from schemas.knowledge_check import (
    KnowledgeCheckQuestion,
    KnowledgeCheckResult,
    KnowledgeCheckStartResponse,
    QuizAttemptQuestion,
    QuizAttemptResponse,
)
from services.concept_inventory_service import get_concept_inventory_service, resolve_concept
from services.learning_map_service import get_learning_map_service
from services.learner_tracker import get_learner_tracker
from services.knowledge_graph_service import get_knowledge_graph_service


logger = logging.getLogger(__name__)

_QUESTION_TYPES = ["mcq", "true_false", "fill_blank", "short_answer"]
_WORD_RE = re.compile(r"[a-z0-9]+")
_LOCAL_FALLBACK_MODEL = "gemma4:e2b"


class _ShortAnswerGrade(BaseModel):
    score: float = Field(ge=0.0, le=1.0)
    feedback: str = ""


class _GeneratedCheck(BaseModel):
    question_type: str
    bloom_level: str = "understand"
    prompt: str
    options: list[str] = Field(default_factory=list)
    correct_index: int | None = None
    answer: str | bool | None = None
    accepted_answers: list[str] = Field(default_factory=list)
    rubric: str = ""


class KnowledgeAssessmentService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def ensure_schema(self, session: AsyncSession) -> None:
        connection = await session.connection()

        def create_tables(sync_connection) -> None:  # noqa: ANN001
            KnowledgeCheckRecord.__table__.create(sync_connection, checkfirst=True)
            KnowledgeAttemptRecord.__table__.create(sync_connection, checkfirst=True)

        await connection.run_sync(create_tables)

    async def start_checks(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        concept_id: uuid.UUID | None = None,
        phase_id: uuid.UUID | None = None,
        objective_id: uuid.UUID | None = None,
        count: int = 1,
        question_types: list[str] | None = None,
        llm_options: dict[str, Any] | None = None,
    ) -> KnowledgeCheckStartResponse:
        await self.ensure_schema(session)
        concepts = await get_concept_inventory_service().load_concepts(session, conversation_id)
        phases, objectives = await get_learning_map_service().load_map(session, conversation_id)
        selected = await self._select_concepts(
            session,
            conversation_id,
            concepts,
            phases,
            objectives,
            concept_id,
            phase_id,
            objective_id,
            count,
        )
        chunks = await self._load_chunks(session, conversation_id)
        checks: list[KnowledgeCheckRecord] = []
        for index, concept in enumerate(selected):
            qtype = self._choose_question_type(index, concept, question_types)
            phase, objective = _learning_context_for_concept(concept, phases, objectives, phase_id, objective_id)
            check = await self._build_check(
                conversation_id,
                concept,
                chunks,
                qtype,
                concepts,
                phase=phase,
                objective=objective,
                llm_options=llm_options,
            )
            session.add(check)
            checks.append(check)
        await session.flush()
        state = await get_learner_tracker().load_state(session, conversation_id)
        return KnowledgeCheckStartResponse(
            checks=[self._to_question(check, concepts) for check in checks],
            learner_state=state,
        )

    async def submit_check(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        check_id: uuid.UUID,
        answer: Any,
        *,
        llm_options: dict[str, Any] | None = None,
        question_index: int | None = None,
    ) -> tuple[KnowledgeCheckResult, Any]:
        await self.ensure_schema(session)
        check = await session.get(KnowledgeCheckRecord, check_id)
        if check is None or check.conversation_id != conversation_id:
            raise LookupError("knowledge check not found")

        concepts = await get_concept_inventory_service().load_concepts(session, conversation_id)
        concept = next((item for item in concepts if item.id == check.concept_id), None)
        if concept is None:
            raise LookupError("knowledge check concept not found")

        score, is_correct, feedback = await self._grade_check(
            check,
            concept,
            answer,
            llm_options=llm_options,
        )
        progress = await get_learner_tracker().apply_assessment_result(
            session,
            conversation_id,
            check.concept_id,
            score=score,
            bloom_level=check.bloom_level,
        )
        remediation_paths: list[dict[str, Any]] = []
        if not is_correct:
            try:
                remediation_paths = await get_knowledge_graph_service().remediation_for_wrong_answer(
                    session,
                    conversation_id,
                    check.concept_id,
                )
                progress.state.remediation_paths = remediation_paths
            except Exception:  # noqa: BLE001
                logger.exception("knowledge graph remediation failed; continuing assessment")
        attempt = KnowledgeAttemptRecord(
            conversation_id=conversation_id,
            check_id=check.id,
            concept_id=check.concept_id,
            answer={"value": answer},
            score=score,
            is_correct=is_correct,
            feedback=feedback,
            evidence_strength=progress.evidence_strength,
            mastery_delta=progress.mastery_delta,
            attempt_metadata={
                "source": "knowledge_check",
                "question_index": question_index,
                "remediation_paths": remediation_paths,
            },
            created_at=datetime.now(timezone.utc),
        )
        session.add(attempt)
        await session.flush()
        return (
            KnowledgeCheckResult(
                check_id=check.id,
                concept_id=concept.id,
                concept_name=concept.canonical_name,
                question_index=question_index,
                score=score,
                is_correct=is_correct,
                feedback=feedback,
                evidence_strength=progress.evidence_strength,
                mastery_delta=progress.mastery_delta,
                remediation_paths=remediation_paths,
            ),
            progress.state,
        )

    async def submit_quiz_attempt(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        questions: list[QuizAttemptQuestion],
        answers_by_index: dict[int, Any],
        *,
        llm_options: dict[str, Any] | None = None,
    ) -> QuizAttemptResponse:
        await self.ensure_schema(session)
        concepts = await get_concept_inventory_service().load_concepts(session, conversation_id)
        chunks = await self._load_chunks(session, conversation_id)
        results: list[KnowledgeCheckResult] = []
        latest_state = await get_learner_tracker().load_state(session, conversation_id)

        for index, question in enumerate(questions):
            if index not in answers_by_index:
                continue
            concept = self._resolve_quiz_concept(question, concepts)
            if concept is None:
                continue
            check = self._check_from_quiz_question(conversation_id, question, concept, chunks)
            session.add(check)
            await session.flush()
            result, latest_state = await self.submit_check(
                session,
                conversation_id,
                check.id,
                answers_by_index[index],
                llm_options=llm_options,
                question_index=index,
            )
            results.append(result)

        score = sum(1 for result in results if result.is_correct)
        return QuizAttemptResponse(
            results=results,
            learner_state=latest_state,
            score=float(score),
            total=len(questions),
        )

    async def _select_concepts(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        concepts: list[CourseConceptRecord],
        phases: list[CourseLearningPhaseRecord],
        objectives: list[CourseLearningObjectiveRecord],
        concept_id: uuid.UUID | None,
        phase_id: uuid.UUID | None,
        objective_id: uuid.UUID | None,
        count: int,
    ) -> list[CourseConceptRecord]:
        concepts_by_id = {str(concept.id): concept for concept in concepts}
        if concept_id is not None:
            return [concept for concept in concepts if concept.id == concept_id][:1]
        if objective_id is not None:
            objective = next((item for item in objectives if item.id == objective_id), None)
            if objective is None:
                return []
            return [
                concepts_by_id[concept_id]
                for concept_id in objective.concept_ids
                if concept_id in concepts_by_id
            ][:count]
        if phase_id is not None:
            selected: list[CourseConceptRecord] = []
            for objective in sorted(
                [item for item in objectives if item.phase_id == phase_id],
                key=lambda item: item.order_index,
            ):
                for raw_id in objective.concept_ids:
                    concept = concepts_by_id.get(str(raw_id))
                    if concept is not None and concept not in selected:
                        selected.append(concept)
            return selected[:count]

        state = await get_learner_tracker().load_state(session, conversation_id)
        progress = {item.concept_id: item for item in state.concept_progress}
        if objectives:
            objective_rank = sorted(
                state.objective_progress,
                key=lambda item: (
                    item.mastery,
                    item.encounters,
                    item.order_index,
                    item.objective_text.casefold(),
                ),
            )
            selected: list[CourseConceptRecord] = []
            for objective_progress in objective_rank:
                objective = next(
                    (item for item in objectives if str(item.id) == objective_progress.objective_id),
                    None,
                )
                if objective is None:
                    continue
                for raw_id in objective.concept_ids:
                    concept = concepts_by_id.get(str(raw_id))
                    if concept is not None and concept not in selected:
                        selected.append(concept)
                    if len(selected) >= count:
                        return selected
        return sorted(
            concepts,
            key=lambda concept: (
                progress.get(str(concept.id)).mastery if progress.get(str(concept.id)) else 0.0,
                progress.get(str(concept.id)).encounters if progress.get(str(concept.id)) else 0,
                -float(concept.importance),
                concept.canonical_name.casefold(),
            ),
        )[:count]

    async def _load_chunks(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> list[SearchChunkRecord]:
        result = await session.execute(
            select(SearchChunkRecord)
            .where(SearchChunkRecord.conversation_id == conversation_id)
            .order_by(SearchChunkRecord.document_id, SearchChunkRecord.chunk_index)
        )
        return list(result.scalars().all())

    def _choose_question_type(
        self,
        index: int,
        concept: CourseConceptRecord,
        question_types: list[str] | None,
    ) -> str:
        allowed = [item for item in (question_types or _QUESTION_TYPES) if item in _QUESTION_TYPES]
        if not allowed:
            allowed = _QUESTION_TYPES
        if concept.bloom_level in {"apply", "analyze"} and "short_answer" in allowed:
            return "short_answer"
        return allowed[index % len(allowed)]

    async def _build_check(
        self,
        conversation_id: uuid.UUID,
        concept: CourseConceptRecord,
        chunks: list[SearchChunkRecord],
        question_type: str,
        concepts: list[CourseConceptRecord],
        *,
        phase: CourseLearningPhaseRecord | None = None,
        objective: CourseLearningObjectiveRecord | None = None,
        llm_options: dict[str, Any] | None = None,
    ) -> KnowledgeCheckRecord:
        source_chunks = self._concept_chunks(concept, chunks)
        description = concept.description or self._snippet(source_chunks)
        alternatives = [
            item.canonical_name
            for item in concepts
            if item.id != concept.id and _normalize_answer(item.canonical_name) != _normalize_answer(concept.canonical_name)
        ][:8]
        try:
            generated = await self._generate_check(
                concept=concept,
                source_chunks=source_chunks,
                question_type=question_type,
                alternatives=alternatives,
                llm_options=llm_options,
            )
            question_type, prompt, options, answer_key, rubric = _normalize_generated_check(
                generated,
                concept,
                requested_type=question_type,
                fallback_description=description,
            )
        except Exception:  # noqa: BLE001
            logger.exception("knowledge-check generation failed; using deterministic fallback")
            question_type, prompt, options, answer_key, rubric = _fallback_check_data(
                concept=concept,
                question_type=question_type,
                description=description,
                alternatives=alternatives,
            )
        generation_method = "llm" if answer_key.pop("_generated", False) else "fallback"

        return KnowledgeCheckRecord(
            conversation_id=conversation_id,
            concept_id=concept.id,
            question_type=question_type,
            bloom_level=concept.bloom_level if concept.bloom_level in {"remember", "understand", "apply", "analyze"} else "understand",
            prompt=prompt[:1200],
            options=options,
            answer_key=answer_key,
            rubric=(rubric or description)[:1200],
            source_chunk_ids=[chunk.id for chunk in source_chunks[:3]],
            check_metadata={
                "source": "knowledge_check",
                "generation": generation_method,
                "phase_id": str(phase.id) if phase else None,
                "objective_id": str(objective.id) if objective else None,
            },
            created_at=datetime.now(timezone.utc),
        )

    async def _generate_check(
        self,
        *,
        concept: CourseConceptRecord,
        source_chunks: list[SearchChunkRecord],
        question_type: str,
        alternatives: list[str],
        llm_options: dict[str, Any] | None,
    ) -> _GeneratedCheck:
        source_text = "\n\n".join(
            f"[{chunk.id}] {' '.join(chunk.text.split())[:900]}"
            for chunk in source_chunks[:3]
        )
        user_prompt = (
            f"Concept: {concept.canonical_name}\n"
            f"Aliases: {', '.join(concept.aliases or []) or '(none)'}\n"
            f"Bloom level: {concept.bloom_level}\n"
            f"Requested question_type: {question_type}\n"
            f"Possible distractor concepts: {', '.join(alternatives) or '(none)'}\n\n"
            f"Course source snippets:\n{source_text}"
        )
        last_error: Exception | None = None
        for label, client in self._llm_clients(llm_options):
            try:
                return await client.chat_structured(
                    messages=[
                        {"role": "system", "content": _CHECK_GENERATION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    schema=_GeneratedCheck,
                    options={"temperature": 0.15, "num_predict": 900, "max_tokens": 900},
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "knowledge-check generation with %s model %s failed",
                    label,
                    client.model,
                    exc_info=True,
                )
        if last_error is not None:
            raise last_error
        raise RuntimeError("no LLM clients available for knowledge-check generation")

    def _check_from_quiz_question(
        self,
        conversation_id: uuid.UUID,
        question: QuizAttemptQuestion,
        concept: CourseConceptRecord,
        chunks: list[SearchChunkRecord],
    ) -> KnowledgeCheckRecord:
        source_chunks = self._concept_chunks(concept, chunks)
        qtype = question.type if question.type in _QUESTION_TYPES else "short_answer"
        answer_key: dict[str, Any] = {"concept_name": concept.canonical_name}
        options = list(question.options or [])
        if qtype == "mcq":
            answer_key["correct_index"] = question.correct_index
        elif qtype == "true_false":
            answer_key["answer"] = bool(question.answer)
            options = ["True", "False"]
        elif qtype == "fill_blank":
            answer_key["answer"] = question.answer
            answer_key["accepted_answers"] = [*question.accepted_answers, str(question.answer or "")]
        else:
            answer_key["expected_terms"] = _expected_terms(concept)
        return KnowledgeCheckRecord(
            conversation_id=conversation_id,
            concept_id=concept.id,
            question_type=qtype,
            bloom_level=_coerce_bloom(str(question.bloom_level or concept.bloom_level)),
            prompt=question.question,
            options=options,
            answer_key=answer_key,
            rubric=question.explanation or concept.description,
            source_chunk_ids=[question.source_chunk_id] if question.source_chunk_id else [chunk.id for chunk in source_chunks[:3]],
            check_metadata={
                "source": "quiz_attempt",
                "phase_id": str(question.phase_id) if question.phase_id else None,
                "objective_id": str(question.objective_id) if question.objective_id else None,
            },
            created_at=datetime.now(timezone.utc),
        )

    async def _grade_check(
        self,
        check: KnowledgeCheckRecord,
        concept: CourseConceptRecord,
        answer: Any,
        *,
        llm_options: dict[str, Any] | None = None,
    ) -> tuple[float, bool, str]:
        if check.question_type == "mcq":
            expected_index = check.answer_key.get("correct_index")
            selected_index = _coerce_index(answer, check.options)
            is_correct = selected_index is not None and selected_index == expected_index
            return (1.0 if is_correct else 0.0, is_correct, "Correct." if is_correct else "Review this concept and try another check.")
        if check.question_type == "true_false":
            expected = bool(check.answer_key.get("answer"))
            given = _coerce_bool(answer)
            is_correct = given is not None and given == expected
            return (1.0 if is_correct else 0.0, is_correct, "Correct." if is_correct else "The answer does not match the course statement.")
        if check.question_type == "fill_blank":
            accepted = [str(check.answer_key.get("answer") or ""), *list(check.answer_key.get("accepted_answers") or [])]
            is_correct = _normalize_answer(str(answer)) in {_normalize_answer(item) for item in accepted if item}
            return (1.0 if is_correct else 0.0, is_correct, "Correct." if is_correct else f"The expected concept is {concept.canonical_name}.")
        return await self._grade_short_answer(check, concept, answer, llm_options=llm_options)

    async def _grade_short_answer(
        self,
        check: KnowledgeCheckRecord,
        concept: CourseConceptRecord,
        answer: Any,
        *,
        llm_options: dict[str, Any] | None = None,
    ) -> tuple[float, bool, str]:
        text = str(answer or "").strip()
        if not text:
            return 0.0, False, "No answer was provided."
        last_error: Exception | None = None
        for label, client in self._llm_clients(llm_options):
            try:
                grade = await client.chat_structured(
                    messages=[
                        {"role": "system", "content": _GRADING_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Concept: {concept.canonical_name}\n"
                                f"Rubric/source: {check.rubric}\n"
                                f"Student answer: {text}"
                            ),
                        },
                    ],
                    schema=_ShortAnswerGrade,
                    options={"temperature": 0.0, "num_predict": 400, "max_tokens": 400},
                )
                score = max(0.0, min(1.0, float(grade.score)))
                return score, score >= 0.6, grade.feedback or _feedback_for_score(score, concept)
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "short-answer grading with %s model %s failed",
                    label,
                    client.model,
                    exc_info=True,
                )
        if last_error is not None:
            logger.warning("short-answer grading failed after all LLM attempts; using deterministic fallback")
        score = _heuristic_short_answer_score(text, concept, check)
        return score, score >= 0.6, _feedback_for_score(score, concept)

    def _llm_clients(self, llm_options: dict[str, Any] | None) -> list[tuple[str, OllamaClient]]:
        raw_llm = llm_options.get("llm") if isinstance(llm_options, dict) else None
        primary = build_llm_client_kwargs(
            default_base_url=self._settings.ollama_host,
            default_model=self._settings.ollama_chat_model,
            options=raw_llm if isinstance(raw_llm, dict) else None,
        )
        clients = [
            (
                "configured",
                OllamaClient(
                    str(primary["base_url"]),
                    str(primary["model"]),
                    provider=str(primary["provider"]),
                    api_key=primary["api_key"],
                ),
            )
        ]
        if not (
            primary["provider"] == "ollama"
            and primary["base_url"] == self._settings.ollama_host
            and primary["model"] == _LOCAL_FALLBACK_MODEL
        ):
            clients.append(
                (
                    "local-fallback",
                    OllamaClient(
                        self._settings.ollama_host,
                        _LOCAL_FALLBACK_MODEL,
                        provider="ollama",
                    ),
                )
            )
        return clients

    def _concept_chunks(
        self,
        concept: CourseConceptRecord,
        chunks: list[SearchChunkRecord],
    ) -> list[SearchChunkRecord]:
        wanted = set(str(item) for item in (concept.source_chunk_ids or []))
        matched = [chunk for chunk in chunks if chunk.id in wanted]
        return matched or chunks[:3]

    def _snippet(self, chunks: list[SearchChunkRecord]) -> str:
        for chunk in chunks:
            text = " ".join(chunk.text.split())
            if text:
                return text[:320]
        return "an important idea from the uploaded course material."

    def _resolve_quiz_concept(
        self,
        question: QuizAttemptQuestion,
        concepts: list[CourseConceptRecord],
    ) -> CourseConceptRecord | None:
        if question.concept_id is not None:
            direct = next((concept for concept in concepts if concept.id == question.concept_id), None)
            if direct is not None:
                return direct
        if question.concept:
            return resolve_concept(question.concept, concepts)
        return None

    def _to_question(
        self,
        check: KnowledgeCheckRecord,
        concepts: list[CourseConceptRecord],
    ) -> KnowledgeCheckQuestion:
        concept = next((item for item in concepts if item.id == check.concept_id), None)
        metadata = check.check_metadata or {}
        return KnowledgeCheckQuestion(
            id=check.id,
            concept_id=check.concept_id,
            concept_name=concept.canonical_name if concept else "Course concept",
            phase_id=uuid.UUID(str(metadata["phase_id"])) if metadata.get("phase_id") else None,
            objective_id=uuid.UUID(str(metadata["objective_id"])) if metadata.get("objective_id") else None,
            question_type=check.question_type,  # type: ignore[arg-type]
            bloom_level=_coerce_bloom(check.bloom_level),  # type: ignore[arg-type]
            prompt=check.prompt,
            options=list(check.options or []),
            source_chunk_ids=list(check.source_chunk_ids or []),
        )


def _expected_terms(concept: CourseConceptRecord) -> list[str]:
    terms = [concept.canonical_name, *list(concept.aliases or [])]
    if concept.description:
        terms.extend(_WORD_RE.findall(concept.description.casefold())[:8])
    return _dedupe(terms)[:12]


def _learning_context_for_concept(
    concept: CourseConceptRecord,
    phases: list[CourseLearningPhaseRecord],
    objectives: list[CourseLearningObjectiveRecord],
    requested_phase_id: uuid.UUID | None,
    requested_objective_id: uuid.UUID | None,
) -> tuple[CourseLearningPhaseRecord | None, CourseLearningObjectiveRecord | None]:
    concept_id = str(concept.id)
    objective = None
    if requested_objective_id is not None:
        objective = next((item for item in objectives if item.id == requested_objective_id), None)
    if objective is None:
        candidates = [
            item
            for item in objectives
            if concept_id in [str(raw_id) for raw_id in (item.concept_ids or [])]
            and (requested_phase_id is None or item.phase_id == requested_phase_id)
        ]
        objective = sorted(candidates, key=lambda item: item.order_index)[0] if candidates else None

    phase = None
    if requested_phase_id is not None:
        phase = next((item for item in phases if item.id == requested_phase_id), None)
    if phase is None and objective is not None:
        phase = next((item for item in phases if item.id == objective.phase_id), None)
    return phase, objective


def _normalize_generated_check(
    generated: _GeneratedCheck,
    concept: CourseConceptRecord,
    *,
    requested_type: str,
    fallback_description: str,
) -> tuple[str, str, list[str], dict[str, Any], str]:
    qtype = requested_type if requested_type == "mcq" else generated.question_type
    if qtype not in _QUESTION_TYPES:
        qtype = requested_type
    prompt = " ".join(generated.prompt.split()).strip()
    rubric = " ".join((generated.rubric or fallback_description).split()).strip()
    if not prompt:
        raise ValueError("generated question has no prompt")

    answer_key: dict[str, Any] = {"concept_name": concept.canonical_name, "_generated": True}
    if qtype == "mcq":
        options = _dedupe([str(item) for item in generated.options])[:6]
        correct_index = generated.correct_index
        if correct_index is None or not 0 <= correct_index < len(options) or len(options) < 3:
            raise ValueError("generated MCQ has invalid options or correct_index")
        answer_key["correct_index"] = int(correct_index)
        return qtype, prompt, options, answer_key, rubric

    if qtype == "true_false":
        if not isinstance(generated.answer, bool):
            raise ValueError("generated true_false has no boolean answer")
        answer_key["answer"] = generated.answer
        return qtype, prompt, ["True", "False"], answer_key, rubric

    if qtype == "fill_blank":
        answer = str(generated.answer or concept.canonical_name).strip()
        accepted = _dedupe([answer, *generated.accepted_answers, *list(concept.aliases or []), concept.canonical_name])
        if not accepted:
            raise ValueError("generated fill_blank has no accepted answers")
        answer_key["answer"] = answer
        answer_key["accepted_answers"] = accepted
        return qtype, prompt, [], answer_key, rubric

    qtype = "short_answer"
    answer_key["expected_terms"] = _dedupe([*generated.accepted_answers, *_expected_terms(concept)])
    return qtype, prompt, [], answer_key, rubric


def _fallback_check_data(
    *,
    concept: CourseConceptRecord,
    question_type: str,
    description: str,
    alternatives: list[str],
) -> tuple[str, str, list[str], dict[str, Any], str]:
    short_description = _clean_description(description)
    answer_key: dict[str, Any] = {"concept_name": concept.canonical_name}

    if question_type == "mcq":
        distractors = _dedupe(
            [
                *alternatives,
                f"A different course concept, not {concept.canonical_name}",
                "A general example that is not the course idea",
                "An unrelated term from outside this review",
            ]
        )
        options = _stable_shuffle(
            _dedupe([concept.canonical_name, *distractors])[:4],
            str(concept.id),
        )
        answer_key["correct_index"] = options.index(concept.canonical_name)
        prompt = f"In the course, which option best matches this idea: {short_description}"
        return "mcq", prompt, options, answer_key, short_description

    if question_type == "true_false":
        options = ["True", "False"]
        other = alternatives[0] if alternatives else ""
        if other and int(str(concept.id).replace("-", "")[-1], 16) % 2:
            answer_key["answer"] = False
            prompt = f"True or false: The course treats {other} as another name for {concept.canonical_name}."
        else:
            answer_key["answer"] = True
            prompt = f"True or false: {concept.canonical_name} is connected in the course to this idea: {short_description}"
        return "true_false", prompt, options, answer_key, short_description

    if question_type == "fill_blank":
        accepted = _dedupe([*list(concept.aliases or []), concept.canonical_name])
        answer_key["answer"] = concept.canonical_name
        answer_key["accepted_answers"] = accepted
        prompt = f"Fill in the blank using the course concept: ____ is connected to this idea: {short_description}"
        return "fill_blank", prompt, [], answer_key, short_description

    answer_key["expected_terms"] = _expected_terms(concept)
    prompt = f"Explain {concept.canonical_name} in your own words using the course material."
    return "short_answer", prompt, [], answer_key, short_description


def _clean_description(value: str) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "an important idea from the uploaded course material."
    return text[:320]


def _heuristic_short_answer_score(text: str, concept: CourseConceptRecord, check: KnowledgeCheckRecord) -> float:
    answer_terms = set(_WORD_RE.findall(text.casefold()))
    expected = set()
    for value in check.answer_key.get("expected_terms") or _expected_terms(concept):
        expected.update(_WORD_RE.findall(str(value).casefold()))
    expected = {term for term in expected if len(term) > 2}
    if not expected:
        return 0.0
    coverage = len(answer_terms & expected) / len(expected)
    length_bonus = 0.2 if len(answer_terms) >= 8 else 0.0
    return max(0.0, min(1.0, coverage + length_bonus))


def _feedback_for_score(score: float, concept: CourseConceptRecord) -> str:
    if score >= 0.85:
        return f"Strong answer on {concept.canonical_name}."
    if score >= 0.6:
        return f"Partial understanding of {concept.canonical_name}; add more course-specific detail next time."
    return f"Review {concept.canonical_name} and try another check."


def _coerce_index(answer: Any, options: list[str]) -> int | None:
    if isinstance(answer, int):
        return answer if 0 <= answer < len(options) else None
    text = str(answer or "").strip()
    if text.isdigit():
        value = int(text)
        return value if 0 <= value < len(options) else None
    normalized = _normalize_answer(text)
    for index, option in enumerate(options):
        if _normalize_answer(option) == normalized:
            return index
    return None


def _coerce_bool(answer: Any) -> bool | None:
    if isinstance(answer, bool):
        return answer
    text = str(answer or "").strip().casefold()
    if text in {"true", "t", "yes", "1"}:
        return True
    if text in {"false", "f", "no", "0"}:
        return False
    return None


def _normalize_answer(value: str) -> str:
    return " ".join(_WORD_RE.findall(str(value or "").casefold()))


def _coerce_bloom(value: str) -> str:
    text = str(value or "understand").casefold()
    return text if text in {"remember", "understand", "apply", "analyze"} else "understand"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = str(value or "").strip()
        key = _normalize_answer(text)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def _stable_shuffle(values: list[str], seed: str) -> list[str]:
    return sorted(values, key=lambda value: f"{seed}:{_normalize_answer(value)}")


_CHECK_GENERATION_SYSTEM_PROMPT = """Generate one course-grounded knowledge check for TeacherLM.

Return only JSON matching the schema.

Rules:
- Use the requested question_type unless it would be impossible from the source.
- The question must test understanding or application of the concept, not just name recognition.
- For mcq, provide 3-4 plausible options and a zero-based correct_index.
- For true_false, create a meaningful true or false course claim and set answer to a boolean.
- For fill_blank, ask for a missing course concept or term and include accepted_answers.
- For short_answer, ask the student to explain, apply, or compare using the course source.
- Do not invent facts outside the provided source snippets.
- Do not use fake options like "Not X"."""


_GRADING_SYSTEM_PROMPT = """Grade a student's short answer against the uploaded course concept.

Return only JSON matching the schema. Score from 0 to 1:
- 0.85-1.0: accurate, course-grounded explanation
- 0.60-0.84: partially correct but missing important detail
- below 0.60: incorrect, too vague, or not grounded in the course
Keep feedback short and encouraging."""


_service: KnowledgeAssessmentService | None = None


def get_knowledge_assessment_service() -> KnowledgeAssessmentService:
    global _service
    if _service is None:
        _service = KnowledgeAssessmentService()
    return _service
