from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from teacherlm_core.llm.ollama_client import OllamaClient
from teacherlm_core.llm.runtime import build_llm_client_kwargs

from config import Settings, get_settings
from db.models import (
    ChapterAttemptRecord,
    ChapterQuizRecord,
    CourseChapterRecord,
    CourseConceptRecord,
    CourseLessonBlockRecord,
    CourseLessonRecord,
    CourseLearningObjectiveRecord,
    CourseLearningPhaseRecord,
    KnowledgeCheckRecord,
    SearchChunkRecord,
)
from schemas.course_player import (
    ChapterQuizRead,
    ChapterQuizSubmitResponse,
    CourseChapterRead,
    CourseLessonBlockRead,
    CourseLessonRead,
    CoursePlayerRead,
    CoursePlayerUnlockResponse,
)
from schemas.knowledge_check import KnowledgeCheckResult
from services.concept_inventory_service import get_concept_inventory_service, resolve_concept
from services.knowledge_assessment_service import get_knowledge_assessment_service
from services.knowledge_graph_service import get_knowledge_graph_service
from services.learner_tracker import get_learner_tracker
from services.learning_map_service import get_learning_map_service, normalize_learning_key


PASS_SCORE = 0.7
_WORD_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9]{4,}")
_LOCAL_FALLBACK_MODEL = "gemma4:e2b"
_MAX_COURSE_CHAPTERS = 12
_MAX_LESSONS_PER_CHAPTER = 6
_MAX_BLOCKS_PER_LESSON = 6
_BLOCK_TYPES = {"definition", "explanation", "example", "procedure", "formula", "summary"}

logger = logging.getLogger(__name__)


class _CourseBlockCandidate(BaseModel):
    block_type: str = "explanation"
    title: str = ""
    content: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)


class _CourseLessonCandidate(BaseModel):
    title: str
    summary: str = ""
    objective_id: str | None = None
    objective_text: str = ""
    concept_names: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    blocks: list[_CourseBlockCandidate] = Field(default_factory=list)


class _CourseChapterCandidate(BaseModel):
    title: str
    summary: str = ""
    order_index: int = 0
    phase_id: str | None = None
    objective_ids: list[str] = Field(default_factory=list)
    concept_names: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)
    lessons: list[_CourseLessonCandidate] = Field(default_factory=list)


class _CoursePlanCandidateBatch(BaseModel):
    chapters: list[_CourseChapterCandidate] = Field(default_factory=list)


class CoursePlayerService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def ensure_schema(self, session: AsyncSession) -> None:
        connection = await session.connection()

        def create_tables(sync_connection) -> None:  # noqa: ANN001
            CourseChapterRecord.__table__.create(sync_connection, checkfirst=True)
            CourseLessonRecord.__table__.create(sync_connection, checkfirst=True)
            CourseLessonBlockRecord.__table__.create(sync_connection, checkfirst=True)
            ChapterQuizRecord.__table__.create(sync_connection, checkfirst=True)
            ChapterAttemptRecord.__table__.create(sync_connection, checkfirst=True)

        await connection.run_sync(create_tables)

    async def rebuild_course(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        llm_options: dict[str, Any] | None = None,  # reserved for later LLM enrichment
    ) -> CoursePlayerRead:
        await self.ensure_schema(session)
        phases, objectives = await get_learning_map_service().load_map(session, conversation_id)
        if not phases:
            phases, objectives = await get_learning_map_service().rebuild_map(
                session,
                conversation_id,
                llm_options=llm_options,
            )
        concepts = await get_concept_inventory_service().load_concepts(session, conversation_id)
        chunks = await _load_chunks(session, conversation_id)
        await get_knowledge_assessment_service().ensure_schema(session)

        existing = await _load_existing(session, conversation_id)
        now = datetime.now(timezone.utc)
        desired_chapter_ids: set[uuid.UUID] = set()
        desired_lesson_ids: set[uuid.UUID] = set()
        desired_block_ids: set[uuid.UUID] = set()
        desired_quiz_ids: set[uuid.UUID] = set()
        desired_check_ids: set[uuid.UUID] = set()
        course_plan = await self._course_plan(
            phases,
            objectives,
            concepts,
            chunks,
            llm_options=llm_options,
        )
        for chapter_candidate in course_plan:
            chapter = _chapter_from_candidate(conversation_id, chapter_candidate, phases, objectives, concepts, chunks, now)
            desired_chapter_ids.add(chapter.id)
            current = existing["chapters"].get(chapter.id)
            if current is None:
                session.add(chapter)
                current = chapter
            else:
                _copy_chapter(current, chapter)
            await session.flush()
            lessons = _lessons_for_chapter_candidate(chapter_candidate, phases, objectives)
            for lesson_candidate in lessons:
                lesson = _lesson_from_candidate(
                    conversation_id,
                    current.id,
                    lesson_candidate,
                    objectives,
                    concepts,
                    chunks,
                    now,
                )
                desired_lesson_ids.add(lesson.id)
                lesson_current = existing["lessons"].get(lesson.id)
                if lesson_current is None:
                    session.add(lesson)
                    lesson_current = lesson
                else:
                    _copy_lesson(lesson_current, lesson)
                await session.flush()
                for block in _blocks_from_candidate(
                    conversation_id,
                    lesson_current,
                    lesson_candidate,
                    concepts,
                    chunks,
                    now,
                ):
                    desired_block_ids.add(block.id)
                    block_current = existing["blocks"].get(block.id)
                    if block_current is None:
                        session.add(block)
                    else:
                        _copy_block(block_current, block)

            quiz = _quiz_for_chapter(conversation_id, current, now)
            desired_quiz_ids.add(quiz.id)
            quiz_current = existing["quizzes"].get(quiz.id)
            if quiz_current is None:
                session.add(quiz)
                quiz_current = quiz
            else:
                _copy_quiz(quiz_current, quiz)
            await session.flush()
            questions = _questions_for_chapter(conversation_id, current, quiz_current, concepts, chunks, now)
            for question in questions:
                desired_check_ids.add(question.id)
                question_current = existing["checks"].get(question.id)
                if question_current is None:
                    session.add(question)
                else:
                    _copy_check(question_current, question)
            quiz_current.question_ids = [str(question.id) for question in questions]
        _mark_stale_records(existing["chapters"], desired_chapter_ids, "state_metadata", now)
        _mark_stale_records(existing["lessons"], desired_lesson_ids, "lesson_metadata", now)
        _mark_stale_records(existing["blocks"], desired_block_ids, "block_metadata", now)
        _mark_stale_records(existing["quizzes"], desired_quiz_ids, "quiz_metadata", now)
        _mark_stale_records(existing["checks"], desired_check_ids, "check_metadata", now)
        await session.flush()
        return await self.get_course(session, conversation_id)

    async def _course_plan(
        self,
        phases: list[CourseLearningPhaseRecord],
        objectives: list[CourseLearningObjectiveRecord],
        concepts: list[CourseConceptRecord],
        chunks: list[SearchChunkRecord],
        *,
        llm_options: dict[str, Any] | None = None,
    ) -> list[_CourseChapterCandidate]:
        if not phases:
            return []
        try:
            chapters = await self._llm_course_plan(
                phases,
                objectives,
                concepts,
                chunks,
                llm_options=llm_options,
            )
        except Exception:  # noqa: BLE001
            logger.exception("LLM course-player planning failed; using learning-map fallback")
            chapters = []
        if chapters:
            return chapters
        return _fallback_course_plan(phases, objectives, concepts, chunks)

    async def _llm_course_plan(
        self,
        phases: list[CourseLearningPhaseRecord],
        objectives: list[CourseLearningObjectiveRecord],
        concepts: list[CourseConceptRecord],
        chunks: list[SearchChunkRecord],
        *,
        llm_options: dict[str, Any] | None,
    ) -> list[_CourseChapterCandidate]:
        if not chunks:
            return []
        user_prompt = _format_course_plan_context(phases, objectives, concepts, chunks)
        last_error: Exception | None = None
        for label, client in self._llm_clients(llm_options):
            try:
                response = await client.chat_structured(
                    messages=[
                        {"role": "system", "content": _COURSE_PLAN_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    schema=_CoursePlanCandidateBatch,
                    options={"temperature": 0.15, "num_predict": 5000, "max_tokens": 5000},
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "course-player planning with %s model %s failed",
                    label,
                    client.model,
                    exc_info=True,
                )
                continue
            chapters = _clean_course_plan(response.chapters, phases, objectives, concepts, chunks)
            if chapters:
                return chapters
        if last_error is not None:
            raise last_error
        return []

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

    async def get_course(self, session: AsyncSession, conversation_id: uuid.UUID) -> CoursePlayerRead:
        await self.ensure_schema(session)
        chapters, lessons, blocks, quizzes, attempts, checks = await _load_course_records(session, conversation_id)
        if not chapters:
            state = await get_learner_tracker().load_state(session, conversation_id)
            return CoursePlayerRead(conversation_id=conversation_id, chapters=[], learner_state=state)
        state = await get_learner_tracker().load_state(session, conversation_id)
        concepts = await get_concept_inventory_service().load_concepts(session, conversation_id)
        graph_hints: dict[uuid.UUID, dict[str, Any]] = {}
        graph_service = get_knowledge_graph_service()
        try:
            for lesson in lessons:
                graph_hints[lesson.id] = await graph_service.graph_hints_for_lesson(
                    session,
                    conversation_id,
                    [str(item) for item in lesson.concept_ids or []],
                )
        except Exception:  # noqa: BLE001
            graph_hints = {}
        return CoursePlayerRead(
            conversation_id=conversation_id,
            chapters=_assemble_chapters(chapters, lessons, blocks, quizzes, attempts, checks, concepts, state, graph_hints),
            learner_state=state,
        )

    async def unlock_chapter(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        chapter_id: uuid.UUID,
    ) -> CoursePlayerUnlockResponse:
        chapter = await session.get(CourseChapterRecord, chapter_id)
        if chapter is None or chapter.conversation_id != conversation_id:
            raise LookupError("chapter not found")
        chapter.state_metadata = {**dict(chapter.state_metadata or {}), "soft_lock_overridden": True}
        chapter.updated_at = datetime.now(timezone.utc)
        await session.flush()
        course = await self.get_course(session, conversation_id)
        current = next(item for item in course.chapters if item.id == chapter_id)
        return CoursePlayerUnlockResponse(chapter=current, learner_state=course.learner_state)

    async def submit_chapter_quiz(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        chapter_id: uuid.UUID,
        answers: dict[uuid.UUID, Any],
        *,
        llm_options: dict[str, Any] | None = None,
    ) -> ChapterQuizSubmitResponse:
        chapter = await session.get(CourseChapterRecord, chapter_id)
        if chapter is None or chapter.conversation_id != conversation_id:
            raise LookupError("chapter not found")
        quiz_result = await session.execute(
            select(ChapterQuizRecord).where(
                ChapterQuizRecord.conversation_id == conversation_id,
                ChapterQuizRecord.chapter_id == chapter_id,
            )
        )
        quiz = quiz_result.scalar_one_or_none()
        if quiz is None:
            raise LookupError("chapter quiz not found")

        assessment = get_knowledge_assessment_service()
        results: list[KnowledgeCheckResult] = []
        latest_state = await get_learner_tracker().load_state(session, conversation_id)
        for index, raw_check_id in enumerate(quiz.question_ids or []):
            check_id = uuid.UUID(str(raw_check_id))
            if check_id not in answers:
                continue
            result, latest_state = await assessment.submit_check(
                session,
                conversation_id,
                check_id,
                answers[check_id],
                llm_options=llm_options,
                question_index=index,
            )
            results.append(result)
        total = max(1, len(quiz.question_ids or []))
        score = sum(1 for result in results if result.is_correct) / total
        passed = score >= float(quiz.pass_score)
        attempt = ChapterAttemptRecord(
            conversation_id=conversation_id,
            chapter_id=chapter_id,
            quiz_id=quiz.id,
            score=score,
            passed=passed,
            answers=[{"check_id": str(check_id), "answer": answer} for check_id, answer in answers.items()],
            results=[result.model_dump(mode="json") for result in results],
            attempt_metadata={"pass_score": quiz.pass_score},
            created_at=datetime.now(timezone.utc),
        )
        session.add(attempt)
        await session.flush()
        course = await self.get_course(session, conversation_id)
        current = next(item for item in course.chapters if item.id == chapter_id)
        return ChapterQuizSubmitResponse(
            chapter=current,
            results=results,
            score=score,
            passed=passed,
            learner_state=latest_state,
        )


async def _load_chunks(session: AsyncSession, conversation_id: uuid.UUID) -> list[SearchChunkRecord]:
    result = await session.execute(
        select(SearchChunkRecord)
        .where(SearchChunkRecord.conversation_id == conversation_id)
        .order_by(SearchChunkRecord.document_id, SearchChunkRecord.chunk_index)
    )
    return list(result.scalars().all())


async def _load_existing(session: AsyncSession, conversation_id: uuid.UUID) -> dict[str, dict[uuid.UUID, Any]]:
    chapters, lessons, blocks, quizzes, _attempts, checks = await _load_course_records(
        session,
        conversation_id,
        include_inactive=True,
    )
    return {
        "chapters": {item.id: item for item in chapters},
        "lessons": {item.id: item for item in lessons},
        "blocks": {item.id: item for item in blocks},
        "quizzes": {item.id: item for item in quizzes},
        "checks": {item.id: item for item in checks},
    }


async def _load_course_records(
    session: AsyncSession,
    conversation_id: uuid.UUID,
    *,
    include_inactive: bool = False,
) -> tuple[
    list[CourseChapterRecord],
    list[CourseLessonRecord],
    list[CourseLessonBlockRecord],
    list[ChapterQuizRecord],
    list[ChapterAttemptRecord],
    list[KnowledgeCheckRecord],
]:
    chapters = list((await session.execute(select(CourseChapterRecord).where(CourseChapterRecord.conversation_id == conversation_id).order_by(CourseChapterRecord.order_index, CourseChapterRecord.title))).scalars().all())
    lessons = list((await session.execute(select(CourseLessonRecord).where(CourseLessonRecord.conversation_id == conversation_id).order_by(CourseLessonRecord.order_index, CourseLessonRecord.title))).scalars().all())
    blocks = list((await session.execute(select(CourseLessonBlockRecord).where(CourseLessonBlockRecord.conversation_id == conversation_id).order_by(CourseLessonBlockRecord.order_index))).scalars().all())
    quizzes = list((await session.execute(select(ChapterQuizRecord).where(ChapterQuizRecord.conversation_id == conversation_id))).scalars().all())
    attempts = list((await session.execute(select(ChapterAttemptRecord).where(ChapterAttemptRecord.conversation_id == conversation_id).order_by(ChapterAttemptRecord.created_at))).scalars().all())
    checks = list((await session.execute(select(KnowledgeCheckRecord).where(KnowledgeCheckRecord.conversation_id == conversation_id))).scalars().all())
    if not include_inactive:
        chapters = [item for item in chapters if _metadata_active(item, "state_metadata")]
        lessons = [item for item in lessons if _metadata_active(item, "lesson_metadata")]
        blocks = [item for item in blocks if _metadata_active(item, "block_metadata")]
        quizzes = [item for item in quizzes if _metadata_active(item, "quiz_metadata")]
        checks = [item for item in checks if _metadata_active(item, "check_metadata")]
    return chapters, lessons, blocks, quizzes, attempts, checks


def _chapter_from_candidate(
    conversation_id: uuid.UUID,
    candidate: _CourseChapterCandidate,
    phases: list[CourseLearningPhaseRecord],
    objectives: list[CourseLearningObjectiveRecord],
    concepts: list[CourseConceptRecord],
    chunks: list[SearchChunkRecord],
    now: datetime,
) -> CourseChapterRecord:
    phase = _resolve_phase(candidate.phase_id, candidate.title, phases)
    objective_ids = _resolve_objective_ids(candidate, objectives)
    concept_ids = _concept_ids_for_candidate(candidate, objective_ids, objectives, concepts)
    source_chunk_ids = _source_chunk_ids_for_candidate(candidate, objective_ids, concept_ids, phase, objectives, concepts, chunks)
    title = _clean_text(candidate.title)[:160] or (phase.title if phase else "Course Chapter")
    key = normalize_learning_key(title)
    return CourseChapterRecord(
        id=_stable_id("chapter", conversation_id, key),
        conversation_id=conversation_id,
        phase_id=phase.id if phase is not None else None,
        chapter_key=key,
        title=title,
        summary=_clean_text(candidate.summary)[:900] or _summary_from_sources(source_chunk_ids, chunks) or f"Study {title}.",
        order_index=max(0, int(candidate.order_index)),
        objective_ids=[str(item) for item in objective_ids],
        concept_ids=concept_ids,
        source_chunk_ids=source_chunk_ids,
        state_metadata={"generation": "llm_course_plan"},
        created_at=now,
        updated_at=now,
    )


def _lessons_for_chapter_candidate(
    candidate: _CourseChapterCandidate,
    phases: list[CourseLearningPhaseRecord],
    objectives: list[CourseLearningObjectiveRecord],
) -> list[_CourseLessonCandidate]:
    lessons = [_clean_lesson_candidate(item) for item in candidate.lessons[:_MAX_LESSONS_PER_CHAPTER]]
    lessons = [item for item in lessons if item is not None]
    if lessons:
        return lessons

    chapter_objective_ids = _resolve_objective_ids(candidate, objectives)
    if not chapter_objective_ids and candidate.phase_id:
        phase = _resolve_phase(candidate.phase_id, candidate.title, phases)
        if phase is not None:
            chapter_objective_ids = [item.id for item in objectives if item.phase_id == phase.id]
    out: list[_CourseLessonCandidate] = []
    objectives_by_id = {item.id: item for item in objectives}
    for objective in [objectives_by_id[item] for item in chapter_objective_ids if item in objectives_by_id]:
        out.append(
            _CourseLessonCandidate(
                title=_lesson_title(objective),
                summary=objective.objective_text,
                objective_id=str(objective.id),
                objective_text=objective.objective_text,
                source_chunk_ids=list(objective.source_chunk_ids or []),
            )
        )
    if out:
        return out[:_MAX_LESSONS_PER_CHAPTER]
    return [
        _CourseLessonCandidate(
            title=candidate.title,
            summary=candidate.summary,
            objective_text=f"Explain the main ideas in {candidate.title}",
            source_chunk_ids=list(candidate.source_chunk_ids or []),
        )
    ]


def _lesson_from_candidate(
    conversation_id: uuid.UUID,
    chapter_id: uuid.UUID,
    candidate: _CourseLessonCandidate,
    objectives: list[CourseLearningObjectiveRecord],
    concepts: list[CourseConceptRecord],
    chunks: list[SearchChunkRecord],
    now: datetime,
) -> CourseLessonRecord:
    objective = _resolve_objective(candidate.objective_id, candidate.objective_text, objectives)
    concept_ids = _concept_ids_for_lesson_candidate(candidate, objective, concepts)
    source_chunk_ids = _valid_chunk_ids(candidate.source_chunk_ids, chunks)
    if not source_chunk_ids and objective is not None:
        source_chunk_ids = list(objective.source_chunk_ids or [])
    if not source_chunk_ids:
        source_chunk_ids = _chunks_for_concepts(concept_ids, concepts)
    title = _clean_text(candidate.title)[:160] or (candidate.objective_text or "Lesson")
    key = normalize_learning_key(f"{chapter_id}:{objective.id if objective else title}")
    return CourseLessonRecord(
        id=_stable_id("lesson", conversation_id, key),
        conversation_id=conversation_id,
        chapter_id=chapter_id,
        objective_id=objective.id if objective is not None else None,
        lesson_key=key,
        title=title,
        summary=_clean_text(candidate.summary)[:900] or _summary_from_sources(source_chunk_ids, chunks) or candidate.objective_text,
        order_index=max(0, _objective_order(objective)),
        concept_ids=concept_ids,
        source_chunk_ids=source_chunk_ids,
        lesson_metadata={
            "generation": "llm_course_plan",
            "bloom_level": objective.bloom_level if objective is not None else "understand",
        },
        created_at=now,
        updated_at=now,
    )


def _blocks_from_candidate(
    conversation_id: uuid.UUID,
    lesson: CourseLessonRecord,
    candidate: _CourseLessonCandidate,
    concepts: list[CourseConceptRecord],
    chunks: list[SearchChunkRecord],
    now: datetime,
) -> list[CourseLessonBlockRecord]:
    records: list[CourseLessonBlockRecord] = []
    for index, block in enumerate(candidate.blocks[:_MAX_BLOCKS_PER_LESSON]):
        block_type = str(block.block_type or "explanation").strip().lower()
        if block_type not in _BLOCK_TYPES:
            block_type = "explanation"
        content = _clean_block_content(block.content)
        if not content:
            continue
        records.append(
            CourseLessonBlockRecord(
                id=_stable_id("block", conversation_id, f"{lesson.id}:{index}:{block_type}"),
                conversation_id=conversation_id,
                lesson_id=lesson.id,
                block_type=block_type,
                title=_clean_text(block.title)[:160],
                content=content[:2200],
                order_index=index,
                source_chunk_ids=_valid_chunk_ids(block.source_chunk_ids, chunks) or list(lesson.source_chunk_ids or [])[:3],
                block_metadata={"generation": "llm_course_plan"},
                created_at=now,
                updated_at=now,
            )
        )
    if records:
        return records
    return _fallback_block_records(
        conversation_id,
        lesson,
        lesson.summary or candidate.objective_text or lesson.title,
        concepts,
        chunks,
        now,
        generation="fallback_after_llm_plan",
    )


def _fallback_course_plan(
    phases: list[CourseLearningPhaseRecord],
    objectives: list[CourseLearningObjectiveRecord],
    concepts: list[CourseConceptRecord],
    chunks: list[SearchChunkRecord],
) -> list[_CourseChapterCandidate]:
    concepts_by_id = {str(concept.id): concept for concept in concepts}
    chapters: list[_CourseChapterCandidate] = []
    for phase in phases[:_MAX_COURSE_CHAPTERS]:
        phase_objectives = [item for item in objectives if item.phase_id == phase.id]
        lessons: list[_CourseLessonCandidate] = []
        for objective in phase_objectives[:_MAX_LESSONS_PER_CHAPTER]:
            objective_concepts = [
                concepts_by_id[concept_id].canonical_name
                for concept_id in objective.concept_ids or []
                if concept_id in concepts_by_id
            ]
            lessons.append(
                _CourseLessonCandidate(
                    title=_lesson_title(objective),
                    summary=_summary_from_sources(list(objective.source_chunk_ids or []), chunks) or objective.objective_text,
                    objective_id=str(objective.id),
                    objective_text=objective.objective_text,
                    concept_names=objective_concepts,
                    source_chunk_ids=list(objective.source_chunk_ids or []),
                )
            )
        if not lessons:
            lessons = [
                _CourseLessonCandidate(
                    title=phase.title,
                    summary=phase.summary,
                    objective_text=f"Explain the main ideas in {phase.title}",
                    source_chunk_ids=list(phase.source_chunk_ids or []),
                )
            ]
        phase_concepts = _dedupe(
            concept_name
            for lesson in lessons
            for concept_name in lesson.concept_names
        )
        chapters.append(
            _CourseChapterCandidate(
                title=phase.title,
                summary=phase.summary,
                order_index=phase.order_index,
                phase_id=str(phase.id),
                objective_ids=[str(item.id) for item in phase_objectives],
                concept_names=phase_concepts,
                source_chunk_ids=list(phase.source_chunk_ids or []),
                lessons=lessons,
            )
        )
    return chapters


def _clean_course_plan(
    chapters: list[_CourseChapterCandidate],
    phases: list[CourseLearningPhaseRecord],
    objectives: list[CourseLearningObjectiveRecord],
    concepts: list[CourseConceptRecord],
    chunks: list[SearchChunkRecord],
) -> list[_CourseChapterCandidate]:
    cleaned: list[_CourseChapterCandidate] = []
    seen: set[str] = set()
    for index, chapter in enumerate(chapters[:_MAX_COURSE_CHAPTERS]):
        title = _clean_text(chapter.title)
        if not _valid_course_title(title):
            continue
        key = normalize_learning_key(title)
        if key in seen:
            continue
        seen.add(key)
        lessons = [_clean_lesson_candidate(item) for item in chapter.lessons[:_MAX_LESSONS_PER_CHAPTER]]
        lessons = [item for item in lessons if item is not None]
        if not lessons:
            continue
        phase = _resolve_phase(chapter.phase_id, title, phases)
        objective_ids = [str(item) for item in _resolve_objective_ids(chapter, objectives)]
        source_chunk_ids = _valid_chunk_ids(chapter.source_chunk_ids, chunks)
        if not source_chunk_ids and phase is not None:
            source_chunk_ids = list(phase.source_chunk_ids or [])
        cleaned.append(
            chapter.model_copy(
                update={
                    "title": title,
                    "summary": _clean_text(chapter.summary),
                    "order_index": index,
                    "phase_id": str(phase.id) if phase is not None else None,
                    "objective_ids": objective_ids,
                    "concept_names": _dedupe(chapter.concept_names),
                    "source_chunk_ids": source_chunk_ids,
                    "lessons": lessons,
                }
            )
        )
    if cleaned:
        return cleaned
    return _fallback_course_plan(phases, objectives, concepts, chunks)


def _clean_lesson_candidate(candidate: _CourseLessonCandidate) -> _CourseLessonCandidate | None:
    title = _clean_text(candidate.title)
    if not _valid_course_title(title):
        return None
    blocks: list[_CourseBlockCandidate] = []
    for block in candidate.blocks[:_MAX_BLOCKS_PER_LESSON]:
        content = _clean_block_content(block.content)
        if not content:
            continue
        block_type = str(block.block_type or "explanation").strip().lower()
        if block_type not in _BLOCK_TYPES:
            block_type = "explanation"
        blocks.append(
            block.model_copy(
                update={
                    "block_type": block_type,
                    "title": _clean_text(block.title),
                    "content": content,
                }
            )
        )
    return candidate.model_copy(
        update={
            "title": title,
            "summary": _clean_text(candidate.summary),
            "objective_text": _clean_text(candidate.objective_text),
            "concept_names": _dedupe(candidate.concept_names),
            "blocks": blocks,
        }
    )


def _resolve_phase(
    phase_id: str | None,
    title: str,
    phases: list[CourseLearningPhaseRecord],
) -> CourseLearningPhaseRecord | None:
    if phase_id:
        for phase in phases:
            if str(phase.id) == str(phase_id):
                return phase
    title_key = normalize_learning_key(title)
    return next((phase for phase in phases if phase.phase_key == title_key), None)


def _resolve_objective_ids(
    candidate: _CourseChapterCandidate,
    objectives: list[CourseLearningObjectiveRecord],
) -> list[uuid.UUID]:
    objective_by_id = {str(objective.id): objective for objective in objectives}
    selected: list[uuid.UUID] = []
    for raw_id in candidate.objective_ids:
        if str(raw_id) in objective_by_id:
            selected.append(objective_by_id[str(raw_id)].id)
    for lesson in candidate.lessons:
        objective = _resolve_objective(lesson.objective_id, lesson.objective_text, objectives)
        if objective is not None and objective.id not in selected:
            selected.append(objective.id)
    return selected


def _resolve_objective(
    objective_id: str | None,
    objective_text: str,
    objectives: list[CourseLearningObjectiveRecord],
) -> CourseLearningObjectiveRecord | None:
    if objective_id:
        for objective in objectives:
            if str(objective.id) == str(objective_id):
                return objective
    key = normalize_learning_key(objective_text)
    return next((objective for objective in objectives if normalize_learning_key(objective.objective_text) == key), None)


def _objective_order(objective: CourseLearningObjectiveRecord | None) -> int:
    return int(objective.order_index) if objective is not None else 0


def _concept_ids_for_candidate(
    candidate: _CourseChapterCandidate,
    objective_ids: list[uuid.UUID],
    objectives: list[CourseLearningObjectiveRecord],
    concepts: list[CourseConceptRecord],
) -> list[str]:
    selected = [str(item) for objective in objectives if objective.id in objective_ids for item in objective.concept_ids or []]
    for name in candidate.concept_names:
        concept = resolve_concept(name, concepts)
        if concept is not None:
            selected.append(str(concept.id))
    for lesson in candidate.lessons:
        selected.extend(_concept_ids_for_lesson_candidate(lesson, None, concepts))
    return _dedupe(selected)


def _concept_ids_for_lesson_candidate(
    candidate: _CourseLessonCandidate,
    objective: CourseLearningObjectiveRecord | None,
    concepts: list[CourseConceptRecord],
) -> list[str]:
    selected = [str(item) for item in (objective.concept_ids if objective is not None else []) or []]
    for name in candidate.concept_names:
        concept = resolve_concept(name, concepts)
        if concept is not None:
            selected.append(str(concept.id))
    return _dedupe(selected)


def _source_chunk_ids_for_candidate(
    candidate: _CourseChapterCandidate,
    objective_ids: list[uuid.UUID],
    concept_ids: list[str],
    phase: CourseLearningPhaseRecord | None,
    objectives: list[CourseLearningObjectiveRecord],
    concepts: list[CourseConceptRecord],
    chunks: list[SearchChunkRecord],
) -> list[str]:
    selected = _valid_chunk_ids(candidate.source_chunk_ids, chunks)
    if phase is not None:
        selected.extend(list(phase.source_chunk_ids or []))
    selected.extend(
        chunk_id
        for objective in objectives
        if objective.id in objective_ids
        for chunk_id in objective.source_chunk_ids or []
    )
    selected.extend(_chunks_for_concepts(concept_ids, concepts))
    return _dedupe(selected)


def _valid_chunk_ids(raw_ids: list[str], chunks: list[SearchChunkRecord]) -> list[str]:
    available = {chunk.id for chunk in chunks}
    return _dedupe(str(chunk_id) for chunk_id in raw_ids if str(chunk_id) in available)


def _summary_from_sources(source_chunk_ids: list[str], chunks: list[SearchChunkRecord]) -> str:
    wanted = set(source_chunk_ids)
    parts: list[str] = []
    for chunk in chunks:
        if chunk.id not in wanted:
            continue
        text = " ".join(chunk.text.split())
        if text:
            parts.append(text[:260])
        if len(parts) >= 2:
            break
    return " ".join(parts)[:700]


def _chapter_from_phase(
    conversation_id: uuid.UUID,
    phase: CourseLearningPhaseRecord,
    objectives: list[CourseLearningObjectiveRecord],
    now: datetime,
) -> CourseChapterRecord:
    phase_objectives = [item for item in objectives if item.phase_id == phase.id]
    concept_ids = _dedupe(raw for objective in phase_objectives for raw in objective.concept_ids or [])
    source_chunk_ids = _dedupe([
        *list(phase.source_chunk_ids or []),
        *(raw for objective in phase_objectives for raw in objective.source_chunk_ids or []),
    ])
    key = normalize_learning_key(phase.title)
    return CourseChapterRecord(
        id=_stable_id("chapter", conversation_id, key),
        conversation_id=conversation_id,
        phase_id=phase.id,
        chapter_key=key,
        title=phase.title,
        summary=phase.summary or f"Study the main ideas in {phase.title}.",
        order_index=phase.order_index,
        objective_ids=[str(item.id) for item in phase_objectives],
        concept_ids=concept_ids,
        source_chunk_ids=source_chunk_ids,
        state_metadata={},
        created_at=now,
        updated_at=now,
    )


def _synthetic_objective(phase: CourseLearningPhaseRecord) -> CourseLearningObjectiveRecord:
    return CourseLearningObjectiveRecord(
        id=_stable_id("synthetic-objective", phase.conversation_id, str(phase.id)),
        conversation_id=phase.conversation_id,
        phase_id=phase.id,
        objective_key=phase.phase_key,
        objective_text=f"Explain the main ideas in {phase.title}",
        bloom_level="understand",
        order_index=0,
        concept_ids=[],
        source_file_ids=phase.source_file_ids,
        source_section_ids=phase.source_section_ids,
        source_chunk_ids=phase.source_chunk_ids,
        objective_metadata={"synthetic": True},
    )


def _lesson_from_objective(
    conversation_id: uuid.UUID,
    chapter_id: uuid.UUID,
    objective: CourseLearningObjectiveRecord,
    concepts: list[CourseConceptRecord],
    chunks: list[SearchChunkRecord],
    now: datetime,
) -> CourseLessonRecord:
    concept_ids = [str(item) for item in objective.concept_ids or []]
    source_chunk_ids = list(objective.source_chunk_ids or []) or _chunks_for_concepts(concept_ids, concepts)
    title = _lesson_title(objective)
    key = normalize_learning_key(f"{chapter_id}:{objective.id}:{title}")
    return CourseLessonRecord(
        id=_stable_id("lesson", conversation_id, key),
        conversation_id=conversation_id,
        chapter_id=chapter_id,
        objective_id=objective.id if not (objective.objective_metadata or {}).get("synthetic") else None,
        lesson_key=key,
        title=title,
        summary=_snippet(source_chunk_ids, chunks) or objective.objective_text,
        order_index=objective.order_index,
        concept_ids=concept_ids,
        source_chunk_ids=source_chunk_ids,
        lesson_metadata={"bloom_level": objective.bloom_level},
        created_at=now,
        updated_at=now,
    )


def _blocks_for_lesson(
    conversation_id: uuid.UUID,
    lesson: CourseLessonRecord,
    objective: CourseLearningObjectiveRecord,
    concepts: list[CourseConceptRecord],
    chunks: list[SearchChunkRecord],
    now: datetime,
) -> list[CourseLessonBlockRecord]:
    return _fallback_block_records(
        conversation_id,
        lesson,
        objective.objective_text,
        concepts,
        chunks,
        now,
        generation="fallback",
    )


def _fallback_block_records(
    conversation_id: uuid.UUID,
    lesson: CourseLessonRecord,
    objective_text: str,
    concepts: list[CourseConceptRecord],
    chunks: list[SearchChunkRecord],
    now: datetime,
    *,
    generation: str,
) -> list[CourseLessonBlockRecord]:
    lesson_concepts = [concept for concept in concepts if str(concept.id) in set(lesson.concept_ids or [])]
    concept_names = ", ".join(concept.canonical_name for concept in lesson_concepts[:4]) or lesson.title
    snippet = _snippet(lesson.source_chunk_ids, chunks)
    blocks = [
        ("definition", "Key definitions", _definitions(lesson_concepts) or f"{concept_names}: key terms from this part of the course."),
        ("explanation", "What you should understand", f"{objective_text}. {snippet}".strip()),
        ("example", "Example", _example_text(lesson.title, snippet)),
        ("summary", "Checkpoint", f"Before moving on, make sure you can explain {concept_names} in your own words."),
    ]
    return [
        CourseLessonBlockRecord(
            id=_stable_id("block", conversation_id, f"{lesson.id}:{index}:{block_type}"),
            conversation_id=conversation_id,
            lesson_id=lesson.id,
            block_type=block_type,
            title=title,
            content=content[:1800],
            order_index=index,
            source_chunk_ids=list(lesson.source_chunk_ids or [])[:3],
            block_metadata={"generation": generation},
            created_at=now,
            updated_at=now,
        )
        for index, (block_type, title, content) in enumerate(blocks)
    ]


def _quiz_for_chapter(conversation_id: uuid.UUID, chapter: CourseChapterRecord, now: datetime) -> ChapterQuizRecord:
    return ChapterQuizRecord(
        id=_stable_id("chapter-quiz", conversation_id, str(chapter.id)),
        conversation_id=conversation_id,
        chapter_id=chapter.id,
        question_ids=[],
        pass_score=PASS_SCORE,
        quiz_metadata={"source": "course_player"},
        created_at=now,
        updated_at=now,
    )


def _questions_for_chapter(
    conversation_id: uuid.UUID,
    chapter: CourseChapterRecord,
    quiz: ChapterQuizRecord,
    concepts: list[CourseConceptRecord],
    chunks: list[SearchChunkRecord],
    now: datetime,
) -> list[KnowledgeCheckRecord]:
    concept_by_id = {str(concept.id): concept for concept in concepts}
    selected = [concept_by_id[item] for item in chapter.concept_ids or [] if item in concept_by_id][:5]
    if not selected:
        selected = concepts[:3]
    questions: list[KnowledgeCheckRecord] = []
    for index, concept in enumerate(selected):
        qtype = "mcq" if index % 3 == 0 else "true_false" if index % 3 == 1 else "short_answer"
        answer_key: dict[str, Any] = {"concept_name": concept.canonical_name}
        options: list[str] = []
        prompt = f"Explain {concept.canonical_name} using the course material."
        if qtype == "mcq":
            distractors = [item.canonical_name for item in concepts if item.id != concept.id][:3]
            options = [concept.canonical_name, *distractors]
            answer_key["correct_index"] = 0
            prompt = "Which option is the best match for this lesson idea?"
        elif qtype == "true_false":
            answer_key["answer"] = True
            options = ["True", "False"]
            prompt = f"True or false: {concept.canonical_name} is an important idea in this chapter."
        else:
            answer_key["expected_terms"] = _expected_terms(concept)
        source_chunks = _concept_chunks(concept, chunks)
        questions.append(
            KnowledgeCheckRecord(
                id=_stable_id("chapter-question", conversation_id, f"{quiz.id}:{index}:{concept.id}"),
                conversation_id=conversation_id,
                concept_id=concept.id,
                question_type=qtype,
                bloom_level=concept.bloom_level if concept.bloom_level in {"remember", "understand", "apply", "analyze"} else "understand",
                prompt=prompt,
                options=options,
                answer_key=answer_key,
                rubric=concept.description or _snippet([chunk.id for chunk in source_chunks], chunks),
                source_chunk_ids=[chunk.id for chunk in source_chunks[:3]],
                check_metadata={"source": "chapter_quiz", "chapter_id": str(chapter.id), "quiz_id": str(quiz.id)},
                created_at=now,
            )
        )
    return questions


def _assemble_chapters(
    chapters: list[CourseChapterRecord],
    lessons: list[CourseLessonRecord],
    blocks: list[CourseLessonBlockRecord],
    quizzes: list[ChapterQuizRecord],
    attempts: list[ChapterAttemptRecord],
    checks: list[KnowledgeCheckRecord],
    concepts: list[CourseConceptRecord],
    learner_state: Any,
    graph_hints: dict[uuid.UUID, dict[str, Any]] | None = None,
) -> list[CourseChapterRead]:
    lessons_by_chapter: dict[uuid.UUID, list[CourseLessonRecord]] = {}
    for lesson in lessons:
        lessons_by_chapter.setdefault(lesson.chapter_id, []).append(lesson)
    blocks_by_lesson: dict[uuid.UUID, list[CourseLessonBlockRecord]] = {}
    for block in blocks:
        blocks_by_lesson.setdefault(block.lesson_id, []).append(block)
    quiz_by_chapter = {quiz.chapter_id: quiz for quiz in quizzes}
    attempts_by_chapter: dict[uuid.UUID, list[ChapterAttemptRecord]] = {}
    for attempt in attempts:
        attempts_by_chapter.setdefault(attempt.chapter_id, []).append(attempt)
    check_by_id = {str(check.id): check for check in checks}
    concept_by_id = {str(concept.id): concept for concept in concepts}
    concept_progress = {item.concept_id: item.mastery for item in learner_state.concept_progress}
    reads: list[CourseChapterRead] = []
    previous_completed = True
    for chapter in chapters:
        chapter_attempts = attempts_by_chapter.get(chapter.id, [])
        best_score = max((attempt.score for attempt in chapter_attempts), default=0.0)
        completed = best_score >= PASS_SCORE
        overridden = bool((chapter.state_metadata or {}).get("soft_lock_overridden"))
        state = "completed" if completed else "available" if previous_completed or overridden else "locked"
        concept_scores = [concept_progress.get(concept_id, 0.0) for concept_id in chapter.concept_ids or []]
        concept_mastery = sum(concept_scores) / len(concept_scores) if concept_scores else 0.0
        progress = max(concept_mastery, best_score if completed else best_score * 0.7)
        quiz = quiz_by_chapter.get(chapter.id)
        quiz_read = None
        if quiz is not None:
            quiz_checks = [check_by_id[item] for item in quiz.question_ids or [] if item in check_by_id]
            quiz_read = ChapterQuizRead(
                id=quiz.id,
                chapter_id=chapter.id,
                pass_score=quiz.pass_score,
                question_ids=[check.id for check in quiz_checks],
                questions=[get_knowledge_assessment_service()._to_question(check, list(concept_by_id.values())) for check in quiz_checks],
            )
        reads.append(
            CourseChapterRead(
                id=chapter.id,
                phase_id=chapter.phase_id,
                title=chapter.title,
                summary=chapter.summary,
                order_index=chapter.order_index,
                objective_ids=[uuid.UUID(str(item)) for item in chapter.objective_ids or []],
                concept_ids=[uuid.UUID(str(item)) for item in chapter.concept_ids or []],
                source_chunk_ids=list(chapter.source_chunk_ids or []),
                state=state,  # type: ignore[arg-type]
                best_score=best_score,
                attempts=len(chapter_attempts),
                soft_lock_overridden=overridden,
                progress=progress,
                lessons=[
                    CourseLessonRead(
                        id=lesson.id,
                        chapter_id=lesson.chapter_id,
                        objective_id=lesson.objective_id,
                        title=lesson.title,
                        summary=lesson.summary,
                        order_index=lesson.order_index,
                        concept_ids=[uuid.UUID(str(item)) for item in lesson.concept_ids or []],
                        source_chunk_ids=list(lesson.source_chunk_ids or []),
                        prerequisite_concept_ids=[
                            uuid.UUID(str(item))
                            for item in (graph_hints or {}).get(lesson.id, {}).get("prerequisite_concept_ids", [])
                        ],
                        next_concept_ids=[
                            uuid.UUID(str(item))
                            for item in (graph_hints or {}).get(lesson.id, {}).get("next_concept_ids", [])
                        ],
                        related_example_ids=[
                            uuid.UUID(str(item))
                            for item in (graph_hints or {}).get(lesson.id, {}).get("related_example_ids", [])
                        ],
                        remediation_objective_ids=[],
                        graph_hints=dict((graph_hints or {}).get(lesson.id, {})),
                        blocks=[
                            CourseLessonBlockRead(
                                id=block.id,
                                lesson_id=block.lesson_id,
                                block_type=block.block_type,
                                title=block.title,
                                content=block.content,
                                order_index=block.order_index,
                                source_chunk_ids=list(block.source_chunk_ids or []),
                                metadata=dict(block.block_metadata or {}),
                            )
                            for block in sorted(blocks_by_lesson.get(lesson.id, []), key=lambda item: item.order_index)
                        ],
                    )
                    for lesson in sorted(
                        lessons_by_chapter.get(chapter.id, []),
                        key=lambda item: (
                            len((graph_hints or {}).get(item.id, {}).get("prerequisite_concept_ids", [])),
                            item.order_index,
                        ),
                    )
                ],
                quiz=quiz_read,
            )
        )
        previous_completed = completed
    return reads


def _copy_chapter(target: CourseChapterRecord, source: CourseChapterRecord) -> None:
    preserved = _without_inactive_metadata(target.state_metadata)
    target.phase_id = source.phase_id
    target.chapter_key = source.chapter_key
    target.title = source.title
    target.summary = source.summary
    target.order_index = source.order_index
    target.objective_ids = source.objective_ids
    target.concept_ids = source.concept_ids
    target.source_chunk_ids = source.source_chunk_ids
    target.state_metadata = {**source.state_metadata, **preserved}
    target.updated_at = source.updated_at


def _copy_lesson(target: CourseLessonRecord, source: CourseLessonRecord) -> None:
    target.chapter_id = source.chapter_id
    target.objective_id = source.objective_id
    target.lesson_key = source.lesson_key
    target.title = source.title
    target.summary = source.summary
    target.order_index = source.order_index
    target.concept_ids = source.concept_ids
    target.source_chunk_ids = source.source_chunk_ids
    target.lesson_metadata = source.lesson_metadata
    target.updated_at = source.updated_at


def _copy_block(target: CourseLessonBlockRecord, source: CourseLessonBlockRecord) -> None:
    target.lesson_id = source.lesson_id
    target.block_type = source.block_type
    target.title = source.title
    target.content = source.content
    target.order_index = source.order_index
    target.source_chunk_ids = source.source_chunk_ids
    target.block_metadata = source.block_metadata
    target.updated_at = source.updated_at


def _copy_quiz(target: ChapterQuizRecord, source: ChapterQuizRecord) -> None:
    target.chapter_id = source.chapter_id
    target.pass_score = source.pass_score
    target.quiz_metadata = source.quiz_metadata
    target.updated_at = source.updated_at


def _copy_check(target: KnowledgeCheckRecord, source: KnowledgeCheckRecord) -> None:
    target.concept_id = source.concept_id
    target.question_type = source.question_type
    target.bloom_level = source.bloom_level
    target.prompt = source.prompt
    target.options = source.options
    target.answer_key = source.answer_key
    target.rubric = source.rubric
    target.source_chunk_ids = source.source_chunk_ids
    target.check_metadata = source.check_metadata


def _metadata_active(record: Any, field_name: str) -> bool:
    return not (getattr(record, field_name, None) or {}).get("inactive")


def _without_inactive_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    cleaned = dict(metadata or {})
    cleaned.pop("inactive", None)
    cleaned.pop("inactive_reason", None)
    cleaned.pop("inactive_at", None)
    return cleaned


def _mark_stale_records(
    records: dict[uuid.UUID, Any],
    desired_ids: set[uuid.UUID],
    field_name: str,
    now: datetime,
) -> None:
    for record_id, record in records.items():
        if record_id in desired_ids:
            setattr(record, field_name, _without_inactive_metadata(getattr(record, field_name, None)))
            if hasattr(record, "updated_at"):
                record.updated_at = now
            continue
        metadata = dict(getattr(record, field_name, None) or {})
        if metadata.get("inactive"):
            continue
        metadata["inactive"] = True
        metadata["inactive_reason"] = "course_rebuild_stale"
        metadata["inactive_at"] = now.isoformat()
        setattr(record, field_name, metadata)
        if hasattr(record, "updated_at"):
            record.updated_at = now


def _format_course_plan_context(
    phases: list[CourseLearningPhaseRecord],
    objectives: list[CourseLearningObjectiveRecord],
    concepts: list[CourseConceptRecord],
    chunks: list[SearchChunkRecord],
) -> str:
    objectives_by_phase: dict[uuid.UUID, list[CourseLearningObjectiveRecord]] = {}
    for objective in objectives:
        objectives_by_phase.setdefault(objective.phase_id, []).append(objective)

    phase_lines: list[str] = []
    for phase in phases[:_MAX_COURSE_CHAPTERS]:
        phase_lines.append(
            f"- phase_id={phase.id} | order={phase.order_index} | title={phase.title} | summary={phase.summary[:240]}"
        )
        for objective in objectives_by_phase.get(phase.id, [])[:_MAX_LESSONS_PER_CHAPTER]:
            phase_lines.append(
                f"  - objective_id={objective.id} | bloom={objective.bloom_level} | objective={objective.objective_text}"
            )

    concept_lines = [
        f"- {concept.canonical_name}: {concept.description[:180]}"
        for concept in concepts[:100]
        if not (concept.concept_metadata or {}).get("inactive")
    ]

    chunk_lines: list[str] = []
    for chunk in chunks[:80]:
        text = " ".join(chunk.text.split())
        if len(text) > 900:
            text = text[:900].rsplit(" ", 1)[0].strip()
        chunk_lines.append(
            "\n".join(
                [
                    f"chunk_id: {chunk.id}",
                    f"source: {chunk.source_filename}",
                    f"path: {' > '.join(chunk.heading_path or [])}",
                    f"text: {text}",
                ]
            )
        )

    return (
        "Learning map candidates:\n"
        f"{chr(10).join(phase_lines) or '(none)'}\n\n"
        "Canonical concepts:\n"
        f"{chr(10).join(concept_lines) or '(none)'}\n\n"
        "Course source chunks:\n"
        f"{chr(10).join(['---', *chunk_lines]) if chunk_lines else '(none)'}"
    )


def _clean_text(value: str) -> str:
    text = re.sub(r"</?\w+[^>]*>", " ", str(value or ""))
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"\s+", " ", text).strip(" -*#:;,.")
    return text


def _clean_block_content(value: str) -> str:
    text = str(value or "").replace("\r\n", "\n").strip()
    text = re.sub(r"</?(?:td|th|tr|table)[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _valid_course_title(value: str) -> bool:
    text = _clean_text(value)
    key = normalize_learning_key(text)
    if not key or not 3 <= len(text) <= 180:
        return False
    if re.search(r"</?\w+|^\d+(?:\.\d+)*$|[$\\{}]", text):
        return False
    if key in {"introduction", "summary", "resume", "conclusion", "agenda", "plan"}:
        return False
    return True


def _stable_id(kind: str, conversation_id: uuid.UUID | str, key: str) -> uuid.UUID:
    seed = f"{kind}:{conversation_id}:{key}"
    return uuid.uuid5(uuid.NAMESPACE_URL, f"teacherlm:{seed}:{sha1(seed.encode()).hexdigest()[:12]}")


def _lesson_title(objective: CourseLearningObjectiveRecord) -> str:
    text = objective.objective_text.strip()
    text = re.sub(r"^(explain|understand|describe|apply)\s+", "", text, flags=re.IGNORECASE).strip()
    return text[:90] or "Lesson"


def _definitions(concepts: list[CourseConceptRecord]) -> str:
    lines = []
    for concept in concepts[:4]:
        description = concept.description or "A key idea in this part of the course."
        lines.append(f"- {concept.canonical_name}: {description}")
    return "\n".join(lines)


def _example_text(title: str, snippet: str) -> str:
    if snippet:
        return f"Use this part of the course as your worked example: {snippet}"
    return f"Try explaining {title} with a small example from the uploaded material."


def _snippet(source_chunk_ids: list[str], chunks: list[SearchChunkRecord]) -> str:
    wanted = set(source_chunk_ids)
    for chunk in chunks:
        if chunk.id in wanted:
            return " ".join(chunk.text.split())[:420]
    return ""


def _chunks_for_concepts(concept_ids: list[str], concepts: list[CourseConceptRecord]) -> list[str]:
    wanted = set(concept_ids)
    return _dedupe(chunk_id for concept in concepts if str(concept.id) in wanted for chunk_id in concept.source_chunk_ids or [])


def _concept_chunks(concept: CourseConceptRecord, chunks: list[SearchChunkRecord]) -> list[SearchChunkRecord]:
    wanted = set(concept.source_chunk_ids or [])
    matched = [chunk for chunk in chunks if chunk.id in wanted]
    return matched or chunks[:3]


def _expected_terms(concept: CourseConceptRecord) -> list[str]:
    terms = [concept.canonical_name, *list(concept.aliases or [])]
    terms.extend(_WORD_RE.findall(concept.description or "")[:8])
    return _dedupe(terms)[:12]


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


_COURSE_PLAN_SYSTEM_PROMPT = """You build the complete guided course shown to a student in TeacherLM.

The uploaded files may be from any domain. Build a real study path, not a list
of raw headings. The path must start with prerequisites/fundamentals, then core
ideas, then procedures/applications, and end with advanced/evaluation topics
when the source supports that order.

Return only JSON matching the schema.

Rules:
- Use the provided phase_id and objective_id values when a chapter or lesson
  corresponds to them.
- Use exact chunk_id values in source_chunk_ids.
- Chapter titles should be student-facing modules, not file names, table cells,
  slide fragments, formulas, variables, or one-word labels.
- Each lesson needs concise teachable blocks: definition/explanation/example or
  procedure/formula/summary depending on the source.
- Blocks must be grounded in the source chunks. Do not invent facts.
- Lessons should explain what the student reads and practices before taking the
  chapter quiz.
- Keep the course compact enough for a right sidebar: usually 4-10 chapters,
  1-5 lessons per chapter, and short content blocks."""


_service: CoursePlayerService | None = None


def get_course_player_service() -> CoursePlayerService:
    global _service
    if _service is None:
        _service = CoursePlayerService()
    return _service
