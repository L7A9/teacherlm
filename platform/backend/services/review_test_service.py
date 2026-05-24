from __future__ import annotations

import uuid
import logging
from datetime import datetime, timezone
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from db.models import (
    AnsweredCourseQuestionRecord,
    CourseConceptRecord,
    CourseLearningObjectiveRecord,
    CourseLearningPhaseRecord,
    KnowledgeCheckRecord,
    LearningReviewWindowRecord,
)
from schemas.knowledge_check import KnowledgeCheckResult
from schemas.review_test import (
    ReviewTestActionResponse,
    ReviewTestStartResponse,
    ReviewTestStatusResponse,
    ReviewTestSubmitResponse,
    ReviewWindowSummary,
)
from services.concept_inventory_service import get_concept_inventory_service, resolve_concept
from services.knowledge_assessment_service import get_knowledge_assessment_service
from services.knowledge_graph_service import get_knowledge_graph_service
from services.learning_map_service import get_learning_map_service
from services.learner_tracker import get_learner_tracker
from teacherlm_core.schemas.generator_io import LearnerUpdates


logger = logging.getLogger(__name__)

WINDOW_SIZE = 10
SNOOZE_ANSWER_COUNT = 2
REVIEW_STATUSES_ACTIVE = {"pending", "started", "snoozed"}


class ReviewTestService:
    async def ensure_schema(self, session: AsyncSession) -> None:
        connection = await session.connection()

        def create_tables(sync_connection) -> None:  # noqa: ANN001
            LearningReviewWindowRecord.__table__.create(sync_connection, checkfirst=True)
            AnsweredCourseQuestionRecord.__table__.create(sync_connection, checkfirst=True)

        await connection.run_sync(create_tables)

    async def record_answered_course_question(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        user_message_id: uuid.UUID,
        assistant_message_id: uuid.UUID,
        source_chunk_ids: list[str],
        learner_updates: LearnerUpdates,
    ) -> None:
        await self.ensure_schema(session)
        existing = await session.execute(
            select(AnsweredCourseQuestionRecord).where(
                AnsweredCourseQuestionRecord.assistant_message_id == assistant_message_id
            )
        )
        if existing.scalar_one_or_none() is not None:
            return

        concepts = await get_concept_inventory_service().load_concepts(session, conversation_id)
        phases, objectives = await get_learning_map_service().load_map(session, conversation_id)
        concept_ids = _resolve_update_concepts(learner_updates, concepts)
        concept_ids = concept_ids or _concept_ids_from_sources(source_chunk_ids, concepts)
        objective_ids, phase_ids = _map_concepts_to_learning_map(concept_ids, phases, objectives)
        now = datetime.now(timezone.utc)
        record = AnsweredCourseQuestionRecord(
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            assistant_message_id=assistant_message_id,
            concept_ids=concept_ids,
            objective_ids=objective_ids,
            phase_ids=phase_ids,
            source_chunk_ids=_dedupe(source_chunk_ids),
            question_metadata={"trigger": "retrieval_answer"},
            created_at=now,
        )
        session.add(record)
        await session.flush()
        await self._create_due_windows(session, conversation_id)

    async def status(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        include_learner_state: bool = False,
    ) -> ReviewTestStatusResponse:
        await self.ensure_schema(session)
        total = await self._answered_count(session, conversation_id)
        window = await self._current_window(session, conversation_id)
        due = bool(window and _window_due(window, total))
        pending = await self._pending_unwindowed_count(session, conversation_id)
        learner_state = (
            await get_learner_tracker().load_state(session, conversation_id)
            if include_learner_state
            else None
        )
        return ReviewTestStatusResponse(
            answered_count=total,
            pending_count=pending,
            due=due,
            window=_window_summary(window) if window else None,
            learner_state=learner_state,
        )

    async def start_review(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        llm_options: dict[str, Any] | None = None,
    ) -> ReviewTestStartResponse:
        await self.ensure_schema(session)
        await self._create_due_windows(session, conversation_id)
        total = await self._answered_count(session, conversation_id)
        window = await self._current_window(session, conversation_id)
        if window is None:
            window = await self._create_manual_window(session, conversation_id, total)
        if window is None:
            raise LookupError("No answered course questions are available for a review yet.")

        concepts = await get_concept_inventory_service().load_concepts(session, conversation_id)
        phases, objectives = await get_learning_map_service().load_map(session, conversation_id)
        checks = await self._ensure_checks(
            session,
            conversation_id,
            window,
            concepts,
            phases,
            objectives,
            llm_options=llm_options,
        )
        window.status = "started"
        window.updated_at = datetime.now(timezone.utc)
        await session.flush()
        learner_state = await get_learner_tracker().load_state(session, conversation_id)
        return ReviewTestStartResponse(
            window=_window_summary(window),
            checks=[get_knowledge_assessment_service()._to_question(check, concepts) for check in checks],
            learner_state=learner_state,
        )

    async def submit_review(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        window_id: uuid.UUID,
        answers: dict[uuid.UUID, Any],
        *,
        llm_options: dict[str, Any] | None = None,
    ) -> ReviewTestSubmitResponse:
        await self.ensure_schema(session)
        window = await self._get_window(session, conversation_id, window_id)
        allowed = {str(item) for item in window.generated_check_ids or []}
        if not allowed:
            raise LookupError("Review test has not been started.")

        results: list[KnowledgeCheckResult] = []
        latest_state = await get_learner_tracker().load_state(session, conversation_id)
        assessment = get_knowledge_assessment_service()
        for index, raw_check_id in enumerate(window.generated_check_ids or []):
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

        window.status = "completed"
        window.updated_at = datetime.now(timezone.utc)
        await session.flush()
        return ReviewTestSubmitResponse(
            window=_window_summary(window),
            results=results,
            learner_state=latest_state,
        )

    async def snooze(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        window_id: uuid.UUID,
    ) -> ReviewTestActionResponse:
        window = await self._get_window(session, conversation_id, window_id)
        total = await self._answered_count(session, conversation_id)
        window.status = "snoozed"
        window.snooze_until_count = total + SNOOZE_ANSWER_COUNT
        window.updated_at = datetime.now(timezone.utc)
        await session.flush()
        return ReviewTestActionResponse(
            window=_window_summary(window),
            answered_count=total,
            due=False,
        )

    async def dismiss(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        window_id: uuid.UUID,
    ) -> ReviewTestActionResponse:
        window = await self._get_window(session, conversation_id, window_id)
        total = await self._answered_count(session, conversation_id)
        window.status = "dismissed"
        window.updated_at = datetime.now(timezone.utc)
        await session.flush()
        return ReviewTestActionResponse(
            window=_window_summary(window),
            answered_count=total,
            due=False,
        )

    async def _ensure_checks(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        window: LearningReviewWindowRecord,
        concepts: list[CourseConceptRecord],
        phases: list[CourseLearningPhaseRecord],
        objectives: list[CourseLearningObjectiveRecord],
        *,
        llm_options: dict[str, Any] | None,
    ) -> list[KnowledgeCheckRecord]:
        assessment = get_knowledge_assessment_service()
        await assessment.ensure_schema(session)
        existing_ids = [uuid.UUID(str(item)) for item in window.generated_check_ids or []]
        if existing_ids:
            result = await session.execute(
                select(KnowledgeCheckRecord)
                .where(KnowledgeCheckRecord.id.in_(existing_ids))
                .order_by(KnowledgeCheckRecord.created_at)
            )
            checks = list(result.scalars().all())
            if checks:
                return checks

        ranked = await self._rank_window_concepts(session, conversation_id, window, concepts, objectives)
        count = _review_question_count(window, ranked)
        chunks = await assessment._load_chunks(session, conversation_id)
        checks: list[KnowledgeCheckRecord] = []
        for index, concept in enumerate(ranked[:count]):
            qtype = assessment._choose_question_type(index, concept, None)
            phase, objective = _learning_context_for_concept_id(concept.id, phases, objectives)
            check = await assessment._build_check(
                conversation_id,
                concept,
                chunks,
                qtype,
                concepts,
                phase=phase,
                objective=objective,
                llm_options=llm_options,
            )
            check.check_metadata = {
                **dict(check.check_metadata or {}),
                "source": "review_test",
                "review_window_id": str(window.id),
            }
            session.add(check)
            checks.append(check)
        await session.flush()
        window.generated_check_ids = [str(check.id) for check in checks]
        window.updated_at = datetime.now(timezone.utc)
        return checks

    async def _rank_window_concepts(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        window: LearningReviewWindowRecord,
        concepts: list[CourseConceptRecord],
        objectives: list[CourseLearningObjectiveRecord],
    ) -> list[CourseConceptRecord]:
        concept_by_id = {str(concept.id): concept for concept in concepts}
        state = await get_learner_tracker().load_state(session, conversation_id)
        progress = {item.concept_id: item for item in state.concept_progress}
        window_ids = [str(item) for item in window.concept_ids or []]
        if not window_ids:
            for objective_id in window.objective_ids or []:
                objective = next((item for item in objectives if str(item.id) == str(objective_id)), None)
                if objective:
                    window_ids.extend(str(item) for item in objective.concept_ids or [])
        ranked = [concept_by_id[item] for item in _dedupe(window_ids) if item in concept_by_id]
        try:
            prereqs = await get_knowledge_graph_service().concept_prerequisites(
                session,
                conversation_id,
                [str(concept.id) for concept in ranked],
            )
            for prereq_id in _dedupe(raw for ids in prereqs.values() for raw in ids):
                concept = concept_by_id.get(prereq_id)
                if concept is not None and concept not in ranked:
                    ranked.append(concept)
        except Exception:  # noqa: BLE001
            logger.exception("review-test graph prerequisite expansion failed; continuing without graph")
        return sorted(
            ranked,
            key=lambda concept: (
                progress.get(str(concept.id)).mastery if progress.get(str(concept.id)) else 0.0,
                progress.get(str(concept.id)).encounters if progress.get(str(concept.id)) else 0,
                -float(concept.importance),
                concept.canonical_name.casefold(),
            ),
        )

    async def _create_due_windows(self, session: AsyncSession, conversation_id: uuid.UUID) -> None:
        while True:
            result = await session.execute(
                select(AnsweredCourseQuestionRecord)
                .where(
                    AnsweredCourseQuestionRecord.conversation_id == conversation_id,
                    AnsweredCourseQuestionRecord.review_window_id.is_(None),
                )
                .order_by(AnsweredCourseQuestionRecord.created_at, AnsweredCourseQuestionRecord.id)
                .limit(WINDOW_SIZE)
            )
            questions = list(result.scalars().all())
            if len(questions) < WINDOW_SIZE:
                return
            total = await self._answered_count(session, conversation_id)
            window = _window_from_questions(conversation_id, questions, total)
            session.add(window)
            await session.flush()
            for question in questions:
                question.review_window_id = window.id
            await session.flush()

    async def _create_manual_window(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        total_answered: int,
    ) -> LearningReviewWindowRecord | None:
        result = await session.execute(
            select(AnsweredCourseQuestionRecord)
            .where(AnsweredCourseQuestionRecord.conversation_id == conversation_id)
            .order_by(AnsweredCourseQuestionRecord.created_at.desc(), AnsweredCourseQuestionRecord.id.desc())
            .limit(WINDOW_SIZE)
        )
        questions = list(result.scalars().all())[::-1]
        if not questions:
            return None
        window = _window_from_questions(conversation_id, questions, total_answered, status="started")
        window.review_metadata = {**dict(window.review_metadata or {}), "manual": True}
        session.add(window)
        await session.flush()
        return window

    async def _answered_count(self, session: AsyncSession, conversation_id: uuid.UUID) -> int:
        result = await session.execute(
            select(func.count())
            .select_from(AnsweredCourseQuestionRecord)
            .where(AnsweredCourseQuestionRecord.conversation_id == conversation_id)
        )
        return int(result.scalar_one() or 0)

    async def _pending_unwindowed_count(self, session: AsyncSession, conversation_id: uuid.UUID) -> int:
        result = await session.execute(
            select(func.count())
            .select_from(AnsweredCourseQuestionRecord)
            .where(
                AnsweredCourseQuestionRecord.conversation_id == conversation_id,
                AnsweredCourseQuestionRecord.review_window_id.is_(None),
            )
        )
        return int(result.scalar_one() or 0)

    async def _current_window(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> LearningReviewWindowRecord | None:
        result = await session.execute(
            select(LearningReviewWindowRecord)
            .where(
                LearningReviewWindowRecord.conversation_id == conversation_id,
                LearningReviewWindowRecord.status.in_(REVIEW_STATUSES_ACTIVE),
            )
            .order_by(LearningReviewWindowRecord.created_at, LearningReviewWindowRecord.id)
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def _get_window(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        window_id: uuid.UUID,
    ) -> LearningReviewWindowRecord:
        window = await session.get(LearningReviewWindowRecord, window_id)
        if window is None or window.conversation_id != conversation_id:
            raise LookupError("review window not found")
        return window


def _window_from_questions(
    conversation_id: uuid.UUID,
    questions: list[AnsweredCourseQuestionRecord],
    total_answered: int,
    *,
    status: str = "pending",
) -> LearningReviewWindowRecord:
    now = datetime.now(timezone.utc)
    return LearningReviewWindowRecord(
        conversation_id=conversation_id,
        status=status,
        answered_question_ids=[str(item.id) for item in questions],
        user_message_ids=[str(item.user_message_id) for item in questions],
        assistant_message_ids=[str(item.assistant_message_id) for item in questions],
        concept_ids=_dedupe([raw for item in questions for raw in item.concept_ids or []]),
        objective_ids=_dedupe([raw for item in questions for raw in item.objective_ids or []]),
        phase_ids=_dedupe([raw for item in questions for raw in item.phase_ids or []]),
        source_chunk_ids=_dedupe([raw for item in questions for raw in item.source_chunk_ids or []]),
        answer_count=len(questions),
        due_count=total_answered,
        review_metadata={"window_size": WINDOW_SIZE},
        created_at=now,
        updated_at=now,
    )


def _window_due(window: LearningReviewWindowRecord, total_answered: int) -> bool:
    if window.status in {"pending", "started"}:
        return True
    if window.status == "snoozed":
        return total_answered >= int(window.snooze_until_count or window.due_count)
    return False


def _window_summary(window: LearningReviewWindowRecord) -> ReviewWindowSummary:
    return ReviewWindowSummary(
        id=window.id,
        status=window.status,
        answered_count=window.answer_count,
        due_count=window.due_count,
        snooze_until_count=window.snooze_until_count,
        concept_ids=[uuid.UUID(str(item)) for item in window.concept_ids or []],
        objective_ids=[uuid.UUID(str(item)) for item in window.objective_ids or []],
        phase_ids=[uuid.UUID(str(item)) for item in window.phase_ids or []],
        source_chunk_ids=list(window.source_chunk_ids or []),
        generated_check_ids=[uuid.UUID(str(item)) for item in window.generated_check_ids or []],
    )


def _review_question_count(
    window: LearningReviewWindowRecord,
    concepts: list[CourseConceptRecord],
) -> int:
    richness = max(len(concepts), len(window.objective_ids or []), len(window.source_chunk_ids or []) // 2)
    if not concepts:
        return 0
    if richness <= 3:
        return min(len(concepts), 3)
    if richness >= 7:
        return min(len(concepts), 7)
    return min(len(concepts), 5)


def _resolve_update_concepts(
    updates: LearnerUpdates,
    concepts: list[CourseConceptRecord],
) -> list[str]:
    ids: list[str] = []
    labels = [
        *updates.concepts_covered,
        *updates.concepts_demonstrated,
        *updates.concepts_struggled,
    ]
    for label in labels:
        resolved = resolve_concept(label, concepts)
        if resolved is not None:
            ids.append(str(resolved.id))
    return _dedupe(ids)


def _concept_ids_from_sources(
    source_chunk_ids: list[str],
    concepts: list[CourseConceptRecord],
) -> list[str]:
    source_set = set(source_chunk_ids)
    return _dedupe(
        str(concept.id)
        for concept in concepts
        if source_set & {str(item) for item in concept.source_chunk_ids or []}
    )


def _map_concepts_to_learning_map(
    concept_ids: list[str],
    phases: list[CourseLearningPhaseRecord],
    objectives: list[CourseLearningObjectiveRecord],
) -> tuple[list[str], list[str]]:
    concept_set = set(concept_ids)
    objective_ids: list[str] = []
    phase_ids: list[str] = []
    phase_by_id = {str(phase.id): phase for phase in phases}
    for objective in objectives:
        if concept_set & {str(item) for item in objective.concept_ids or []}:
            objective_ids.append(str(objective.id))
            if str(objective.phase_id) in phase_by_id:
                phase_ids.append(str(objective.phase_id))
    return _dedupe(objective_ids), _dedupe(phase_ids)


def _learning_context_for_concept_id(
    concept_id: uuid.UUID,
    phases: list[CourseLearningPhaseRecord],
    objectives: list[CourseLearningObjectiveRecord],
) -> tuple[CourseLearningPhaseRecord | None, CourseLearningObjectiveRecord | None]:
    concept_key = str(concept_id)
    objective = next(
        (item for item in objectives if concept_key in [str(raw_id) for raw_id in item.concept_ids or []]),
        None,
    )
    phase = next((item for item in phases if objective and item.id == objective.phase_id), None)
    return phase, objective


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


_service: ReviewTestService | None = None


def get_review_test_service() -> ReviewTestService:
    global _service
    if _service is None:
        _service = ReviewTestService()
    return _service
