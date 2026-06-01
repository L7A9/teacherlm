from __future__ import annotations

import logging
import re
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from teacherlm_core.llm.ollama_client import OllamaClient
from teacherlm_core.llm.runtime import build_llm_client_kwargs, has_llm_override

from config import Settings, get_settings
from db.models import (
    CourseConceptRecord,
    CourseBuilderChapterAttemptRecord,
    CourseBuilderChapterRecord,
    CourseBuilderCourseRecord,
    CourseBuilderLessonBlockRecord,
    CourseBuilderLessonRecord,
    CourseBuilderProgressEventRecord,
    CourseBuilderQuizQuestionRecord,
    CourseBuilderQuizRecord,
    CourseDocumentRecord,
    CourseLearningObjectiveRecord,
    CourseLearningPhaseRecord,
    CourseSectionRecord,
    SearchChunkRecord,
    UploadedFile,
)
from schemas.coursebuilder import (
    CourseBuilderChapterRead,
    CourseBuilderCitation,
    CourseBuilderLessonBlockRead,
    CourseBuilderLessonRead,
    CourseBuilderProgressEventRead,
    CourseBuilderQuizQuestionRead,
    CourseBuilderQuizRead,
    CourseBuilderQuizResult,
    CourseBuilderQuizSubmitResponse,
    CourseBuilderRead,
)
from services.coursebuilder_rag import get_coursebuilder_rag_service
from services.coursebuilder_validation import (
    insufficient_source_message,
    normalize_block_type,
    support_status,
    validate_chart_spec,
)
from services.concept_inventory_service import get_concept_inventory_service
from services.course_content_store import get_course_content_store
from services.coursebuilder_jobs import new_coursebuilder_generation_id
from services.learning_map_service import get_learning_map_service
from services.storage_service import get_storage


logger = logging.getLogger(__name__)

PASS_SCORE = 0.7
# Change this one value to control CourseBuilder chapter locking:
# False = every chapter is open; True = later chapters unlock after the previous quiz is passed.
LOCK_COURSEBUILDER_CHAPTERS = False
LOCAL_FALLBACK_MODEL = "gemma4:e2b"
MAX_CHAPTERS = 10
MAX_LESSONS_PER_CHAPTER = 12
MAX_BLOCKS_PER_LESSON = 5
MAX_QUIZ_QUESTIONS = 7
MAX_MARKDOWN_PLANNING_CHARS = 70000


class _OutlineLesson(BaseModel):
    title: str
    learning_objectives: list[str] = Field(default_factory=list)
    source_queries: list[str] = Field(default_factory=list)
    objective_ids: list[str] = Field(default_factory=list)


class _OutlineChapter(BaseModel):
    title: str
    description: str = ""
    phase_id: str | None = None
    objective_ids: list[str] = Field(default_factory=list)
    source_queries: list[str] = Field(default_factory=list)
    lessons: list[_OutlineLesson] = Field(default_factory=list)


class _CourseOutline(BaseModel):
    title: str
    description: str = ""
    learning_objectives: list[str] = Field(default_factory=list)
    prerequisites: list[str] = Field(default_factory=list)
    language: str = "auto"
    chapters: list[_OutlineChapter] = Field(default_factory=list)


class _LessonBlockCandidate(BaseModel):
    block_type: str = "explanation"
    title: str = ""
    content: str = ""
    data_json: dict[str, Any] = Field(default_factory=dict)
    source_chunk_ids: list[str] = Field(default_factory=list)


class _LessonContent(BaseModel):
    blocks: list[_LessonBlockCandidate] = Field(default_factory=list)
    support_status: str = "supported"


class _LessonBlockPlan(BaseModel):
    block_type: str = "explanation"
    title: str = ""
    source_query: str = ""


class _LessonBlockPlanBatch(BaseModel):
    blocks: list[_LessonBlockPlan] = Field(default_factory=list)


class _QuizQuestionCandidate(BaseModel):
    prompt: str
    options: list[str] = Field(default_factory=list)
    correct_index: int = 0
    explanation: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)


class _QuizCandidate(BaseModel):
    questions: list[_QuizQuestionCandidate] = Field(default_factory=list)


@dataclass(slots=True)
class CourseBuilderContextPack:
    rich_summary: str
    documents: list[CourseDocumentRecord]
    sections: list[CourseSectionRecord]
    phases: list[CourseLearningPhaseRecord]
    objectives: list[CourseLearningObjectiveRecord]
    concepts: list[CourseConceptRecord]
    representative_chunks: list[SearchChunkRecord]
    source_structure: list[_OutlineChapter] = field(default_factory=list)
    markdown_planning_context: str = ""
    markdown_source_count: int = 0
    markdown_raw_chars: int = 0


class CourseBuilderService:
    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._rag = get_coursebuilder_rag_service()

    async def ensure_schema(self, session: AsyncSession) -> None:
        connection = await session.connection()

        def create_tables(sync_connection) -> None:  # noqa: ANN001
            for table in (
                CourseBuilderCourseRecord.__table__,
                CourseBuilderChapterRecord.__table__,
                CourseBuilderLessonRecord.__table__,
                CourseBuilderLessonBlockRecord.__table__,
                CourseBuilderQuizRecord.__table__,
                CourseBuilderQuizQuestionRecord.__table__,
                CourseBuilderChapterAttemptRecord.__table__,
                CourseBuilderProgressEventRecord.__table__,
            ):
                table.create(sync_connection, checkfirst=True)

        await connection.run_sync(create_tables)

    async def file_counts(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> tuple[int, int]:
        result = await session.execute(
            select(UploadedFile).where(UploadedFile.conversation_id == conversation_id)
        )
        files = list(result.scalars().all())
        pending = [file for file in files if file.status != "ready"]
        return len(files), len(pending)

    async def current_course(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> CourseBuilderCourseRecord | None:
        await self.ensure_schema(session)
        result = await session.execute(
            select(CourseBuilderCourseRecord)
            .where(CourseBuilderCourseRecord.conversation_id == conversation_id)
            .order_by(CourseBuilderCourseRecord.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def queue_course(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        llm_options: dict[str, Any] | None = None,
        restart_queued: bool = False,
    ) -> CourseBuilderCourseRecord:
        await self.ensure_schema(session)
        existing = await self.current_course(session, conversation_id)
        if existing and existing.status in {
            "queued",
            "analyzing",
            "generating_outline",
            "generating_chapters",
            "generating_lessons",
            "generating_quizzes",
            "validating",
        }:
            if existing.status != "queued" or not restart_queued:
                return existing

        course_id = _stable_id(conversation_id, "coursebuilder-course")
        course = await session.get(CourseBuilderCourseRecord, course_id)
        if course is None:
            course = CourseBuilderCourseRecord(
                id=course_id,
                conversation_id=conversation_id,
                title="",
                status="queued",
                generation_metadata={
                    "queued_at": datetime.now(timezone.utc).isoformat(),
                    "generation_id": new_coursebuilder_generation_id(),
                },
            )
            session.add(course)
        else:
            course.status = "queued"
            course.error = None
            course.generation_metadata = {
                **(course.generation_metadata or {}),
                "queued_at": datetime.now(timezone.utc).isoformat(),
                "generation_id": new_coursebuilder_generation_id(),
            }
        if llm_options:
            course.generation_metadata = {
                **(course.generation_metadata or {}),
                "llm_options": _safe_options(llm_options),
            }
        await self.record_event(
            session,
            conversation_id,
            course_id=course.id,
            stage="queued",
            message="Course generation has been queued.",
            percent=1,
        )
        await session.flush()
        return course

    async def rebuild_course(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        llm_options: dict[str, Any] | None = None,
    ) -> CourseBuilderRead:
        await self.ensure_schema(session)
        await self._clear_course(session, conversation_id)
        course = await self.queue_course(session, conversation_id, llm_options=llm_options)
        await session.flush()
        return await self.generate_course(session, conversation_id, llm_options=llm_options, course=course)

    async def generate_course(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        llm_options: dict[str, Any] | None = None,
        course: CourseBuilderCourseRecord | None = None,
    ) -> CourseBuilderRead:
        await self.ensure_schema(session)
        course = course or await self.queue_course(session, conversation_id, llm_options=llm_options)
        try:
            total_files, pending_files = await self.file_counts(session, conversation_id)
            if total_files == 0:
                raise ValueError("Upload course files before generating a course.")
            if pending_files:
                raise ValueError("Course generation waits until every uploaded file is ready.")

            course.status = "analyzing"
            await self.record_event(
                session,
                conversation_id,
                course_id=course.id,
                stage="analyzing",
                message="Reading processed chunks from this conversation.",
                percent=8,
            )
            await session.commit()
            chunks = await self._rag.load_chunks(session, conversation_id)
            if not chunks:
                raise ValueError("No processed chunks were found for this conversation.")
            context_pack = await self._context_pack(
                session,
                conversation_id,
                chunks,
                llm_options=llm_options,
            )
            retrieval_pool = _chunk_pool(context_pack.representative_chunks, chunks)

            course.status = "generating_outline"
            await self.record_event(
                session,
                conversation_id,
                course_id=course.id,
                stage="generating_outline",
                message="Designing the course outline.",
                percent=18,
            )
            await session.commit()
            outline = await self._outline(conversation_id, chunks, context_pack, llm_options=llm_options)
            source_structure_lesson_count = sum(len(chapter.lessons) for chapter in context_pack.source_structure)
            course.title = _clean_title(outline.title) or "Generated Course"
            course.description = outline.description.strip()
            course.learning_objectives = _clean_list(outline.learning_objectives)
            course.prerequisites = _clean_list(outline.prerequisites)
            course.language = _language_from_options(llm_options) or _safe_language(outline.language)
            course.generation_metadata = {
                **(course.generation_metadata or {}),
                "context_pack_version": "coursebuilder-context-pack-v1",
                "chunk_count": len(chunks),
                "representative_chunk_count": len(context_pack.representative_chunks),
                "rich_summary_chars": len(context_pack.rich_summary),
                "plan_phase_count": len(context_pack.phases),
                "plan_objective_count": len(context_pack.objectives),
                "concept_count": len(context_pack.concepts),
                "source_structure_chapter_count": len(context_pack.source_structure),
                "source_structure_lesson_count": source_structure_lesson_count,
                "markdown_source_count": context_pack.markdown_source_count,
                "markdown_raw_chars": context_pack.markdown_raw_chars,
                "markdown_planning_chars": len(context_pack.markdown_planning_context),
                "chapter_count": len(outline.chapters),
                "chapter_retrieval_count": 0,
                "lesson_retrieval_count": 0,
                "block_retrieval_count": 0,
            }
            retrieval_counts = {
                "chapter_retrieval_count": 0,
                "lesson_retrieval_count": 0,
                "block_retrieval_count": 0,
            }
            await self._delete_course_children(session, course.id)
            await session.flush()

            course.status = "generating_chapters"
            await self.record_event(
                session,
                conversation_id,
                course_id=course.id,
                stage="generating_chapters",
                message="Creating ordered chapters and lesson shells.",
                percent=30,
            )
            await session.commit()

            chapters = _usable_chapters(outline, chunks)
            all_chapters: list[CourseBuilderChapterRecord] = []
            chapter_chunks_by_id: dict[uuid.UUID, list[SearchChunkRecord]] = {}
            for chapter_index, chapter_candidate in enumerate(chapters):
                chapter_id = _stable_id(conversation_id, f"chapter:{chapter_index}:{chapter_candidate.title}")
                chapter_query = _chapter_query(chapter_candidate, context_pack)
                retrieved_chapter_chunks = await self._rag.retrieve_lesson_chunks(
                    session,
                    conversation_id,
                    chapter_query,
                    fallback_chunks=retrieval_pool,
                    top_k=10,
                )
                chapter_chunks = _title_supported_chunks(
                    retrieved_chapter_chunks,
                    chapter_candidate.title,
                    *chapter_candidate.source_queries,
                ) or retrieved_chapter_chunks
                retrieval_counts["chapter_retrieval_count"] += 1
                chapter = CourseBuilderChapterRecord(
                    id=chapter_id,
                    course_id=course.id,
                    conversation_id=conversation_id,
                    title=_clean_title(chapter_candidate.title) or f"Chapter {chapter_index + 1}",
                    description=chapter_candidate.description.strip(),
                    order_index=chapter_index,
                    summary=chapter_candidate.description.strip(),
                    source_chunk_ids=[chunk.id for chunk in chapter_chunks],
                    is_locked=_coursebuilder_chapter_locked(chapter_index, prior_completed=False),
                    unlock_rule=_coursebuilder_unlock_rule(),
                )
                session.add(chapter)
                all_chapters.append(chapter)
                chapter_chunks_by_id[chapter_id] = chapter_chunks
            await session.flush()

            course.status = "generating_lessons"
            await self.record_event(
                session,
                conversation_id,
                course_id=course.id,
                stage="generating_lessons",
                message="Writing grounded lessons with citations.",
                percent=45,
            )
            await session.commit()
            total_lesson_count = sum(
                len(_usable_lessons(chapter_candidate, chunks))
                for chapter_candidate in chapters
            )
            completed_lesson_count = 0
            for chapter_index, chapter in enumerate(all_chapters):
                chapter_candidate = chapters[chapter_index]
                chapter_chunks = chapter_chunks_by_id.get(chapter.id) or _chunks_by_ids(
                    chunks,
                    chapter.source_chunk_ids,
                )
                lesson_candidates = _usable_lessons(chapter_candidate, chunks)
                for lesson_index, lesson_candidate in enumerate(lesson_candidates):
                    lesson_query = _lesson_query(chapter_candidate, lesson_candidate, context_pack)
                    lesson_retrieval_pool = _chunk_pool(chapter_chunks, retrieval_pool)
                    lesson_chunks = await self._rag.retrieve_lesson_chunks(
                        session,
                        conversation_id,
                        lesson_query,
                        fallback_chunks=lesson_retrieval_pool,
                        top_k=8,
                    )
                    lesson_specific_chunks = _title_supported_chunks(
                        lesson_chunks,
                        lesson_candidate.title,
                        *lesson_candidate.source_queries,
                        *lesson_candidate.learning_objectives,
                    )
                    lesson_chunks = lesson_specific_chunks or _title_supported_chunks(
                        lesson_chunks,
                        chapter_candidate.title,
                        *chapter_candidate.source_queries,
                    )
                    retrieval_counts["lesson_retrieval_count"] += 1
                    lesson_id = _stable_id(
                        conversation_id,
                        f"lesson:{chapter_index}:{lesson_index}:{lesson_candidate.title}",
                    )
                    lesson_content = await self._lesson_content(
                        session,
                        conversation_id,
                        chapter,
                        lesson_candidate,
                        lesson_chunks,
                        fallback_chunks=lesson_retrieval_pool,
                        llm_options=llm_options,
                    )
                    citations = self._rag.citations_for(lesson_chunks)
                    lesson = CourseBuilderLessonRecord(
                        id=lesson_id,
                        chapter_id=chapter.id,
                        course_id=course.id,
                        conversation_id=conversation_id,
                        title=_clean_title(lesson_candidate.title) or f"Lesson {lesson_index + 1}",
                        order_index=lesson_index,
                        learning_objectives=_clean_list(lesson_candidate.learning_objectives),
                        source_chunk_ids=[chunk.id for chunk in lesson_chunks],
                        support_status=support_status(citations),
                    )
                    session.add(lesson)
                    await session.flush()
                    blocks = lesson_content.blocks[:MAX_BLOCKS_PER_LESSON] or _fallback_lesson_blocks(
                        lesson.title,
                        lesson_chunks,
                    )
                    block_retrievals = (
                        int(lesson_content.support_status.removeprefix("block_retrievals:"))
                        if lesson_content.support_status.startswith("block_retrievals:")
                        else 0
                    )
                    retrieval_counts["block_retrieval_count"] += block_retrievals
                    for block_index, block_candidate in enumerate(blocks):
                        block_candidate.source_chunk_ids = _valid_source_chunk_ids(
                            block_candidate.source_chunk_ids,
                            lesson_chunks,
                        )
                        content = block_candidate.content.strip()
                        if not content:
                            fallback_blocks = _fallback_lesson_blocks(
                                block_candidate.title or lesson.title,
                                _chunks_by_ids(lesson_chunks, block_candidate.source_chunk_ids) or lesson_chunks,
                            )
                            if fallback_blocks:
                                fallback_block = fallback_blocks[0]
                                content = fallback_block.content.strip()
                                block_candidate.source_chunk_ids = _valid_source_chunk_ids(
                                    fallback_block.source_chunk_ids or block_candidate.source_chunk_ids,
                                    lesson_chunks,
                                )
                        block_citations = self._rag.citations_for(
                            lesson_chunks,
                            block_candidate.source_chunk_ids,
                        )
                        block_type = normalize_block_type(block_candidate.block_type)
                        data_json = block_candidate.data_json or {}
                        if block_type == "chart":
                            data_json = validate_chart_spec(data_json)
                        validation_status = (
                            "supported"
                            if block_citations and content
                            else "insufficient_source_material"
                        )
                        if not block_citations or not content:
                            content = insufficient_source_message()
                        session.add(
                            CourseBuilderLessonBlockRecord(
                                id=_stable_id(
                                    conversation_id,
                                    f"block:{chapter_index}:{lesson_index}:{block_index}:{block_type}",
                                ),
                                lesson_id=lesson.id,
                                block_type=block_type,
                                title=_clean_title(block_candidate.title),
                                content=content,
                                order_index=block_index,
                                data_json=data_json,
                                source_citations=block_citations,
                                validation_status=validation_status,
                            )
                        )
                    course.generation_metadata = {
                        **(course.generation_metadata or {}),
                        **retrieval_counts,
                    }
                    completed_lesson_count += 1
                    course.generation_metadata = {
                        **(course.generation_metadata or {}),
                        "lesson_generation_count": completed_lesson_count,
                        "lesson_generation_total": total_lesson_count,
                    }
                    lesson_percent = 45.0
                    if total_lesson_count:
                        lesson_percent += min(25.0, 25.0 * completed_lesson_count / total_lesson_count)
                    await self.record_event(
                        session,
                        conversation_id,
                        course_id=course.id,
                        stage="generating_lessons",
                        message=(
                            f"Generated lesson {completed_lesson_count}/"
                            f"{total_lesson_count}: {lesson.title}"
                        ),
                        percent=lesson_percent,
                        metadata={
                            "lesson_id": str(lesson.id),
                            "chapter_id": str(chapter.id),
                            "completed_lessons": completed_lesson_count,
                            "total_lessons": total_lesson_count,
                        },
                    )
                    await session.commit()

            course.status = "generating_quizzes"
            await self.record_event(
                session,
                conversation_id,
                course_id=course.id,
                stage="generating_quizzes",
                message="Building chapter quizzes.",
                percent=75,
            )
            await session.commit()
            for chapter in all_chapters:
                chapter_chunks = _chunks_by_ids(chunks, chapter.source_chunk_ids) or chunks[:8]
                quiz_id = _stable_id(conversation_id, f"quiz:{chapter.order_index}:{chapter.title}")
                quiz_candidate = await self._quiz(chapter, chapter_chunks, llm_options=llm_options)
                question_rows = _valid_quiz_question_rows(
                    quiz_candidate.questions[:MAX_QUIZ_QUESTIONS],
                    chapter_chunks,
                    self._rag,
                )
                if not question_rows:
                    question_rows = _valid_quiz_question_rows(
                        _fallback_questions(chapter, chapter_chunks),
                        chapter_chunks,
                        self._rag,
                    )
                quiz = CourseBuilderQuizRecord(
                    id=quiz_id,
                    chapter_id=chapter.id,
                    course_id=course.id,
                    pass_score=PASS_SCORE,
                    question_count=len(question_rows),
                    source_chunk_ids=[chunk.id for chunk in chapter_chunks],
                )
                session.add(quiz)
                await session.flush()
                for question_index, (candidate, options, correct_index, citations) in enumerate(question_rows):
                    session.add(
                        CourseBuilderQuizQuestionRecord(
                            id=_stable_id(
                                conversation_id,
                                f"quiz-question:{chapter.order_index}:{question_index}:{candidate.prompt}",
                            ),
                            quiz_id=quiz.id,
                            chapter_id=chapter.id,
                            question_type="mcq",
                            prompt=candidate.prompt.strip(),
                            options=options,
                            answer_key={"correct_index": correct_index},
                            explanation=candidate.explanation.strip(),
                            source_citations=citations,
                            order_index=question_index,
                        )
                    )

            course.status = "validating"
            course.generation_metadata = {
                **(course.generation_metadata or {}),
                **retrieval_counts,
            }
            await self.record_event(
                session,
                conversation_id,
                course_id=course.id,
                stage="validating",
                message="Validating source support and unlock rules.",
                percent=90,
            )
            await session.commit()
            await self._normalize_unlocks(session, course.id)
            course.status = "ready"
            course.error = None
            await self.record_event(
                session,
                conversation_id,
                course_id=course.id,
                stage="ready",
                message="Your course is ready.",
                percent=100,
            )
            await session.commit()
            return await self.get_course(session, conversation_id)
        except Exception as exc:
            logger.exception("CourseBuilder generation failed for conversation %s", conversation_id)
            failed_course_id = course.id
            await session.rollback()
            course = await session.get(CourseBuilderCourseRecord, failed_course_id) or course
            course.status = "failed"
            course.error = f"{type(exc).__name__}: {exc}"
            await self.record_event(
                session,
                conversation_id,
                course_id=course.id,
                stage="failed",
                message=course.error,
                percent=100,
            )
            await session.commit()
            return await self.get_course(session, conversation_id)

    async def get_course(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> CourseBuilderRead:
        await self.ensure_schema(session)
        total_files, pending_files = await self.file_counts(session, conversation_id)
        course = await self.current_course(session, conversation_id)
        events = await self._events(session, conversation_id, course.id if course else None)
        if course is None:
            return CourseBuilderRead(
                conversation_id=conversation_id,
                status="queued" if total_files else "queued",
                progress_events=events,
                pending_file_count=pending_files,
                total_file_count=total_files,
            )

        chapters = await self._read_chapters(session, course.id)
        return CourseBuilderRead(
            id=course.id,
            conversation_id=conversation_id,
            title=course.title,
            description=course.description,
            learning_objectives=course.learning_objectives or [],
            prerequisites=course.prerequisites or [],
            status=course.status,
            language=course.language,
            error=course.error,
            generation_metadata=course.generation_metadata or {},
            chapters=chapters,
            progress_events=events,
            pending_file_count=pending_files,
            total_file_count=total_files,
        )

    async def submit_quiz(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        chapter_id: uuid.UUID,
        answers: dict[uuid.UUID, str | int],
    ) -> CourseBuilderQuizSubmitResponse:
        await self.ensure_schema(session)
        course = await self.current_course(session, conversation_id)
        if course is None:
            raise LookupError("course not found")
        chapter = await session.get(CourseBuilderChapterRecord, chapter_id)
        if chapter is None or chapter.course_id != course.id or chapter.conversation_id != conversation_id:
            raise LookupError("chapter not found")
        if await self._chapter_is_locked(session, course.id, chapter):
            raise PermissionError("chapter is locked")
        quiz_result = await session.execute(
            select(CourseBuilderQuizRecord).where(CourseBuilderQuizRecord.chapter_id == chapter_id)
        )
        quiz = quiz_result.scalar_one_or_none()
        if quiz is None:
            raise LookupError("quiz not found")
        q_result = await session.execute(
            select(CourseBuilderQuizQuestionRecord)
            .where(CourseBuilderQuizQuestionRecord.quiz_id == quiz.id)
            .order_by(CourseBuilderQuizQuestionRecord.order_index.asc())
        )
        questions = list(q_result.scalars().all())
        results: list[CourseBuilderQuizResult] = []
        correct = 0
        for question in questions:
            raw_answer = answers.get(question.id)
            selected = _selected_index(raw_answer, question.options)
            correct_index = int((question.answer_key or {}).get("correct_index", 0))
            is_correct = selected == correct_index
            if is_correct:
                correct += 1
            results.append(
                CourseBuilderQuizResult(
                    question_id=question.id,
                    is_correct=is_correct,
                    correct_index=correct_index,
                    selected_index=selected,
                    feedback=question.explanation or ("Correct." if is_correct else "Review the cited lesson material."),
                )
            )
        score = correct / len(questions) if questions else 0.0
        passed = score >= quiz.pass_score
        session.add(
            CourseBuilderChapterAttemptRecord(
                conversation_id=conversation_id,
                course_id=course.id,
                chapter_id=chapter.id,
                quiz_id=quiz.id,
                score=score,
                passed=passed,
                answers=[
                    {"question_id": str(question_id), "answer": answer}
                    for question_id, answer in answers.items()
                ],
                feedback=[result.model_dump(mode="json") for result in results],
            )
        )
        if passed:
            next_chapter = await session.scalar(
                select(CourseBuilderChapterRecord)
                .where(
                    CourseBuilderChapterRecord.course_id == course.id,
                    CourseBuilderChapterRecord.order_index == chapter.order_index + 1,
                )
                .limit(1)
            )
            if next_chapter is not None:
                next_chapter.is_locked = False
        await session.flush()
        response_course = await self.get_course(session, conversation_id)
        response_chapter = next(item for item in response_course.chapters if item.id == chapter_id)
        return CourseBuilderQuizSubmitResponse(
            chapter=response_chapter,
            score=score,
            passed=passed,
            results=results,
            course=response_course,
        )

    async def clear_course(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> None:
        await self.ensure_schema(session)
        await self._clear_course(session, conversation_id)

    async def mark_course_failed(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        message: str,
        *,
        course_id: uuid.UUID | None = None,
    ) -> None:
        await self.ensure_schema(session)
        course = await session.get(CourseBuilderCourseRecord, course_id) if course_id else None
        if course is None:
            course = await self.current_course(session, conversation_id)
        if course is None:
            return
        course.status = "failed"
        course.error = message
        await self.record_event(
            session,
            conversation_id,
            course_id=course.id,
            stage="failed",
            message=message,
            percent=100,
        )
        await session.flush()

    async def record_event(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        course_id: uuid.UUID | None,
        stage: str,
        message: str,
        percent: float,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        session.add(
            CourseBuilderProgressEventRecord(
                conversation_id=conversation_id,
                course_id=course_id,
                stage=stage,
                message=message,
                percent=max(0.0, min(100.0, float(percent))),
                event_metadata=metadata or {},
            )
        )

    async def _context_pack(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        chunks: list[SearchChunkRecord],
        *,
        llm_options: dict[str, Any] | None,
    ) -> CourseBuilderContextPack:
        store = get_course_content_store()
        documents = await store.get_documents(session, conversation_id)
        sections = await store.get_sections(session, conversation_id)

        concept_service = get_concept_inventory_service()
        concepts = await concept_service.load_concepts(session, conversation_id)
        if not concepts:
            try:
                concepts = await concept_service.rebuild_concepts(
                    session,
                    conversation_id,
                    llm_options=llm_options,
                )
            except Exception:  # noqa: BLE001
                logger.exception("CourseBuilder concept context failed; continuing without concepts")
                await session.rollback()
                try:
                    concepts = await concept_service.load_concepts(session, conversation_id)
                except Exception:  # noqa: BLE001
                    logger.exception("CourseBuilder concept fallback load failed")
                    await session.rollback()
                    concepts = []

        map_service = get_learning_map_service()
        phases, objectives = await map_service.load_map(session, conversation_id)
        if not phases:
            try:
                phases, objectives = await map_service.rebuild_map(
                    session,
                    conversation_id,
                    llm_options=llm_options,
                )
            except Exception:  # noqa: BLE001
                logger.exception("CourseBuilder learning-map context failed; continuing without plan")
                await session.rollback()
                try:
                    phases, objectives = await map_service.load_map(session, conversation_id)
                except Exception:  # noqa: BLE001
                    logger.exception("CourseBuilder learning-map fallback load failed")
                    await session.rollback()
                    phases, objectives = [], []

        representative_chunks = select_representative_chunks(chunks)
        source_structure = extract_source_structure(chunks, sections)
        markdown_context, markdown_source_count, markdown_raw_chars = await _load_markdown_planning_context(
            documents
        )
        rich_summary = _build_rich_summary(
            documents,
            sections,
            concepts,
            phases,
            objectives,
            representative_chunks,
        )
        return CourseBuilderContextPack(
            rich_summary=rich_summary,
            documents=documents,
            sections=sections,
            phases=phases,
            objectives=objectives,
            concepts=concepts,
            representative_chunks=representative_chunks,
            source_structure=source_structure,
            markdown_planning_context=markdown_context,
            markdown_source_count=markdown_source_count,
            markdown_raw_chars=markdown_raw_chars,
        )

    async def _outline(
        self,
        conversation_id: uuid.UUID,
        chunks: list[SearchChunkRecord],
        context_pack: CourseBuilderContextPack,
        *,
        llm_options: dict[str, Any] | None,
    ) -> _CourseOutline:
        source_outline = _outline_from_source_structure(context_pack, chunks)
        markdown_outline = await self._markdown_outline(
            conversation_id,
            context_pack,
            source_outline=source_outline,
            llm_options=llm_options,
        )
        if markdown_outline:
            return markdown_outline
        if source_outline:
            return source_outline

        return _fallback_outline(chunks)

    async def _markdown_outline(
        self,
        conversation_id: uuid.UUID,
        context_pack: CourseBuilderContextPack,
        *,
        source_outline: _CourseOutline | None,
        llm_options: dict[str, Any] | None,
    ) -> _CourseOutline | None:
        if not context_pack.markdown_planning_context.strip():
            return None

        source_candidate = _outline_from_markdown_structure(context_pack)
        if source_candidate is None:
            return None

        messages = [
            {
                "role": "system",
                "content": (
                    "You are the CourseBuilder planning step for TeacherLM. "
                    "Normalize a source-extracted course plan without changing its skeleton. "
                    "Keep the exact chapter order and lesson/sub-chapter order from the provided "
                    "source outline. Do not add, remove, or rename chapters or lessons unless the "
                    "same title is explicitly supported by the parser markdown. You may only fill "
                    "description, learning objectives, prerequisite, language, and source_queries. "
                    "Return ONLY valid JSON "
                    "matching the requested schema. Do not return prose, summaries, or Markdown fences."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Normalize this source-extracted course outline. The chapters and lessons below "
                    "are the source of truth; preserve them as the course skeleton. Add source_queries "
                    "that include the chapter and lesson titles so retrieval can fetch evidence later.\n\n"
                    f"Conversation: {conversation_id}\n\n"
                    "Source outline to preserve:\n"
                    f"{_format_source_structure(source_candidate.chapters)}\n\n"
                    "Course plan phases/objectives hint:\n"
                    f"{_format_course_plan(context_pack.phases, context_pack.objectives)}\n\n"
                    "Parser markdown files:\n"
                    f"{context_pack.markdown_planning_context}"
                ),
            },
        ]
        try:
            outline = await self._structured(messages, _CourseOutline, llm_options=llm_options)
        except Exception:
            logger.warning("CourseBuilder markdown planning LLM failed; using parser markdown TOC skeleton")
            return source_candidate
        outline = _normalize_markdown_outline(outline, context_pack)
        return _merge_outline_metadata(outline, source_candidate)

    async def _lesson_content(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        chapter: CourseBuilderChapterRecord,
        lesson: _OutlineLesson,
        chunks: list[SearchChunkRecord],
        *,
        fallback_chunks: list[SearchChunkRecord],
        llm_options: dict[str, Any] | None,
    ) -> _LessonContent:
        if not chunks:
            return _LessonContent(
                blocks=_fallback_lesson_blocks(lesson.title, []),
                support_status="insufficient_source_material",
            )
        block_plans = await self._lesson_block_plan(chapter, lesson, chunks, llm_options=llm_options)
        if not block_plans:
            return await self._lesson_content_from_chunks(chapter, lesson, chunks, llm_options=llm_options)

        blocks: list[_LessonBlockCandidate] = []
        retrieval_count = 0
        for plan_index, plan in enumerate(block_plans[:MAX_BLOCKS_PER_LESSON]):
            query = "\n".join(
                item
                for item in [
                    chapter.title,
                    lesson.title,
                    plan.title,
                    plan.source_query,
                    " ".join(lesson.learning_objectives),
                    " ".join(lesson.source_queries),
                ]
                if item
            )
            block_chunks = await self._rag.retrieve_chunks(
                session,
                conversation_id,
                query,
                fallback_chunks=chunks,
                top_k=6,
            )
            retrieval_count += 1
            block = await self._block_content(chapter, lesson, plan, block_chunks, llm_options=llm_options)
            block.source_chunk_ids = _valid_source_chunk_ids(block.source_chunk_ids, block_chunks)
            blocks.append(block)
        if blocks:
            return _LessonContent(blocks=blocks, support_status=f"block_retrievals:{retrieval_count}")
        return await self._lesson_content_from_chunks(chapter, lesson, chunks, llm_options=llm_options)

    async def _lesson_content_from_chunks(
        self,
        chapter: CourseBuilderChapterRecord,
        lesson: _OutlineLesson,
        chunks: list[SearchChunkRecord],
        *,
        llm_options: dict[str, Any] | None,
    ) -> _LessonContent:
        if not chunks:
            return _LessonContent(
                blocks=_fallback_lesson_blocks(lesson.title, []),
                support_status="insufficient_source_material",
            )
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate concise text-course lesson blocks grounded only in the provided chunks. "
                    "Every block must cite chunk ids from the provided source list. "
                    "Use block types only from: explanation, definition, example, table, equation, "
                    "chart, diagram, procedure, warning, summary. If support is weak, say so clearly."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Chapter: {chapter.title}\nLesson: {lesson.title}\n"
                    f"Learning objectives: {lesson.learning_objectives}\n\n"
                    f"Source chunks:\n{_chunk_context(chunks, max_chars=9000)}"
                ),
            },
        ]
        try:
            return await self._structured(messages, _LessonContent, llm_options=llm_options)
        except Exception:
            logger.exception("CourseBuilder lesson LLM failed; using fallback")
            return _LessonContent(blocks=_fallback_lesson_blocks(lesson.title, chunks))

    async def _lesson_block_plan(
        self,
        chapter: CourseBuilderChapterRecord,
        lesson: _OutlineLesson,
        chunks: list[SearchChunkRecord],
        *,
        llm_options: dict[str, Any] | None,
    ) -> list[_LessonBlockPlan]:
        if not chunks:
            return []
        messages = [
            {
                "role": "system",
                "content": (
                    "Plan the concise blocks needed for one course lesson. "
                    "Return only JSON. Do not write final lesson content yet. "
                    "Each block needs a type, title, and a source_query for retrieving exact source chunks."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Chapter: {chapter.title}\nLesson: {lesson.title}\n"
                    f"Learning objectives: {lesson.learning_objectives}\n"
                    f"Lesson source queries: {lesson.source_queries}\n\n"
                    "Available lesson chunks:\n"
                    f"{_chunk_context(chunks, max_chars=5000)}"
                ),
            },
        ]
        try:
            planned = await self._structured(messages, _LessonBlockPlanBatch, llm_options=llm_options)
            return [
                plan
                for plan in planned.blocks
                if _clean_title(plan.title) or str(plan.source_query or "").strip()
            ][:MAX_BLOCKS_PER_LESSON]
        except Exception:
            logger.exception("CourseBuilder block planning LLM failed; using fallback")
            return _fallback_block_plans(lesson.title, chunks)

    async def _block_content(
        self,
        chapter: CourseBuilderChapterRecord,
        lesson: _OutlineLesson,
        plan: _LessonBlockPlan,
        chunks: list[SearchChunkRecord],
        *,
        llm_options: dict[str, Any] | None,
    ) -> _LessonBlockCandidate:
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate exactly one concise lesson block grounded only in the provided chunks. "
                    "Use exact chunk ids from the provided source list in source_chunk_ids. "
                    "Do not invent facts. If support is weak, use a warning block."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Chapter: {chapter.title}\nLesson: {lesson.title}\n"
                    f"Block type: {plan.block_type}\nBlock title: {plan.title}\n"
                    f"Retrieval query: {plan.source_query}\n\n"
                    f"Source chunks:\n{_chunk_context(chunks, max_chars=7000)}"
                ),
            },
        ]
        try:
            block = await self._structured(messages, _LessonBlockCandidate, llm_options=llm_options)
            block.block_type = normalize_block_type(block.block_type or plan.block_type)
            block.title = _clean_title(block.title or plan.title)
            return block
        except Exception:
            logger.exception("CourseBuilder block LLM failed; using fallback")
            fallback = _fallback_lesson_blocks(plan.title or lesson.title, chunks)
            return fallback[0] if fallback else _LessonBlockCandidate(
                block_type="warning",
                title=plan.title,
                content=insufficient_source_message(),
            )

    async def _quiz(
        self,
        chapter: CourseBuilderChapterRecord,
        chunks: list[SearchChunkRecord],
        *,
        llm_options: dict[str, Any] | None,
    ) -> _QuizCandidate:
        messages = [
            {
                "role": "system",
                "content": (
                    "Generate source-grounded MCQ quiz questions. Each question must have "
                    "4 options, exactly one correct_index, a short explanation, and source_chunk_ids. "
                    "Questions should test understanding, not just labels."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Chapter: {chapter.title}\nDescription: {chapter.description}\n"
                    "Generate 5-7 MCQ questions from these chunks only:\n"
                    f"{_chunk_context(chunks, max_chars=10000)}"
                ),
            },
        ]
        try:
            return await self._structured(messages, _QuizCandidate, llm_options=llm_options)
        except Exception:
            logger.exception("CourseBuilder quiz LLM failed; using fallback")
            return _QuizCandidate(questions=_fallback_questions(chapter, chunks))

    async def _structured[T: BaseModel](
        self,
        messages: list[dict[str, str]],
        schema: type[T],
        *,
        llm_options: dict[str, Any] | None,
    ) -> T:
        clients = self._llm_clients(llm_options)
        last_error: Exception | None = None
        for client in clients:
            try:
                return await client.chat_structured(
                    messages,
                    schema,
                    options={"temperature": 0.15, "num_predict": 4096},
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        raise RuntimeError(f"all CourseBuilder LLM clients failed: {last_error}")

    def _llm_clients(self, llm_options: dict[str, Any] | None) -> list[OllamaClient]:
        clients: list[OllamaClient] = []
        override = (llm_options or {}).get("llm") if isinstance(llm_options, dict) else None
        if has_llm_override(override if isinstance(override, dict) else None):
            kwargs = build_llm_client_kwargs(
                default_base_url=self._settings.ollama_host,
                default_model=self._settings.ollama_chat_model,
                options=override,
            )
            clients.append(OllamaClient(**kwargs))  # type: ignore[arg-type]
        clients.append(
            OllamaClient(
                base_url=self._settings.ollama_host,
                model=LOCAL_FALLBACK_MODEL,
                provider="ollama",
            )
        )
        if LOCAL_FALLBACK_MODEL != self._settings.ollama_chat_model:
            clients.append(
                OllamaClient(
                    base_url=self._settings.ollama_host,
                    model=self._settings.ollama_chat_model,
                    provider="ollama",
                )
            )
        return clients

    async def _clear_course(self, session: AsyncSession, conversation_id: uuid.UUID) -> None:
        existing = await session.execute(
            select(CourseBuilderCourseRecord.id).where(
                CourseBuilderCourseRecord.conversation_id == conversation_id
            )
        )
        ids = list(existing.scalars().all())
        for course_id in ids:
            await self._delete_course_children(session, course_id, include_events=True)
        await session.execute(
            delete(CourseBuilderCourseRecord).where(
                CourseBuilderCourseRecord.conversation_id == conversation_id
            )
        )
        await session.flush()

    async def _delete_course_children(
        self,
        session: AsyncSession,
        course_id: uuid.UUID,
        *,
        include_events: bool = False,
    ) -> None:
        await session.execute(delete(CourseBuilderChapterAttemptRecord).where(CourseBuilderChapterAttemptRecord.course_id == course_id))
        quiz_ids = list(
            (
                await session.execute(
                    select(CourseBuilderQuizRecord.id).where(CourseBuilderQuizRecord.course_id == course_id)
                )
            )
            .scalars()
            .all()
        )
        if quiz_ids:
            await session.execute(delete(CourseBuilderQuizQuestionRecord).where(CourseBuilderQuizQuestionRecord.quiz_id.in_(quiz_ids)))
        await session.execute(delete(CourseBuilderQuizRecord).where(CourseBuilderQuizRecord.course_id == course_id))
        lesson_ids = list(
            (
                await session.execute(
                    select(CourseBuilderLessonRecord.id).where(CourseBuilderLessonRecord.course_id == course_id)
                )
            )
            .scalars()
            .all()
        )
        if lesson_ids:
            await session.execute(delete(CourseBuilderLessonBlockRecord).where(CourseBuilderLessonBlockRecord.lesson_id.in_(lesson_ids)))
        await session.execute(delete(CourseBuilderLessonRecord).where(CourseBuilderLessonRecord.course_id == course_id))
        await session.execute(delete(CourseBuilderChapterRecord).where(CourseBuilderChapterRecord.course_id == course_id))
        if include_events:
            await session.execute(delete(CourseBuilderProgressEventRecord).where(CourseBuilderProgressEventRecord.course_id == course_id))

    async def _normalize_unlocks(self, session: AsyncSession, course_id: uuid.UUID) -> None:
        result = await session.execute(
            select(CourseBuilderChapterRecord)
            .where(CourseBuilderChapterRecord.course_id == course_id)
            .order_by(CourseBuilderChapterRecord.order_index.asc())
        )
        for index, chapter in enumerate(result.scalars().all()):
            chapter.is_locked = _coursebuilder_chapter_locked(index, prior_completed=False)

    async def _chapter_is_locked(
        self,
        session: AsyncSession,
        course_id: uuid.UUID,
        chapter: CourseBuilderChapterRecord,
    ) -> bool:
        if not LOCK_COURSEBUILDER_CHAPTERS:
            return False
        if chapter.order_index <= 0:
            return False
        previous_chapter_id = await session.scalar(
            select(CourseBuilderChapterRecord.id)
            .where(
                CourseBuilderChapterRecord.course_id == course_id,
                CourseBuilderChapterRecord.order_index == chapter.order_index - 1,
            )
            .limit(1)
        )
        if previous_chapter_id is None:
            return False
        previous_passed = await session.scalar(
            select(CourseBuilderChapterAttemptRecord.id)
            .where(
                CourseBuilderChapterAttemptRecord.course_id == course_id,
                CourseBuilderChapterAttemptRecord.chapter_id == previous_chapter_id,
                CourseBuilderChapterAttemptRecord.passed.is_(True),
            )
            .limit(1)
        )
        return previous_passed is None

    async def _events(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        course_id: uuid.UUID | None,
    ) -> list[CourseBuilderProgressEventRead]:
        stmt = (
            select(CourseBuilderProgressEventRecord)
            .where(CourseBuilderProgressEventRecord.conversation_id == conversation_id)
            .order_by(CourseBuilderProgressEventRecord.created_at.desc())
            .limit(80)
        )
        if course_id:
            stmt = stmt.where(
                (CourseBuilderProgressEventRecord.course_id == course_id)
                | (CourseBuilderProgressEventRecord.course_id.is_(None))
            )
        result = await session.execute(stmt)
        records = list(result.scalars().all())
        records.reverse()
        return [_event_read(record) for record in records]

    async def _read_chapters(
        self,
        session: AsyncSession,
        course_id: uuid.UUID,
    ) -> list[CourseBuilderChapterRead]:
        chapters = list(
            (
                await session.execute(
                    select(CourseBuilderChapterRecord)
                    .where(CourseBuilderChapterRecord.course_id == course_id)
                    .order_by(CourseBuilderChapterRecord.order_index.asc())
                )
            )
            .scalars()
            .all()
        )
        attempts = list(
            (
                await session.execute(
                    select(CourseBuilderChapterAttemptRecord).where(
                        CourseBuilderChapterAttemptRecord.course_id == course_id
                    )
                )
            )
            .scalars()
            .all()
        )
        attempts_by_chapter: dict[uuid.UUID, list[CourseBuilderChapterAttemptRecord]] = {}
        for attempt in attempts:
            attempts_by_chapter.setdefault(attempt.chapter_id, []).append(attempt)
        prior_completed = True
        reads: list[CourseBuilderChapterRead] = []
        for chapter in chapters:
            chapter_attempts = attempts_by_chapter.get(chapter.id, [])
            best_score = max((attempt.score for attempt in chapter_attempts), default=0.0)
            completed = any(attempt.passed for attempt in chapter_attempts)
            strict_locked = _coursebuilder_chapter_locked(chapter.order_index, prior_completed=prior_completed)
            chapter.is_locked = strict_locked
            lessons = await self._read_lessons(session, chapter.id)
            quiz = await self._read_quiz(session, chapter.id)
            reads.append(
                CourseBuilderChapterRead(
                    id=chapter.id,
                    course_id=chapter.course_id,
                    title=chapter.title,
                    description=chapter.description,
                    order_index=chapter.order_index,
                    summary=chapter.summary,
                    source_chunk_ids=chapter.source_chunk_ids or [],
                    is_locked=strict_locked,
                    unlock_rule=chapter.unlock_rule or {},
                    best_score=best_score,
                    attempts=len(chapter_attempts),
                    completed=completed,
                    lessons=lessons,
                    quiz=quiz,
                )
            )
            prior_completed = completed
        return reads

    async def _read_lessons(
        self,
        session: AsyncSession,
        chapter_id: uuid.UUID,
    ) -> list[CourseBuilderLessonRead]:
        lessons = list(
            (
                await session.execute(
                    select(CourseBuilderLessonRecord)
                    .where(CourseBuilderLessonRecord.chapter_id == chapter_id)
                    .order_by(CourseBuilderLessonRecord.order_index.asc())
                )
            )
            .scalars()
            .all()
        )
        out: list[CourseBuilderLessonRead] = []
        for lesson in lessons:
            blocks = list(
                (
                    await session.execute(
                        select(CourseBuilderLessonBlockRecord)
                        .where(CourseBuilderLessonBlockRecord.lesson_id == lesson.id)
                        .order_by(CourseBuilderLessonBlockRecord.order_index.asc())
                    )
                )
                .scalars()
                .all()
            )
            out.append(
                CourseBuilderLessonRead(
                    id=lesson.id,
                    chapter_id=lesson.chapter_id,
                    title=lesson.title,
                    order_index=lesson.order_index,
                    learning_objectives=lesson.learning_objectives or [],
                    source_chunk_ids=lesson.source_chunk_ids or [],
                    support_status=lesson.support_status,
                    blocks=[
                        CourseBuilderLessonBlockRead(
                            id=block.id,
                            lesson_id=block.lesson_id,
                            block_type=block.block_type,
                            title=block.title,
                            content=block.content,
                            order_index=block.order_index,
                            data_json=block.data_json or {},
                            source_citations=[
                                CourseBuilderCitation.model_validate(citation)
                                for citation in (block.source_citations or [])
                            ],
                            validation_status=block.validation_status,
                        )
                        for block in blocks
                    ],
                )
            )
        return out

    async def _read_quiz(
        self,
        session: AsyncSession,
        chapter_id: uuid.UUID,
    ) -> CourseBuilderQuizRead | None:
        quiz = await session.scalar(
            select(CourseBuilderQuizRecord).where(CourseBuilderQuizRecord.chapter_id == chapter_id)
        )
        if quiz is None:
            return None
        questions = list(
            (
                await session.execute(
                    select(CourseBuilderQuizQuestionRecord)
                    .where(CourseBuilderQuizQuestionRecord.quiz_id == quiz.id)
                    .order_by(CourseBuilderQuizQuestionRecord.order_index.asc())
                )
            )
            .scalars()
            .all()
        )
        return CourseBuilderQuizRead(
            id=quiz.id,
            chapter_id=quiz.chapter_id,
            pass_score=quiz.pass_score,
            question_count=quiz.question_count,
            source_chunk_ids=quiz.source_chunk_ids or [],
            questions=[
                CourseBuilderQuizQuestionRead(
                    id=question.id,
                    quiz_id=question.quiz_id,
                    chapter_id=question.chapter_id,
                    question_type=question.question_type,
                    prompt=question.prompt,
                    options=question.options or [],
                    explanation=question.explanation,
                    order_index=question.order_index,
                    source_citations=[
                        CourseBuilderCitation.model_validate(citation)
                        for citation in (question.source_citations or [])
                    ],
                )
                for question in questions
            ],
        )


def _event_read(record: CourseBuilderProgressEventRecord) -> CourseBuilderProgressEventRead:
    return CourseBuilderProgressEventRead(
        id=record.id,
        conversation_id=record.conversation_id,
        course_id=record.course_id,
        stage=record.stage,
        message=record.message,
        percent=record.percent,
        metadata=record.event_metadata or {},
        created_at=record.created_at,
    )


def _stable_id(conversation_id: uuid.UUID, key: str) -> uuid.UUID:
    normalized = re.sub(r"\s+", " ", key.strip().lower())
    return uuid.uuid5(uuid.NAMESPACE_URL, f"teacherlm:coursebuilder:{conversation_id}:{normalized}")


def _safe_options(options: dict[str, Any]) -> dict[str, Any]:
    safe = dict(options)
    llm = safe.get("llm")
    if isinstance(llm, dict) and "api_key" in llm:
        safe["llm"] = {**llm, "api_key": "***"}
    return safe


def _coursebuilder_chapter_locked(
    order_index: int,
    *,
    prior_completed: bool,
    lock_chapters: bool | None = None,
) -> bool:
    enabled = LOCK_COURSEBUILDER_CHAPTERS if lock_chapters is None else lock_chapters
    return enabled and order_index > 0 and not prior_completed


def _coursebuilder_unlock_rule() -> dict[str, Any]:
    if not LOCK_COURSEBUILDER_CHAPTERS:
        return {"type": "none", "strict": False}
    return {
        "type": "previous_chapter_quiz",
        "pass_score": PASS_SCORE,
        "strict": True,
    }


def _language_from_options(options: dict[str, Any] | None) -> str | None:
    if not isinstance(options, dict):
        return None
    value = str(options.get("language") or "").strip().lower()
    return value if value and value not in {"auto", "__auto__"} else None


def _safe_language(value: str) -> str | None:
    language = str(value or "").strip().lower()
    return None if language in {"", "auto", "__auto__", "unknown"} else language[:32]


def _clean_title(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip())[:512]


def _clean_list(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        clean = re.sub(r"\s+", " ", str(value or "").strip())
        if not clean or clean.lower() in seen:
            continue
        seen.add(clean.lower())
        out.append(clean[:500])
    return out[:12]


def select_representative_chunks(
    chunks: list[SearchChunkRecord],
    *,
    max_chunks: int = 80,
) -> list[SearchChunkRecord]:
    if len(chunks) <= max_chunks:
        return list(chunks)

    selected: list[SearchChunkRecord] = []
    seen: set[str] = set()
    by_document: dict[uuid.UUID, list[SearchChunkRecord]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.document_id, []).append(chunk)

    per_document = max(1, max_chunks // max(1, len(by_document)))
    for document_chunks in by_document.values():
        for chunk in _spread_pick(document_chunks, per_document):
            if chunk.id not in seen:
                seen.add(chunk.id)
                selected.append(chunk)

    if len(selected) < max_chunks:
        by_section: dict[uuid.UUID, list[SearchChunkRecord]] = {}
        for chunk in chunks:
            by_section.setdefault(chunk.section_id, []).append(chunk)
        for section_chunks in by_section.values():
            for chunk in _spread_pick(section_chunks, 1):
                if chunk.id not in seen:
                    seen.add(chunk.id)
                    selected.append(chunk)
                    if len(selected) >= max_chunks:
                        return selected

    return selected[:max_chunks]


def _spread_pick(chunks: list[SearchChunkRecord], count: int) -> list[SearchChunkRecord]:
    if count <= 0 or not chunks:
        return []
    if len(chunks) <= count:
        return list(chunks)
    if count == 1:
        return [chunks[0]]
    indexes = {
        round(index * (len(chunks) - 1) / (count - 1))
        for index in range(count)
    }
    return [chunks[index] for index in sorted(indexes)]


def _chunk_pool(
    preferred: list[SearchChunkRecord],
    all_chunks: list[SearchChunkRecord],
) -> list[SearchChunkRecord]:
    out: list[SearchChunkRecord] = []
    seen: set[str] = set()
    for chunk in [*preferred, *all_chunks]:
        if chunk.id in seen:
            continue
        seen.add(chunk.id)
        out.append(chunk)
    return out


def _title_supported_chunks(
    chunks: list[SearchChunkRecord],
    *query_parts: str,
) -> list[SearchChunkRecord]:
    term_groups = [_support_terms([part]) for part in query_parts]
    term_groups = [group for group in term_groups if group]
    if not term_groups:
        return list(chunks)
    supported: list[SearchChunkRecord] = []
    for chunk in chunks:
        haystack = " ".join(
            [
                " ".join(chunk.heading_path or []),
                chunk.source_filename,
                chunk.text,
            ]
        ).casefold()
        if any(_matches_support_group(haystack, group) for group in term_groups):
            supported.append(chunk)
    return supported


def _matches_support_group(haystack: str, terms: list[str]) -> bool:
    if len(terms) <= 1:
        return bool(terms and terms[0] in haystack)
    return sum(1 for term in terms if term in haystack) >= min(2, len(terms))


def _support_terms(values: Any) -> list[str]:
    stopwords = {
        "and",
        "chapter",
        "course",
        "from",
        "into",
        "lesson",
        "module",
        "overview",
        "section",
        "study",
        "subchapter",
        "the",
        "this",
        "understand",
        "with",
    }
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for raw in re.findall(r"[\w][\w'_-]{2,}", str(value or "").casefold(), flags=re.UNICODE):
            term = raw.strip("_-'")
            if not term or term in stopwords or term.isdigit() or len(term) < 3:
                continue
            if term in seen:
                continue
            seen.add(term)
            out.append(term)
            if len(out) >= 16:
                return out
    return out


async def _load_markdown_planning_context(
    documents: list[CourseDocumentRecord],
) -> tuple[str, int, int]:
    storage = get_storage()
    parts: list[str] = []
    source_count = 0
    raw_chars = 0
    remaining = MAX_MARKDOWN_PLANNING_CHARS

    for index, document in enumerate(documents[:10], start=1):
        key = document.raw_markdown_path or document.cleaned_text_path
        if not key or remaining <= 0:
            continue
        try:
            markdown = await storage.get_text(key)
        except Exception:  # noqa: BLE001
            logger.exception("CourseBuilder could not load parser markdown for %s", document.source_filename)
            continue
        raw_chars += len(markdown)
        source_count += 1
        excerpt = _markdown_planning_view(markdown, max_chars=max(2000, remaining))
        block = (
            f"### Markdown source {index}: {document.source_filename}\n"
            f"title: {_clean_title(document.title or document.source_filename)}\n"
            f"storage_key: {key}\n\n"
            "```markdown\n"
            f"{excerpt}\n"
            "```"
        )
        if len(block) > remaining:
            block = block[:remaining].rsplit("\n", 1)[0].strip()
        parts.append(block)
        remaining -= len(block)

    return "\n\n".join(parts), source_count, raw_chars


def _markdown_planning_view(markdown: str, *, max_chars: int) -> str:
    text = str(markdown or "").strip()
    if len(text) <= max_chars:
        return text

    headings = _markdown_heading_lines(text)
    toc = _markdown_toc_lines(text)
    reserved = min(max_chars // 2, 30000)
    opening = text[:reserved].strip()
    planning_parts = [
        "# Extracted heading index",
        "\n".join(headings[:500]) or "(no markdown headings detected)",
        "",
        "# Likely table of contents lines",
        "\n".join(toc[:500]) or "(no table of contents lines detected)",
        "",
        "# Opening markdown excerpt",
        opening,
    ]
    view = "\n".join(planning_parts).strip()
    if len(view) > max_chars:
        view = view[:max_chars].rsplit("\n", 1)[0].strip()
    return view


def _markdown_heading_lines(markdown: str) -> list[str]:
    lines: list[str] = []
    for raw_line in markdown.splitlines():
        line = _clean_toc_line(raw_line)
        if re.match(r"^#{1,6}\s+\S+", line):
            lines.append(line)
    return _dedupe(lines)


def _markdown_toc_lines(markdown: str) -> list[str]:
    raw_lines = markdown.splitlines()
    selected: list[str] = []
    in_contents = False
    for index, raw_line in enumerate(raw_lines[:1200]):
        line = _clean_toc_line(raw_line)
        lower = line.lower()
        if lower in {"contents", "table of contents"} or lower.endswith(" contents"):
            in_contents = True
            selected.append(line)
            continue
        if in_contents and line:
            if re.match(r"^#{1,6}\s+(maps|preface|transcription|notes|index)\b", lower):
                break
            table_title = _html_table_cell_text(line)
            if table_title:
                page = _next_html_table_page(raw_lines, index)
                if page and _valid_structure_title(table_title):
                    selected.append(f"{table_title} {page}")
                continue
            selected.append(line)
        elif _toc_chapter_title(line) or _toc_lesson_title(line):
            selected.append(line)
    return _dedupe(line for line in selected if _valid_toc_line(line))


def _html_table_cell_text(line: str) -> str:
    match = re.search(r"<td[^>]*>(.*?)</td>", line, flags=re.IGNORECASE)
    if not match:
        return ""
    return re.sub(r"<[^>]+>", "", match.group(1)).strip()


def _next_html_table_page(lines: list[str], index: int) -> str:
    for raw_line in lines[index + 1 : index + 6]:
        cell = _html_table_cell_text(_clean_toc_line(raw_line))
        if re.fullmatch(r"\d{1,4}", cell):
            return cell
        if cell and not re.fullmatch(r"\d{1,4}", cell):
            return ""
    return ""


def _normalize_markdown_outline(
    outline: _CourseOutline,
    context_pack: CourseBuilderContextPack,
) -> _CourseOutline:
    chapters: list[_OutlineChapter] = []
    for chapter in _usable_chapters(outline, []):
        lessons: list[_OutlineLesson] = []
        for lesson in _usable_lessons(chapter, [])[:MAX_LESSONS_PER_CHAPTER]:
            lesson_queries = _clean_list([*lesson.source_queries, chapter.title, lesson.title])
            lessons.append(
                _OutlineLesson(
                    title=_clean_title(lesson.title),
                    learning_objectives=_clean_list(lesson.learning_objectives)
                    or [f"Understand {lesson.title}"],
                    source_queries=lesson_queries,
                    objective_ids=_clean_list(lesson.objective_ids),
                )
            )
        chapters.append(
            _OutlineChapter(
                title=_clean_title(chapter.title),
                description=chapter.description.strip()
                or f"Study {chapter.title} from the uploaded course materials.",
                phase_id=chapter.phase_id,
                objective_ids=_clean_list(chapter.objective_ids),
                source_queries=_clean_list([*chapter.source_queries, chapter.title]),
                lessons=lessons,
            )
        )

    fallback_title = _clean_title(next((doc.title for doc in context_pack.documents if doc.title), ""))
    return _CourseOutline(
        title=_clean_title(outline.title) or fallback_title or "Generated Course",
        description=outline.description.strip()
        or "A structured course generated from the uploaded parser markdown.",
        learning_objectives=_clean_list(outline.learning_objectives)
        or [f"Understand {chapter.title}" for chapter in chapters[:5]],
        prerequisites=_clean_list(outline.prerequisites),
        language=_safe_language(outline.language) or "auto",
        chapters=chapters[:MAX_CHAPTERS],
    )


def _outline_from_markdown_structure(context_pack: CourseBuilderContextPack) -> _CourseOutline | None:
    chapters = _source_structure_from_markdown_text(context_pack.markdown_planning_context)
    if _source_structure_score(chapters) < 3:
        return None
    fallback_title = _clean_title(next((doc.title for doc in context_pack.documents if doc.title), ""))
    markdown_title = _title_from_markdown_text(context_pack.markdown_planning_context)
    return _CourseOutline(
        title=markdown_title or fallback_title or "Generated Course",
        description="A structured course generated from the parser markdown table of contents.",
        learning_objectives=[f"Understand {chapter.title}" for chapter in chapters[:5]],
        chapters=chapters,
    )


def _title_from_markdown_text(markdown: str) -> str:
    headings: list[tuple[int, str]] = []
    for raw_line in str(markdown or "").splitlines()[:200]:
        match = re.match(r"^(#{1,3})\s+(.+)$", raw_line.strip())
        if not match:
            continue
        level = len(match.group(1))
        title = _clean_structure_title(re.sub(r"<[^>]+>", "", match.group(2)))
        if not _valid_course_title_candidate(title):
            continue
        headings.append((level, title))

    for index, (level, title) in enumerate(headings):
        if level != 1:
            continue
        subtitle = next(
            (
                candidate
                for next_level, candidate in headings[index + 1 : index + 4]
                if next_level == 2 and _valid_course_title_candidate(candidate)
            ),
            "",
        )
        if subtitle and subtitle.casefold() not in title.casefold():
            return f"{_title_case(title)}: {_title_case(subtitle)}"
        return _title_case(title)
    return _title_case(headings[0][1]) if headings else ""


def _valid_course_title_candidate(value: str) -> bool:
    text = _clean_title(value)
    lower = text.lower()
    if not _valid_structure_title(text):
        return False
    if lower in {
        "new edition",
        "oneworld",
        "contents",
        "maps",
        "preface",
        "extracted heading index",
        "likely table of contents lines",
        "opening markdown excerpt",
    }:
        return False
    if re.fullmatch(r"c\\.?\\s*r\\.?\\s*pennell", lower):
        return False
    return True


def _title_case(value: str) -> str:
    if value.isupper():
        return value.title()
    return value


def _source_structure_from_markdown_text(markdown: str) -> list[_OutlineChapter]:
    text = str(markdown or "")
    marker = "# Likely table of contents lines"
    if marker in text:
        text = text.split(marker, 1)[1]
    return _source_structure_from_toc_lines(_markdown_toc_lines(text))


def extract_source_structure(
    chunks: list[SearchChunkRecord],
    sections: list[CourseSectionRecord] | None = None,
) -> list[_OutlineChapter]:
    """Extract chapter/sub-chapter skeletons from parsed headings or table-of-contents text."""
    candidates = [
        _source_structure_from_sections(sections or []),
        _source_structure_from_chunk_headings(chunks),
        _source_structure_from_toc(chunks, sections or []),
    ]
    best = max(candidates, key=_source_structure_score, default=[])
    return best if _source_structure_score(best) >= 3 else []


def _source_structure_from_sections(sections: list[CourseSectionRecord]) -> list[_OutlineChapter]:
    items = [
        (section.heading_path or [section.title], section.summary or section.text)
        for section in sorted(sections, key=lambda item: (str(item.document_id), item.order_index))
        if section.title or section.heading_path
    ]
    return _source_structure_from_paths(items)


def _source_structure_from_chunk_headings(chunks: list[SearchChunkRecord]) -> list[_OutlineChapter]:
    items = [
        (chunk.heading_path or [], chunk.text)
        for chunk in chunks
        if chunk.heading_path
    ]
    return _source_structure_from_paths(items)


def _source_structure_from_paths(items: list[tuple[list[str], str]]) -> list[_OutlineChapter]:
    cleaned_items: list[tuple[list[str], str]] = []
    for path, text in items:
        clean_path = [_clean_structure_title(part) for part in path]
        clean_path = [part for part in clean_path if _valid_structure_title(part)]
        if clean_path:
            cleaned_items.append((clean_path, text))
    if not cleaned_items:
        return []

    chapter_index = _chapter_level_index([path for path, _ in cleaned_items])
    lesson_index = chapter_index + 1
    chapters: dict[str, dict[str, Any]] = {}
    for path, text in cleaned_items:
        if len(path) <= chapter_index:
            continue
        chapter_title = path[chapter_index]
        chapter_key = chapter_title.lower()
        chapter = chapters.setdefault(
            chapter_key,
            {
                "title": chapter_title,
                "description": _first_sentence(text),
                "lessons": {},
            },
        )
        lesson_title = path[lesson_index] if len(path) > lesson_index else ""
        if not lesson_title or lesson_title.lower() == chapter_key:
            continue
        lessons: dict[str, _OutlineLesson] = chapter["lessons"]
        if lesson_title.lower() not in lessons and len(lessons) < MAX_LESSONS_PER_CHAPTER:
            lessons[lesson_title.lower()] = _source_lesson(chapter_title, lesson_title)
        if len(chapters) >= MAX_CHAPTERS:
            break

    return _materialize_source_chapters(chapters)


def _source_structure_from_toc(
    chunks: list[SearchChunkRecord],
    sections: list[CourseSectionRecord],
) -> list[_OutlineChapter]:
    lines = _toc_candidate_lines(chunks, sections)
    return _source_structure_from_toc_lines(lines)


def _source_structure_from_toc_lines(lines: list[str]) -> list[_OutlineChapter]:
    chapters: dict[str, dict[str, Any]] = {}
    current_key = ""
    for line in lines:
        chapter_title = _toc_chapter_title(line)
        if chapter_title:
            current_key = chapter_title.lower()
            chapters.setdefault(
                current_key,
                {
                    "title": chapter_title,
                    "description": "",
                    "lessons": {},
                },
            )
            if len(chapters) >= MAX_CHAPTERS:
                continue
            continue

        if not current_key or current_key not in chapters:
            continue
        lesson_title = _toc_lesson_title(line)
        if not lesson_title:
            continue
        lessons: dict[str, _OutlineLesson] = chapters[current_key]["lessons"]
        if lesson_title.lower() not in lessons and len(lessons) < MAX_LESSONS_PER_CHAPTER:
            lessons[lesson_title.lower()] = _source_lesson(chapters[current_key]["title"], lesson_title)

    return _materialize_source_chapters(chapters)


def _toc_candidate_lines(
    chunks: list[SearchChunkRecord],
    sections: list[CourseSectionRecord],
) -> list[str]:
    texts: list[str] = []
    for section in sections[:20]:
        title = " ".join(section.heading_path or [section.title]).lower()
        if "content" in title or "contents" in title or section.order_index < 3:
            texts.append(section.text)
    for chunk in chunks[:40]:
        heading = " ".join(chunk.heading_path or []).lower()
        if "content" in heading or "contents" in heading or chunk.chunk_index < 20:
            texts.append(chunk.text)

    lines: list[str] = []
    for text in texts:
        for raw_line in str(text or "").splitlines():
            line = _clean_toc_line(raw_line)
            if line and _valid_toc_line(line):
                lines.append(line)
    return lines[:500]


def _toc_chapter_title(line: str) -> str:
    chapter_number = (
        r"(?:\d+|[ivxlcdm]+|one|two|three|four|five|six|seven|eight|nine|ten|"
        r"eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|"
        r"nineteen|twenty)"
    )
    patterns = [
        rf"^(?:chapter\s+)?{chapter_number}\s+(.+?)\s+\d{{1,4}}$",
        rf"^chapter\s+{chapter_number}[:.\-\s]+(.+?)(?:\s+\d{{1,4}})?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, line, flags=re.IGNORECASE)
        if match:
            return _clean_structure_title(match.group(1), strip_page_number=True)
    return ""


def _toc_lesson_title(line: str) -> str:
    match = re.match(r"^(.+?)\s+\.{0,}\s*\d{1,4}$", line)
    if not match:
        return ""
    title = _clean_structure_title(match.group(1), strip_page_number=True)
    return title if _valid_structure_title(title) and not _toc_chapter_title(line) else ""


def _clean_toc_line(value: str) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = text.strip(" .\t")
    return text[:240]


def _valid_toc_line(line: str) -> bool:
    text = line.strip().lower()
    if len(text) < 4 or len(text) > 180:
        return False
    return text not in {
        "contents",
        "table of contents",
        "acknowledgements",
        "bibliography",
        "further reading",
        "notes",
        "index",
        "preface",
        "introduction",
    }


def _chapter_level_index(paths: list[list[str]]) -> int:
    with_second = [path for path in paths if len(path) > 1]
    if not with_second:
        return 0
    first_counts = Counter(path[0].lower() for path in with_second)
    dominant, count = first_counts.most_common(1)[0]
    second_titles = {path[1].lower() for path in with_second if _valid_structure_title(path[1])}
    dominant_title = next((path[0] for path in with_second if path[0].lower() == dominant), "")
    if count / max(1, len(with_second)) >= 0.75 and len(second_titles) >= 2:
        if len(dominant_title) > 48 or _looks_like_file_or_document_title(dominant_title):
            return 1
    return 0


def _looks_like_file_or_document_title(value: str) -> bool:
    text = value.lower()
    return any(token in text for token in (".pdf", ".doc", "course", "book", "textbook"))


def _materialize_source_chapters(chapters: dict[str, dict[str, Any]]) -> list[_OutlineChapter]:
    out: list[_OutlineChapter] = []
    for chapter in list(chapters.values())[:MAX_CHAPTERS]:
        lessons = list(chapter["lessons"].values())
        if not lessons:
            lessons = [_source_lesson(chapter["title"], chapter["title"])]
        out.append(
            _OutlineChapter(
                title=chapter["title"],
                description=chapter["description"] or f"Study {chapter['title']} from the uploaded materials.",
                source_queries=[chapter["title"]],
                lessons=lessons[:MAX_LESSONS_PER_CHAPTER],
            )
        )
    return out


def _source_lesson(chapter_title: str, lesson_title: str) -> _OutlineLesson:
    return _OutlineLesson(
        title=lesson_title,
        learning_objectives=[f"Understand {lesson_title}"],
        source_queries=[chapter_title, lesson_title],
    )


def _clean_structure_title(value: str, *, strip_page_number: bool = False) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip())
    text = re.sub(r"^[\-•*]+\s*", "", text)
    text = re.sub(r"\.{2,}", " ", text)
    if strip_page_number:
        text = re.sub(r"\s+\d{1,4}$", "", text)
    text = re.sub(r"^(?:chapter|part|section)\s+(?:\d+|[ivxlcdm]+)[:.\-\s]+", "", text, flags=re.IGNORECASE)
    return _clean_title(text.strip(" :-"))


def _valid_structure_title(value: str) -> bool:
    text = _clean_title(value)
    lower = text.lower()
    if not text or len(text) < 3 or len(text) > 140:
        return False
    if re.fullmatch(r"(?:\d+|[ivxlcdm]+|page\s+\d+|slide\s+\d+)", lower, flags=re.IGNORECASE):
        return False
    if re.search(r"\.(?:pdf|docx?|pptx?|html?)\b", lower):
        return False
    if _looks_like_markup(text):
        return False
    return lower not in {
        "contents",
        "table of contents",
        "copyright",
        "bibliography",
        "references",
        "index",
        "notes",
        "acknowledgements",
        "chapter",
        "course",
        "course outline",
        "document",
        "lesson",
        "module",
        "section",
        "slides",
    }


def _source_structure_score(chapters: list[_OutlineChapter]) -> int:
    if not chapters:
        return 0
    lesson_count = sum(len(chapter.lessons) for chapter in chapters)
    multi_lesson_bonus = sum(1 for chapter in chapters if len(chapter.lessons) > 1)
    return len(chapters) * 2 + lesson_count + multi_lesson_bonus


def _outline_from_source_structure(
    context_pack: CourseBuilderContextPack,
    chunks: list[SearchChunkRecord],
) -> _CourseOutline | None:
    chapters = context_pack.source_structure or extract_source_structure(chunks, context_pack.sections)
    if not chapters:
        return None
    title = _clean_title(next((doc.title for doc in context_pack.documents if doc.title), "")) or "Generated Course"
    return _CourseOutline(
        title=title,
        description="A structured course generated from the uploaded documents' chapter and sub-chapter structure.",
        learning_objectives=[f"Understand {chapter.title}" for chapter in chapters[:5]],
        chapters=chapters,
    )


def _should_prefer_source_outline(source_outline: _CourseOutline, generated_outline: _CourseOutline) -> bool:
    source_score = _source_structure_score(source_outline.chapters)
    generated_score = _source_structure_score(generated_outline.chapters)
    return source_score >= 3 and source_score >= generated_score


def _merge_outline_metadata(
    generated_outline: _CourseOutline,
    source_outline: _CourseOutline,
) -> _CourseOutline:
    return _CourseOutline(
        title=_clean_title(generated_outline.title) or source_outline.title,
        description=generated_outline.description.strip() or source_outline.description,
        learning_objectives=_clean_list(generated_outline.learning_objectives) or source_outline.learning_objectives,
        prerequisites=_clean_list(generated_outline.prerequisites),
        language=_safe_language(generated_outline.language) or source_outline.language,
        chapters=_merge_source_chapter_titles(source_outline.chapters, generated_outline.chapters),
    )


def _merge_source_chapter_titles(
    source_chapters: list[_OutlineChapter],
    generated_chapters: list[_OutlineChapter],
) -> list[_OutlineChapter]:
    merged: list[_OutlineChapter] = []
    for index, source_chapter in enumerate(source_chapters):
        generated_chapter = generated_chapters[index] if index < len(generated_chapters) else None
        chapter_title = _normalized_source_title(source_chapter.title, generated_chapter.title if generated_chapter else "")
        generated_lessons = generated_chapter.lessons if generated_chapter else []
        lessons: list[_OutlineLesson] = []
        for lesson_index, source_lesson in enumerate(source_chapter.lessons):
            generated_lesson = generated_lessons[lesson_index] if lesson_index < len(generated_lessons) else None
            lesson_title = _normalized_source_title(source_lesson.title, generated_lesson.title if generated_lesson else "")
            lessons.append(
                _OutlineLesson(
                    title=lesson_title,
                    learning_objectives=_clean_list(
                        generated_lesson.learning_objectives if generated_lesson else []
                    ) or source_lesson.learning_objectives,
                    source_queries=_clean_list(
                        [
                            *source_lesson.source_queries,
                            *(generated_lesson.source_queries if generated_lesson else []),
                            source_chapter.title,
                            source_lesson.title,
                        ]
                    ),
                    objective_ids=_clean_list(generated_lesson.objective_ids if generated_lesson else [])
                    or source_lesson.objective_ids,
                )
            )
        merged.append(
            _OutlineChapter(
                title=chapter_title,
                description=(generated_chapter.description.strip() if generated_chapter else "")
                or source_chapter.description,
                phase_id=generated_chapter.phase_id if generated_chapter else source_chapter.phase_id,
                objective_ids=_clean_list(generated_chapter.objective_ids if generated_chapter else [])
                or source_chapter.objective_ids,
                source_queries=_clean_list(
                    [
                        *source_chapter.source_queries,
                        *(generated_chapter.source_queries if generated_chapter else []),
                        source_chapter.title,
                    ]
                ),
                lessons=lessons,
            )
        )
    return merged


def _normalized_source_title(source_title: str, generated_title: str) -> str:
    cleaned_generated = _clean_title(generated_title)
    if cleaned_generated and _title_key(cleaned_generated) == _title_key(source_title):
        return cleaned_generated
    return source_title


def _title_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def _build_rich_summary(
    documents: list[CourseDocumentRecord],
    sections: list[CourseSectionRecord],
    concepts: list[CourseConceptRecord],
    phases: list[CourseLearningPhaseRecord],
    objectives: list[CourseLearningObjectiveRecord],
    representative_chunks: list[SearchChunkRecord],
) -> str:
    doc_labels = _dedupe(
        _clean_title(str(doc.title or doc.source_filename))
        for doc in documents
        if str(doc.title or doc.source_filename).strip()
    )[:12]
    section_summaries = []
    concept_labels: list[str] = []
    equation_count = 0
    table_count = 0
    for section in sections[:80]:
        section_label = " > ".join(section.heading_path or [section.title])
        summary = " ".join(str(section.summary or "").split())[:260]
        if section_label or summary:
            section_summaries.append(f"- {section_label}: {summary}".strip()[:360])
        concept_labels.extend(str(item) for item in section.key_concepts or [])
        equation_count += len(section.equations or [])
        table_count += len(section.tables or [])

    concept_labels.extend(concept.canonical_name for concept in concepts[:50])
    phase_lines = [
        f"- {phase.title}: {phase.summary[:240]}"
        for phase in phases[:12]
    ]
    objective_lines = [
        f"- {objective.objective_text[:220]}"
        for objective in objectives[:40]
    ]
    representative_labels = _dedupe(
        " > ".join(chunk.heading_path or [chunk.source_filename])
        for chunk in representative_chunks[:40]
    )[:24]

    parts = [
        "Uploaded files: " + (", ".join(doc_labels) if doc_labels else "(unknown)"),
        "Main concepts: " + (", ".join(_dedupe(concept_labels)[:30]) if concept_labels else "(not extracted)"),
        f"Structured facts: {len(sections)} sections, {equation_count} equations, {table_count} tables.",
        "Learning phases:\n" + ("\n".join(phase_lines) if phase_lines else "(no learning map available)"),
        "Learning objectives:\n" + ("\n".join(objective_lines) if objective_lines else "(no objectives available)"),
        "Representative paths: " + (", ".join(representative_labels) if representative_labels else "(none)"),
        "Section summaries:\n" + ("\n".join(section_summaries[:30]) if section_summaries else "(none)"),
    ]
    summary = "\n\n".join(parts)
    if len(summary) > 9000:
        summary = summary[:9000].rsplit(" ", 1)[0].strip()
    return summary


def _format_context_pack(context_pack: CourseBuilderContextPack) -> str:
    return "\n\n".join(
        [
            "Rich course summary:\n" + context_pack.rich_summary,
            "Extracted source course structure:\n" + _format_source_structure(context_pack.source_structure),
            "Course plan phases/objectives:\n" + _format_course_plan(context_pack.phases, context_pack.objectives),
            "Canonical concepts:\n" + _format_concepts(context_pack.concepts),
            "Representative chunk catalog:\n" + _chunk_context(context_pack.representative_chunks, max_chars=12000),
        ]
    )


def _format_source_structure(chapters: list[_OutlineChapter]) -> str:
    if not chapters:
        return "(no explicit chapter/sub-chapter structure extracted)"
    lines: list[str] = []
    for chapter_index, chapter in enumerate(chapters[:MAX_CHAPTERS], start=1):
        lines.append(f"- chapter {chapter_index}: {chapter.title}")
        for lesson_index, lesson in enumerate(chapter.lessons[:MAX_LESSONS_PER_CHAPTER], start=1):
            lines.append(f"  - subchapter {chapter_index}.{lesson_index}: {lesson.title}")
    return "\n".join(lines)


def _format_course_plan(
    phases: list[CourseLearningPhaseRecord],
    objectives: list[CourseLearningObjectiveRecord],
) -> str:
    objectives_by_phase: dict[uuid.UUID, list[CourseLearningObjectiveRecord]] = {}
    for objective in objectives:
        objectives_by_phase.setdefault(objective.phase_id, []).append(objective)
    lines: list[str] = []
    for phase in phases[:12]:
        lines.append(f"- phase_id={phase.id} | title={phase.title} | summary={phase.summary[:260]}")
        for objective in objectives_by_phase.get(phase.id, [])[:8]:
            lines.append(
                f"  - objective_id={objective.id} | bloom={objective.bloom_level} | "
                f"objective={objective.objective_text[:260]}"
            )
    return "\n".join(lines) or "(no course plan available)"


def _format_concepts(concepts: list[CourseConceptRecord]) -> str:
    lines = [
        f"- {concept.canonical_name}: {concept.description[:220]}"
        for concept in concepts[:80]
    ]
    return "\n".join(lines) or "(no concept inventory available)"


def _chapter_query(chapter: _OutlineChapter, context_pack: CourseBuilderContextPack) -> str:
    objective_text = _objective_text(chapter.objective_ids, context_pack.objectives)
    phase_text = _phase_text(chapter.phase_id, context_pack.phases)
    return "\n".join(
        item
        for item in [
            chapter.title,
            chapter.description,
            phase_text,
            objective_text,
            " ".join(chapter.source_queries),
        ]
        if item
    )


def _lesson_query(
    chapter: _OutlineChapter,
    lesson: _OutlineLesson,
    context_pack: CourseBuilderContextPack,
) -> str:
    objective_ids = lesson.objective_ids or chapter.objective_ids
    return "\n".join(
        item
        for item in [
            chapter.title,
            chapter.description,
            lesson.title,
            " ".join(lesson.learning_objectives),
            _objective_text(objective_ids, context_pack.objectives),
            " ".join([*chapter.source_queries, *lesson.source_queries]),
        ]
        if item
    )


def _objective_text(
    objective_ids: list[str],
    objectives: list[CourseLearningObjectiveRecord],
) -> str:
    wanted = {str(item) for item in objective_ids}
    if not wanted:
        return ""
    return " ".join(
        objective.objective_text
        for objective in objectives
        if str(objective.id) in wanted
    )


def _phase_text(phase_id: str | None, phases: list[CourseLearningPhaseRecord]) -> str:
    if not phase_id:
        return ""
    for phase in phases:
        if str(phase.id) == str(phase_id):
            return f"{phase.title} {phase.summary}"
    return ""


def _valid_source_chunk_ids(source_chunk_ids: list[str], chunks: list[SearchChunkRecord]) -> list[str]:
    valid = {chunk.id for chunk in chunks}
    return _dedupe(chunk_id for chunk_id in source_chunk_ids if chunk_id in valid)


def _dedupe(values: Any) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text or text in seen:
            continue
        seen.add(text)
        out.append(text)
    return out


def _chunk_context(chunks: list[SearchChunkRecord], *, max_chars: int) -> str:
    parts: list[str] = []
    used = 0
    for chunk in chunks:
        section = " > ".join(chunk.heading_path or [])
        text = " ".join(chunk.text.split())
        block = f"[chunk_id={chunk.id}; source={chunk.source_filename}; section={section}]\n{text}\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts)


def _fallback_outline(chunks: list[SearchChunkRecord]) -> _CourseOutline:
    source_structure = extract_source_structure(chunks)
    if source_structure:
        return _CourseOutline(
            title=_clean_title(source_structure[0].title if source_structure else "Generated Course"),
            description="A structured course generated from the uploaded documents.",
            learning_objectives=[f"Understand {chapter.title}" for chapter in source_structure[:5]],
            chapters=source_structure,
        )

    groups: dict[str, list[SearchChunkRecord]] = {}
    for chunk in chunks:
        key = (chunk.heading_path or [chunk.source_filename])[0] or chunk.source_filename
        groups.setdefault(key, []).append(chunk)
    chapters: list[_OutlineChapter] = []
    for title, group in list(groups.items())[:MAX_CHAPTERS]:
        lesson_titles = []
        seen: set[str] = set()
        for chunk in group:
            path = chunk.heading_path or [title]
            candidate = path[-1] or title
            if candidate.lower() not in seen:
                seen.add(candidate.lower())
                lesson_titles.append(candidate)
            if len(lesson_titles) >= MAX_LESSONS_PER_CHAPTER:
                break
        if not lesson_titles:
            lesson_titles = [title]
        chapters.append(
            _OutlineChapter(
                title=title,
                description=_first_sentence(group[0].text),
                lessons=[
                    _OutlineLesson(
                        title=lesson_title,
                        learning_objectives=[f"Understand {lesson_title}"],
                        source_queries=[lesson_title],
                    )
                    for lesson_title in lesson_titles
                ],
            )
        )
    course_title = _clean_title(chapters[0].title if chapters else "Generated Course")
    return _CourseOutline(
        title=course_title,
        description="A structured course generated from the uploaded documents.",
        learning_objectives=[f"Understand {chapter.title}" for chapter in chapters[:5]],
        chapters=chapters,
    )


def _usable_chapters(outline: _CourseOutline, chunks: list[SearchChunkRecord]) -> list[_OutlineChapter]:
    chapters = [
        chapter
        for chapter in outline.chapters
        if _clean_title(chapter.title) and not _looks_like_markup(chapter.title)
    ][:MAX_CHAPTERS]
    if chapters:
        return chapters
    return _fallback_outline(chunks).chapters


def _usable_lessons(chapter: _OutlineChapter, chunks: list[SearchChunkRecord]) -> list[_OutlineLesson]:
    lessons = [
        lesson
        for lesson in chapter.lessons
        if _clean_title(lesson.title) and not _looks_like_markup(lesson.title)
    ][:MAX_LESSONS_PER_CHAPTER]
    if lessons:
        return lessons
    return [
        _OutlineLesson(
            title=chapter.title,
            learning_objectives=[f"Understand {chapter.title}"],
            source_queries=[chapter.title],
        )
    ]


def _looks_like_markup(value: str) -> bool:
    text = str(value or "").strip().lower()
    return any(token in text for token in ("</", "<td", "<th", "<tr", "\\begin{pmatrix}"))


def _fallback_lesson_blocks(title: str, chunks: list[SearchChunkRecord]) -> list[_LessonBlockCandidate]:
    if not chunks:
        return [
            _LessonBlockCandidate(
                block_type="warning",
                title="Insufficient source material",
                content=insufficient_source_message(),
            )
        ]
    first = chunks[0]
    summary = _summarize_text(first.text)
    return [
        _LessonBlockCandidate(
            block_type="explanation",
            title=title,
            content=summary,
            source_chunk_ids=[first.id],
        ),
        _LessonBlockCandidate(
            block_type="example",
            title="Source-grounded example",
            content=_first_sentence(first.text) or summary,
            source_chunk_ids=[first.id],
        ),
        _LessonBlockCandidate(
            block_type="summary",
            title="Key takeaway",
            content=_first_sentence(summary) or summary,
            source_chunk_ids=[first.id],
        ),
    ]


def _fallback_block_plans(title: str, chunks: list[SearchChunkRecord]) -> list[_LessonBlockPlan]:
    if not chunks:
        return []
    first = chunks[0]
    section = " > ".join(first.heading_path or [first.source_filename])
    return [
        _LessonBlockPlan(
            block_type="explanation",
            title=title,
            source_query=f"{title} {section}",
        ),
        _LessonBlockPlan(
            block_type="example",
            title="Source-grounded example",
            source_query=f"example {title} {section}",
        ),
        _LessonBlockPlan(
            block_type="summary",
            title="Key takeaway",
            source_query=f"summary {title} {section}",
        ),
    ]


def _fallback_questions(
    chapter: CourseBuilderChapterRecord,
    chunks: list[SearchChunkRecord],
) -> list[_QuizQuestionCandidate]:
    questions: list[_QuizQuestionCandidate] = []
    snippets = [_first_sentence(chunk.text) for chunk in chunks if _first_sentence(chunk.text)]
    if not snippets:
        snippets = [chapter.description or chapter.title]
    for index, snippet in enumerate(snippets[:5]):
        options = [
            snippet[:180],
            "A point not supported by the uploaded documents.",
            "A summary from an unrelated chapter.",
            "An invented claim with no cited source.",
        ]
        questions.append(
            _QuizQuestionCandidate(
                prompt=f"Which statement is supported by the sources for {chapter.title}?",
                options=options,
                correct_index=0,
                explanation="The correct option is the one directly drawn from the cited source chunk.",
                source_chunk_ids=[chunks[index].id] if index < len(chunks) else [],
            )
        )
    return questions


def _valid_quiz_question_rows(
    questions: list[_QuizQuestionCandidate],
    chunks: list[SearchChunkRecord],
    rag: Any,
) -> list[tuple[_QuizQuestionCandidate, list[str], int, list[dict]]]:
    rows: list[tuple[_QuizQuestionCandidate, list[str], int, list[dict]]] = []
    for candidate in questions:
        prompt = str(candidate.prompt or "").strip()
        options = [str(option).strip() for option in candidate.options if str(option).strip()]
        if not prompt or len(options) < 2:
            continue
        source_chunk_ids = _valid_source_chunk_ids(candidate.source_chunk_ids, chunks)
        if not source_chunk_ids:
            continue
        citations = rag.citations_for(chunks, source_chunk_ids)
        if not citations:
            continue
        correct_index = _safe_correct_index(candidate.correct_index, len(options))
        rows.append(
            (
                _QuizQuestionCandidate(
                    prompt=prompt,
                    options=options,
                    correct_index=correct_index,
                    explanation=candidate.explanation.strip(),
                    source_chunk_ids=source_chunk_ids,
                ),
                options,
                correct_index,
                citations,
            )
        )
        if len(rows) >= MAX_QUIZ_QUESTIONS:
            break
    return rows


def _safe_correct_index(value: Any, option_count: int) -> int:
    try:
        index = int(value or 0)
    except (TypeError, ValueError):
        index = 0
    return min(max(index, 0), max(0, option_count - 1))


def _chunks_by_ids(chunks: list[SearchChunkRecord], chunk_ids: list[str]) -> list[SearchChunkRecord]:
    by_id = {chunk.id: chunk for chunk in chunks}
    return [by_id[chunk_id] for chunk_id in chunk_ids if chunk_id in by_id]


def _first_sentence(text: str) -> str:
    clean = " ".join(str(text or "").split())
    match = re.split(r"(?<=[.!?])\s+", clean, maxsplit=1)
    return (match[0] if match else clean)[:500]


def _summarize_text(text: str) -> str:
    clean = " ".join(str(text or "").split())
    return clean[:900]


def _selected_index(answer: str | int | None, options: list[str]) -> int | None:
    if isinstance(answer, int):
        return answer if 0 <= answer < len(options) else None
    text = str(answer or "").strip()
    if not text:
        return None
    try:
        numeric = int(text)
        if 0 <= numeric < len(options):
            return numeric
    except ValueError:
        pass
    lowered = text.lower()
    for index, option in enumerate(options):
        if lowered == str(option).strip().lower():
            return index
    if len(text) == 1 and text.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        index = ord(text.upper()) - ord("A")
        return index if 0 <= index < len(options) else None
    return None


_service: CourseBuilderService | None = None


def get_coursebuilder_service() -> CourseBuilderService:
    global _service
    if _service is None:
        _service = CourseBuilderService()
    return _service
