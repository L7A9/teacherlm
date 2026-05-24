from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from teacherlm_core.llm.ollama_client import OllamaClient
from teacherlm_core.llm.runtime import build_llm_client_kwargs, has_llm_override

from config import Settings, get_settings
from db.models import (
    CourseBuilderChapterAttemptRecord,
    CourseBuilderChapterRecord,
    CourseBuilderCourseRecord,
    CourseBuilderLessonBlockRecord,
    CourseBuilderLessonRecord,
    CourseBuilderProgressEventRecord,
    CourseBuilderQuizQuestionRecord,
    CourseBuilderQuizRecord,
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


logger = logging.getLogger(__name__)

PASS_SCORE = 0.7
LOCAL_FALLBACK_MODEL = "gemma4:e2b"
MAX_CHAPTERS = 10
MAX_LESSONS_PER_CHAPTER = 5
MAX_BLOCKS_PER_LESSON = 5
MAX_QUIZ_QUESTIONS = 7


class _OutlineLesson(BaseModel):
    title: str
    learning_objectives: list[str] = Field(default_factory=list)
    source_queries: list[str] = Field(default_factory=list)


class _OutlineChapter(BaseModel):
    title: str
    description: str = ""
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


class _QuizQuestionCandidate(BaseModel):
    prompt: str
    options: list[str] = Field(default_factory=list)
    correct_index: int = 0
    explanation: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)


class _QuizCandidate(BaseModel):
    questions: list[_QuizQuestionCandidate] = Field(default_factory=list)


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
            return existing

        course_id = _stable_id(conversation_id, "coursebuilder-course")
        course = await session.get(CourseBuilderCourseRecord, course_id)
        if course is None:
            course = CourseBuilderCourseRecord(
                id=course_id,
                conversation_id=conversation_id,
                title="",
                status="queued",
                generation_metadata={},
            )
            session.add(course)
        else:
            course.status = "queued"
            course.error = None
            course.generation_metadata = {
                **(course.generation_metadata or {}),
                "queued_at": datetime.now(timezone.utc).isoformat(),
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
            chunks = await self._rag.load_chunks(session, conversation_id)
            if not chunks:
                raise ValueError("No processed chunks were found for this conversation.")

            course.status = "generating_outline"
            await self.record_event(
                session,
                conversation_id,
                course_id=course.id,
                stage="generating_outline",
                message="Designing the course outline.",
                percent=18,
            )
            outline = await self._outline(conversation_id, chunks, llm_options=llm_options)
            course.title = _clean_title(outline.title) or "Generated Course"
            course.description = outline.description.strip()
            course.learning_objectives = _clean_list(outline.learning_objectives)
            course.prerequisites = _clean_list(outline.prerequisites)
            course.language = _language_from_options(llm_options) or _safe_language(outline.language)
            course.generation_metadata = {
                **(course.generation_metadata or {}),
                "chunk_count": len(chunks),
                "chapter_count": len(outline.chapters),
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

            chapters = _usable_chapters(outline, chunks)
            all_chapters: list[CourseBuilderChapterRecord] = []
            for chapter_index, chapter_candidate in enumerate(chapters):
                chapter_id = _stable_id(conversation_id, f"chapter:{chapter_index}:{chapter_candidate.title}")
                chapter_chunks = await self._rag.retrieve_lesson_chunks(
                    session,
                    conversation_id,
                    f"{chapter_candidate.title}\n{chapter_candidate.description}",
                    fallback_chunks=chunks,
                    top_k=10,
                )
                chapter = CourseBuilderChapterRecord(
                    id=chapter_id,
                    course_id=course.id,
                    conversation_id=conversation_id,
                    title=_clean_title(chapter_candidate.title) or f"Chapter {chapter_index + 1}",
                    description=chapter_candidate.description.strip(),
                    order_index=chapter_index,
                    summary=chapter_candidate.description.strip(),
                    source_chunk_ids=[chunk.id for chunk in chapter_chunks],
                    is_locked=chapter_index != 0,
                    unlock_rule={
                        "type": "previous_chapter_quiz",
                        "pass_score": PASS_SCORE,
                        "strict": True,
                    },
                )
                session.add(chapter)
                all_chapters.append(chapter)
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
            for chapter_index, chapter in enumerate(all_chapters):
                chapter_candidate = chapters[chapter_index]
                lesson_candidates = _usable_lessons(chapter_candidate, chunks)
                for lesson_index, lesson_candidate in enumerate(lesson_candidates):
                    lesson_chunks = await self._rag.retrieve_lesson_chunks(
                        session,
                        conversation_id,
                        "\n".join(
                            [
                                chapter.title,
                                lesson_candidate.title,
                                " ".join(lesson_candidate.learning_objectives),
                                " ".join(lesson_candidate.source_queries),
                            ]
                        ),
                        fallback_chunks=chunks,
                        top_k=8,
                    )
                    lesson_id = _stable_id(
                        conversation_id,
                        f"lesson:{chapter_index}:{lesson_index}:{lesson_candidate.title}",
                    )
                    lesson_content = await self._lesson_content(
                        chapter,
                        lesson_candidate,
                        lesson_chunks,
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
                    for block_index, block_candidate in enumerate(blocks):
                        block_citations = self._rag.citations_for(
                            lesson_chunks,
                            block_candidate.source_chunk_ids,
                        )
                        block_type = normalize_block_type(block_candidate.block_type)
                        data_json = block_candidate.data_json or {}
                        if block_type == "chart":
                            data_json = validate_chart_spec(data_json)
                        content = block_candidate.content.strip()
                        validation_status = "supported" if block_citations else "insufficient_source_material"
                        if not block_citations:
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

            course.status = "generating_quizzes"
            await self.record_event(
                session,
                conversation_id,
                course_id=course.id,
                stage="generating_quizzes",
                message="Building chapter quizzes.",
                percent=75,
            )
            for chapter in all_chapters:
                chapter_chunks = _chunks_by_ids(chunks, chapter.source_chunk_ids) or chunks[:8]
                quiz_id = _stable_id(conversation_id, f"quiz:{chapter.order_index}:{chapter.title}")
                quiz_candidate = await self._quiz(chapter, chapter_chunks, llm_options=llm_options)
                questions = quiz_candidate.questions[:MAX_QUIZ_QUESTIONS] or _fallback_questions(chapter, chapter_chunks)
                quiz = CourseBuilderQuizRecord(
                    id=quiz_id,
                    chapter_id=chapter.id,
                    course_id=course.id,
                    pass_score=PASS_SCORE,
                    question_count=len(questions),
                    source_chunk_ids=[chunk.id for chunk in chapter_chunks],
                )
                session.add(quiz)
                await session.flush()
                for question_index, candidate in enumerate(questions):
                    options = [str(option).strip() for option in candidate.options if str(option).strip()]
                    if len(options) < 2:
                        continue
                    correct_index = min(max(int(candidate.correct_index or 0), 0), len(options) - 1)
                    citations = self._rag.citations_for(chapter_chunks, candidate.source_chunk_ids)
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
            await self.record_event(
                session,
                conversation_id,
                course_id=course.id,
                stage="validating",
                message="Validating source support and unlock rules.",
                percent=90,
            )
            await session.flush()
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
            await session.flush()
            return await self.get_course(session, conversation_id)
        except Exception as exc:
            logger.exception("CourseBuilder generation failed for conversation %s", conversation_id)
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
            await session.flush()
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
        if chapter.is_locked:
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

    async def _outline(
        self,
        conversation_id: uuid.UUID,
        chunks: list[SearchChunkRecord],
        *,
        llm_options: dict[str, Any] | None,
    ) -> _CourseOutline:
        sample = _chunk_context(chunks, max_chars=15000)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are CourseBuilder for TeacherLM. Create a source-grounded, "
                    "ordered online course outline from uploaded course chunks only. "
                    "Start with fundamentals and end with advanced/final topics. "
                    "Do not invent unsupported chapters. Use the requested language if one is configured."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Build a concise course outline. Return 3-10 ordered chapters. "
                    "Each chapter must contain 2-5 lessons.\n\n"
                    f"Conversation: {conversation_id}\n\nSource chunks:\n{sample}"
                ),
            },
        ]
        try:
            return await self._structured(messages, _CourseOutline, llm_options=llm_options)
        except Exception:
            logger.exception("CourseBuilder outline LLM failed; using fallback")
            return _fallback_outline(chunks)

    async def _lesson_content(
        self,
        chapter: CourseBuilderChapterRecord,
        lesson: _OutlineLesson,
        chunks: list[SearchChunkRecord],
        *,
        llm_options: dict[str, Any] | None,
    ) -> _LessonContent:
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
            chapter.is_locked = index != 0

    async def _events(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        course_id: uuid.UUID | None,
    ) -> list[CourseBuilderProgressEventRead]:
        stmt = (
            select(CourseBuilderProgressEventRecord)
            .where(CourseBuilderProgressEventRecord.conversation_id == conversation_id)
            .order_by(CourseBuilderProgressEventRecord.created_at.asc())
            .limit(80)
        )
        if course_id:
            stmt = stmt.where(
                (CourseBuilderProgressEventRecord.course_id == course_id)
                | (CourseBuilderProgressEventRecord.course_id.is_(None))
            )
        result = await session.execute(stmt)
        return [_event_read(record) for record in result.scalars().all()]

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
            strict_locked = not prior_completed
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
