from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from teacherlm_core.llm.providers import LLMMessage, complete_text
from teacherlm_core.schemas.generator_io import LearnerUpdates

from local_api.db import get_store, new_id, utc_now
from local_api.services.knowledge_graph import get_knowledge_graph_service
from local_api.services.learner import get_learner_service
from local_api.services.settings import get_settings_service


MAX_CHAPTERS = 12
MAX_LESSONS_PER_CHAPTER = 8
MAX_BLOCKS_PER_LESSON = 9
PASS_SCORE = 0.70
COURSEBUILDER_VERSION = "local-coursebuilder-v4-persisted-plan"
COURSE_PLAN_CONTRACT_VERSION = "1.0.0"
STRUCTURE_PENDING_STATUSES = {"uploaded", "parsing", "chunking", "extracting_concepts"}

PedagogicalRole = Literal[
    "definition",
    "context",
    "foundation",
    "core_concept",
    "structure",
    "law",
    "chronology",
    "cause",
    "event",
    "consequence",
    "interpretation",
    "mechanism",
    "mathematical_formulation",
    "derivation",
    "standard_method",
    "specialized_method",
    "reaction",
    "procedure",
    "experiment",
    "integration",
    "advanced",
    "application",
    "evaluation",
    "safety",
    "synthesis",
]

ArchitectureType = Literal[
    "conceptual",
    "historical",
    "chemistry",
    "physics",
    "mathematics",
    "life_science",
    "procedural",
    "mixed",
]

ARCHITECTURE_ROLE_ORDER: dict[str, tuple[str, ...]] = {
    "conceptual": (
        "definition", "context", "foundation", "core_concept", "structure", "mechanism",
        "standard_method", "mathematical_formulation", "specialized_method", "integration",
        "advanced", "application", "evaluation", "synthesis",
    ),
    "historical": (
        "definition", "context", "cause", "chronology", "event", "consequence",
        "interpretation", "synthesis", "evaluation",
    ),
    "chemistry": (
        "definition", "foundation", "structure", "law", "core_concept", "reaction",
        "mechanism", "mathematical_formulation", "procedure", "experiment", "application",
        "safety", "evaluation", "synthesis",
    ),
    "physics": (
        "definition", "foundation", "mathematical_formulation", "law", "core_concept",
        "derivation", "mechanism", "experiment", "application", "evaluation", "synthesis",
    ),
    "mathematics": (
        "definition", "foundation", "structure", "core_concept", "law", "derivation",
        "standard_method", "specialized_method", "application", "evaluation", "synthesis",
    ),
    "life_science": (
        "definition", "foundation", "structure", "core_concept", "mechanism", "procedure",
        "experiment", "application", "evaluation", "synthesis",
    ),
    "procedural": (
        "definition", "context", "foundation", "procedure", "standard_method",
        "specialized_method", "application", "evaluation", "safety", "synthesis",
    ),
    "mixed": (
        "definition", "context", "foundation", "core_concept", "structure", "law", "chronology",
        "mechanism", "mathematical_formulation", "standard_method", "specialized_method",
        "integration", "advanced", "procedure", "experiment", "application", "evaluation",
        "safety", "synthesis",
    ),
}


class OutlineLesson(BaseModel):
    title: str = Field(min_length=2, max_length=140)
    summary: str = Field(default="", max_length=500)
    learning_objectives: list[str] = Field(default_factory=list, max_length=4)
    source_chunk_ids: list[str] = Field(default_factory=list, min_length=1)
    pedagogical_role: PedagogicalRole = "core_concept"
    sequencing_reason: str = Field(default="", max_length=400)
    prerequisite_lesson_titles: list[str] = Field(default_factory=list, max_length=8)


class OutlineChapter(BaseModel):
    title: str = Field(min_length=2, max_length=140)
    description: str = Field(default="", max_length=700)
    learning_objectives: list[str] = Field(default_factory=list, max_length=6)
    source_chunk_ids: list[str] = Field(default_factory=list, min_length=1)
    pedagogical_role: PedagogicalRole = "core_concept"
    sequencing_reason: str = Field(default="", max_length=500)
    prerequisite_chapter_titles: list[str] = Field(default_factory=list, max_length=8)
    lessons: list[OutlineLesson] = Field(min_length=1, max_length=MAX_LESSONS_PER_CHAPTER)


class CourseOutline(BaseModel):
    title: str = Field(min_length=2, max_length=180)
    description: str = Field(default="", max_length=900)
    learning_objectives: list[str] = Field(default_factory=list, max_length=10)
    architecture_type: ArchitectureType = "conceptual"
    architecture_rationale: str = Field(default="", max_length=700)
    chapters: list[OutlineChapter] = Field(min_length=1, max_length=MAX_CHAPTERS)


class DraftBlock(BaseModel):
    block_type: Literal["markdown", "example", "summary"] = "markdown"
    title: str = Field(default="", max_length=140)
    content: str = Field(min_length=20, max_length=7000)
    source_chunk_ids: list[str] = Field(default_factory=list, min_length=1)


class DraftLesson(BaseModel):
    title: str = Field(min_length=2, max_length=140)
    summary: str = Field(default="", max_length=700)
    learning_objectives: list[str] = Field(default_factory=list, max_length=5)
    source_chunk_ids: list[str] = Field(default_factory=list, min_length=1)
    blocks: list[DraftBlock] = Field(min_length=1, max_length=6)


class ChapterDraft(BaseModel):
    summary: str = Field(default="", max_length=1200)
    lessons: list[DraftLesson] = Field(min_length=1, max_length=MAX_LESSONS_PER_CHAPTER)


class QuizDraftQuestion(BaseModel):
    prompt: str = Field(min_length=12, max_length=500)
    options: list[str] = Field(min_length=4, max_length=4)
    correct_index: int = Field(ge=0, le=3)
    explanation: str = Field(min_length=10, max_length=700)
    source_chunk_id: str


class QuizDraft(BaseModel):
    questions: list[QuizDraftQuestion] = Field(min_length=1, max_length=30)


class LocalCourseBuilderService:
    """Grounded, staged course synthesis with a deterministic recovery path."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._plan_locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task[Any]] = set()

    def invalidate_plan(self, conversation_id: str, reason: str) -> None:
        """Make a structural plan unusable as soon as its source set changes."""
        get_store().execute(
            """
            UPDATE coursebuilder_plans
            SET status = 'invalid', error = ?, updated_at = ?
            WHERE conversation_id = ?
            """,
            (reason[:500], utc_now(), conversation_id),
        )

    def get_plan(self, conversation_id: str) -> dict[str, Any]:
        row = self._plan_row(conversation_id)
        if row is None:
            return {"status": "empty", "chapters": []}
        payload = json.loads(row["payload_json"])
        payload["status"] = row.get("status", payload.get("status", "draft"))
        if row.get("error"):
            payload["error"] = row["error"]
        outline = payload.get("outline", {})
        return {
            **payload,
            "chapters": [
                {
                    **{key: value for key, value in chapter.items() if key != "lessons"},
                    "subchapters": chapter.get("lessons", []),
                }
                for chapter in outline.get("chapters", [])
            ],
        }

    async def prepare_plan_async(self, conversation_id: str, *, force: bool = False) -> dict[str, Any]:
        """Prepare the chapters/subchapters after parsing and before embeddings."""
        while True:
            files = get_store().list_files(conversation_id)
            if not files:
                return {"status": "empty", "chapters": []}
            if not any(file.get("status") in STRUCTURE_PENDING_STATUSES for file in files):
                break
            await asyncio.sleep(0.05)

        lock = self._plan_locks.setdefault(conversation_id, asyncio.Lock())
        async with lock:
            files = get_store().list_files(conversation_id)
            chunks = get_store().list_chunks(conversation_id)
            if not chunks:
                return {"status": "empty", "chapters": []}
            _sort_chunks_by_file_order(files, chunks)
            fingerprint = _source_fingerprint(files, chunks)
            existing = self._plan_row(conversation_id)
            if (
                not force
                and existing is not None
                and existing.get("source_fingerprint") == fingerprint
                and existing.get("status") in {"draft", "validated"}
            ):
                return json.loads(existing["payload_json"])

            plan_id = new_id("course_plan")
            fallback = _build_fallback_course(conversation_id, chunks, fingerprint)
            planning_payload = {
                "id": f"courseplan_{conversation_id}",
                "plan_id": plan_id,
                "conversation_id": conversation_id,
                "contract_version": COURSE_PLAN_CONTRACT_VERSION,
                "source_fingerprint": fingerprint,
                "status": "planning",
                "outline": {},
                "metadata": {
                    "stage": "pre_embedding_planning",
                    "chunk_count": len(chunks),
                    "source_file_count": len({chunk["source_file_id"] for chunk in chunks}),
                },
            }
            self._save_plan(planning_payload, quality_mode="llm")
            provider = get_settings_service().get_default_chat_provider_config()
            quality_mode = "fallback"
            error: str | None = None
            if provider is None:
                outline = _outline_from_course(fallback)
            else:
                try:
                    outline = await _build_outline_with_llm(provider, chunks, fallback)
                    quality_mode = "llm"
                except Exception as exc:  # noqa: BLE001 - deterministic planning is the recovery boundary.
                    outline = _outline_from_course(fallback)
                    error = str(exc)[:500]
            outline = _ensure_outline_coverage(outline, chunks, None)
            outline = _sequence_outline(outline)
            plan_payload = {
                **planning_payload,
                "status": "draft",
                "title": outline.title,
                "architecture_type": outline.architecture_type,
                "outline": outline.model_dump(mode="json"),
                "metadata": {
                    **planning_payload["metadata"],
                    "stage": "draft_ready_before_embedding",
                    "quality_mode": quality_mode,
                    **_coverage_metadata(outline, chunks, None),
                },
            }
            self._save_plan(plan_payload, quality_mode=quality_mode, error=error)
            return plan_payload

    async def replan_and_rebuild_async(self, conversation_id: str) -> dict[str, Any]:
        await self.prepare_plan_async(conversation_id, force=True)
        return await self.rebuild_async(conversation_id, force=True)

    def get_or_build(self, conversation_id: str) -> dict[str, Any]:
        files = get_store().list_files(conversation_id)
        if not files:
            return {"chapters": [], "status": "empty", "files_total": 0, "files_pending": 0}
        pending = [file for file in files if file["status"] != "ready"]
        if pending:
            response = {
                "chapters": [],
                "status": "waiting_for_files",
                "files_total": len(files),
                "files_pending": len(pending),
                "files_failed": sum(file["status"] == "failed" for file in pending),
            }
            plan = self.get_plan(conversation_id)
            if plan.get("status") in {"draft", "validated"}:
                response["course_plan"] = plan
                response["metadata"] = {"stage": plan.get("metadata", {}).get("stage", "planning")}
            return response

        chunks = get_store().list_chunks(conversation_id)
        if not chunks:
            return {"chapters": [], "status": "empty", "files_total": len(files), "files_pending": 0}
        fingerprint = _source_fingerprint(files, chunks)
        row = self._course_row(conversation_id)
        if row is not None and row.get("source_fingerprint") == fingerprint:
            return self._public_course(json.loads(row["payload_json"]))

        # A GET never performs model work. It creates a useful fallback immediately,
        # while the ingestion/rebuild path can replace it with LLM synthesis.
        return self.rebuild(conversation_id)

    def rebuild(self, conversation_id: str) -> dict[str, Any]:
        ready = self._ready_material(conversation_id)
        if isinstance(ready, dict):
            return ready
        files, chunks, fingerprint = ready
        graph = _course_graph(conversation_id)
        build_id = new_id("course_build")
        plan = self._plan_row(conversation_id)
        if plan is not None and plan.get("source_fingerprint") == fingerprint and plan.get("status") in {"draft", "validated"}:
            try:
                plan_payload = json.loads(plan["payload_json"])
                outline = _validate_plan_with_graph(
                    CourseOutline.model_validate(plan_payload.get("outline")),
                    chunks,
                    graph,
                )
                payload = _build_course_from_outline_fallback(
                    conversation_id,
                    chunks,
                    fingerprint,
                    outline,
                    build_id,
                    graph,
                )
                payload["metadata"]["plan_id"] = plan_payload.get("plan_id")
            except (ValidationError, ValueError, TypeError, json.JSONDecodeError):
                payload = _build_fallback_course(conversation_id, chunks, fingerprint, graph=graph)
        else:
            payload = _build_fallback_course(conversation_id, chunks, fingerprint, graph=graph)
        self._save_course(payload, build_id=build_id, quality_mode="fallback")
        self._reconcile_progress(payload)
        return self._public_course(payload)

    async def rebuild_async(self, conversation_id: str, *, force: bool = False) -> dict[str, Any]:
        lock = self._locks.setdefault(conversation_id, asyncio.Lock())
        async with lock:
            ready = self._ready_material(conversation_id)
            if isinstance(ready, dict):
                return ready
            files, chunks, fingerprint = ready
            graph = _course_graph(conversation_id)
            existing = self._course_row(conversation_id)
            if (
                not force
                and existing is not None
                and existing.get("source_fingerprint") == fingerprint
                and existing.get("status") == "ready"
                and existing.get("quality_mode") in {"llm", "mixed"}
            ):
                return self._public_course(json.loads(existing["payload_json"]))

            build_id = new_id("course_build")
            job_id = new_id("job")
            self._save_job(job_id, "running", conversation_id, build_id, fingerprint, "validating_plan")
            fallback = _build_fallback_course(conversation_id, chunks, fingerprint, graph=graph)
            plan_payload = await self.prepare_plan_async(conversation_id)
            try:
                if plan_payload.get("source_fingerprint") != fingerprint:
                    raise ValueError("saved course plan does not match the ready source set")
                outline = CourseOutline.model_validate(plan_payload.get("outline"))
                outline_mode = str(plan_payload.get("metadata", {}).get("quality_mode") or "fallback")
            except (ValidationError, ValueError, TypeError):
                outline = _outline_from_course(fallback)
                outline_mode = "fallback"
            outline = _validate_plan_with_graph(outline, chunks, graph)
            validated_plan = {
                **plan_payload,
                "status": "validated",
                "outline": outline.model_dump(mode="json"),
                "metadata": {
                    **plan_payload.get("metadata", {}),
                    "stage": "validated_with_knowledge_graph",
                    **_coverage_metadata(outline, chunks, graph),
                },
            }
            self._save_plan(validated_plan, quality_mode=outline_mode)
            provider = get_settings_service().get_default_chat_provider_config()
            if provider is None:
                planned_fallback = _build_course_from_outline_fallback(
                    conversation_id,
                    chunks,
                    fingerprint,
                    outline,
                    build_id,
                    graph,
                )
                planned_fallback["metadata"]["warnings"] = ["No chat model is configured; source-extracted course used."]
                planned_fallback["metadata"]["plan_id"] = validated_plan.get("plan_id")
                self._save_course(planned_fallback, build_id=build_id, quality_mode="fallback")
                self._reconcile_progress(planned_fallback)
                self._save_job(job_id, "completed", conversation_id, build_id, fingerprint, "complete")
                return self._public_course(planned_fallback)

            planning_payload = _course_shell(conversation_id, fingerprint, outline, build_id)
            planning_payload["metadata"].update(
                {
                    "stage": "using_validated_plan",
                    "ready_chapter_count": 0,
                    "total_chapter_count": len(outline.chapters),
                    "quality_mode": outline_mode,
                    "plan_id": validated_plan.get("plan_id"),
                    "plan_contract_version": validated_plan.get("contract_version"),
                }
            )
            self._save_course(planning_payload, build_id=build_id, quality_mode=outline_mode)

            try:
                payload = _course_shell(conversation_id, fingerprint, outline, build_id)
                payload["metadata"].update(_coverage_metadata(outline, chunks, graph))
                payload["metadata"]["plan_id"] = validated_plan.get("plan_id")
                payload["metadata"]["plan_contract_version"] = validated_plan.get("contract_version")
                self._save_course(payload, build_id=build_id, quality_mode=outline_mode)
                ready_chapters: list[dict[str, Any]] = []
                fallback_chapters = 0
                previous_summary = ""
                fallback_by_sources = _fallback_chapters_by_sources(fallback)
                for index, chapter_outline in enumerate(outline.chapters):
                    self._save_job(
                        job_id,
                        "running",
                        conversation_id,
                        build_id,
                        fingerprint,
                        "generating_chapter",
                        {"chapter_index": index, "chapter_total": len(outline.chapters)},
                    )
                    try:
                        chapter = await _build_chapter_with_llm(
                            provider,
                            conversation_id,
                            index,
                            chapter_outline,
                            chunks,
                            previous_summary,
                        )
                    except Exception:  # noqa: BLE001
                        chapter = _fallback_chapter_for_outline(
                            conversation_id,
                            index,
                            chapter_outline,
                            chunks,
                            fallback_by_sources,
                        )
                        fallback_chapters += 1
                    chapter["prerequisite_chapter_ids"] = [ready_chapters[-1]["id"]] if ready_chapters else []
                    ready_chapters.append(chapter)
                    previous_summary = chapter.get("summary", "")
                    payload["chapters"] = ready_chapters
                    payload["metadata"].update(
                        {
                            "stage": "generating_chapter",
                            "ready_chapter_count": len(ready_chapters),
                            "total_chapter_count": len(outline.chapters),
                        }
                    )
                    self._save_course(
                        payload,
                        build_id=build_id,
                        quality_mode="mixed" if fallback_chapters or outline_mode == "fallback" else "llm",
                    )

                final_count = min(30, max(10, len(ready_chapters) * 2))
                payload["metadata"]["stage"] = "generating_final_quiz"
                self._save_course(payload, build_id=build_id, quality_mode="mixed" if fallback_chapters else outline_mode)
                payload["final_quiz"] = await _build_quiz_with_llm(
                    provider,
                    title=f"{outline.title} final assessment",
                    chunks=chunks,
                    count=final_count,
                    scope="course",
                )
                payload["status"] = "ready"
                payload["metadata"].update(
                    {
                        "stage": "complete",
                        "ready_chapter_count": len(ready_chapters),
                        "quality_mode": "mixed" if fallback_chapters or outline_mode == "fallback" else "llm",
                        "fallback_chapter_count": fallback_chapters,
                    }
                )
                quality_mode = payload["metadata"]["quality_mode"]
                self._save_course(payload, build_id=build_id, quality_mode=quality_mode)
                self._reconcile_progress(payload)
                self._save_job(job_id, "completed", conversation_id, build_id, fingerprint, "complete")
                return self._public_course(payload)
            except Exception as exc:  # noqa: BLE001
                planned_fallback = _build_course_from_outline_fallback(
                    conversation_id,
                    chunks,
                    fingerprint,
                    outline,
                    build_id,
                    graph,
                )
                planned_fallback["metadata"]["plan_id"] = validated_plan.get("plan_id")
                planned_fallback["metadata"]["warnings"] = [
                    "Course synthesis failed validation; a grounded source-extracted course was kept.",
                    str(exc)[:300],
                ]
                self._save_course(planned_fallback, build_id=build_id, quality_mode="fallback", error=str(exc))
                self._reconcile_progress(planned_fallback)
                self._save_job(job_id, "failed", conversation_id, build_id, fingerprint, "failed", error=str(exc))
                return self._public_course(planned_fallback)

    def schedule_rebuild(self, conversation_id: str, *, force: bool = False) -> None:
        task = asyncio.create_task(self.rebuild_async(conversation_id, force=force))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def resume_incomplete_builds(self) -> None:
        rows = get_store().query("SELECT DISTINCT conversation_id FROM uploaded_files")
        for row in rows:
            conversation_id = row["conversation_id"]
            files = get_store().list_files(conversation_id)
            if not files or any(file["status"] != "ready" for file in files):
                continue
            chunks = get_store().list_chunks(conversation_id)
            if not chunks:
                continue
            current = self._course_row(conversation_id)
            fingerprint = _source_fingerprint(files, chunks)
            if current is None or current.get("status") != "ready" or current.get("source_fingerprint") != fingerprint:
                self.schedule_rebuild(conversation_id)

    def mark_lesson_complete(self, conversation_id: str, lesson_id: str) -> dict[str, Any]:
        payload = self._private_course(conversation_id)
        public = self._public_course(payload)
        lesson = _find_lesson(public, lesson_id)
        if lesson is None:
            raise KeyError("lesson not found")
        if lesson.get("is_locked") or lesson.get("generation_status") != "ready":
            raise PermissionError("lesson is locked or is still being generated")
        progress = self._load_progress(conversation_id, payload)
        completed = set(progress.get("completed_lesson_ids", []))
        completed.add(lesson_id)
        progress["completed_lesson_ids"] = sorted(completed)
        self._save_progress(conversation_id, payload, progress)
        return self._public_course(payload)

    def submit_quiz(self, conversation_id: str, quiz_id: str, answers: list[dict[str, str]]) -> dict[str, Any]:
        payload = self._private_course(conversation_id)
        public = self._public_course(payload)
        public_quiz = _find_quiz(public, quiz_id)
        private_quiz = _find_quiz(payload, quiz_id)
        if private_quiz is None or public_quiz is None:
            raise KeyError("quiz not found")
        if public_quiz.get("is_locked"):
            raise PermissionError("quiz is locked")

        selected = {str(item.get("question_id")): str(item.get("option_id")) for item in answers}
        results = []
        correct = 0
        wrong_source_ids: set[str] = set()
        for question in private_quiz.get("questions", []):
            chosen = selected.get(question["id"], "")
            is_correct = chosen == question.get("correct_option_id")
            correct += int(is_correct)
            if not is_correct:
                wrong_source_ids.update(question.get("source_chunk_ids", []))
            results.append(
                {
                    "question_id": question["id"],
                    "selected_option_id": chosen,
                    "correct_option_id": question.get("correct_option_id"),
                    "correct": is_correct,
                    "explanation": question.get("explanation", ""),
                }
            )
        total = max(1, len(private_quiz.get("questions", [])))
        score = correct / total
        passed = score >= float(private_quiz.get("pass_score", PASS_SCORE))
        progress = self._load_progress(conversation_id, payload)
        attempt_counts = dict(progress.get("quiz_attempt_counts", {}))
        attempt_counts[quiz_id] = int(attempt_counts.get(quiz_id, 0)) + 1
        progress["quiz_attempt_counts"] = attempt_counts
        scores = dict(progress.get("quiz_scores", {}))
        scores[quiz_id] = max(float(scores.get(quiz_id, 0.0)), score)
        progress["quiz_scores"] = scores
        if passed:
            passed_ids = set(progress.get("passed_quiz_ids", []))
            passed_ids.add(quiz_id)
            progress["passed_quiz_ids"] = sorted(passed_ids)
        self._save_progress(conversation_id, payload, progress)
        get_store().execute(
            """
            INSERT INTO coursebuilder_quiz_attempts
              (id, conversation_id, course_id, quiz_id, answers_json, score, passed, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                new_id("course_attempt"),
                conversation_id,
                payload["id"],
                quiz_id,
                json.dumps(answers, ensure_ascii=False),
                score,
                1 if passed else 0,
                utc_now(),
            ),
        )
        concept = _quiz_concept(payload, quiz_id)
        get_learner_service().apply_updates(
            conversation_id,
            LearnerUpdates(
                concepts_covered=[concept],
                concepts_demonstrated=[concept] if passed else [],
                concepts_struggled=[] if passed else [concept],
            ),
        )
        review_lesson_ids = _review_lessons(payload, wrong_source_ids)
        return {
            "score": score,
            "passed": passed,
            "pass_score": private_quiz.get("pass_score", PASS_SCORE),
            "attempt_count": attempt_counts[quiz_id],
            "results": results,
            "review_lesson_ids": review_lesson_ids,
            "course": self._public_course(payload),
        }

    def _ready_material(
        self, conversation_id: str
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str] | dict[str, Any]:
        files = get_store().list_files(conversation_id)
        if not files:
            return {"chapters": [], "status": "empty", "files_total": 0, "files_pending": 0}
        pending = [file for file in files if file["status"] != "ready"]
        if pending:
            return {
                "chapters": [],
                "status": "waiting_for_files",
                "files_total": len(files),
                "files_pending": len(pending),
                "files_failed": sum(file["status"] == "failed" for file in pending),
            }
        chunks = get_store().list_chunks(conversation_id)
        if not chunks:
            return {"chapters": [], "status": "empty", "files_total": len(files), "files_pending": 0}
        file_order = {
            file["id"]: index
            for index, file in enumerate(sorted(files, key=lambda item: (item.get("created_at", ""), item["id"])))
        }
        chunks.sort(key=lambda item: (file_order.get(item["source_file_id"], len(file_order)), item.get("chunk_index", 0)))
        return files, chunks, _source_fingerprint(files, chunks)

    def _course_row(self, conversation_id: str) -> dict[str, Any] | None:
        return get_store().one(
            "SELECT * FROM coursebuilder_courses WHERE conversation_id = ? ORDER BY updated_at DESC LIMIT 1",
            (conversation_id,),
        )

    def _plan_row(self, conversation_id: str) -> dict[str, Any] | None:
        return get_store().one(
            "SELECT * FROM coursebuilder_plans WHERE conversation_id = ? LIMIT 1",
            (conversation_id,),
        )

    def _save_plan(
        self,
        payload: dict[str, Any],
        *,
        quality_mode: str,
        error: str | None = None,
    ) -> None:
        payload.setdefault("metadata", {})["quality_mode"] = quality_mode
        now = utc_now()
        get_store().execute(
            """
            INSERT INTO coursebuilder_plans
              (id, conversation_id, plan_id, payload_json, status, source_fingerprint,
               quality_mode, error, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
              id = excluded.id,
              plan_id = excluded.plan_id,
              payload_json = excluded.payload_json,
              status = excluded.status,
              source_fingerprint = excluded.source_fingerprint,
              quality_mode = excluded.quality_mode,
              error = excluded.error,
              updated_at = excluded.updated_at
            """,
            (
                payload["id"],
                payload["conversation_id"],
                payload["plan_id"],
                json.dumps(payload, ensure_ascii=False),
                payload.get("status", "draft"),
                payload.get("source_fingerprint", ""),
                quality_mode,
                error,
                now,
                now,
            ),
        )

    def _private_course(self, conversation_id: str) -> dict[str, Any]:
        row = self._course_row(conversation_id)
        if row is None:
            raise KeyError("course not found")
        return json.loads(row["payload_json"])

    def _save_course(
        self,
        payload: dict[str, Any],
        *,
        build_id: str,
        quality_mode: str,
        error: str | None = None,
    ) -> None:
        payload.setdefault("metadata", {})["quality_mode"] = quality_mode
        now = utc_now()
        get_store().execute(
            """
            INSERT INTO coursebuilder_courses
              (id, conversation_id, payload_json, status, build_id, source_fingerprint,
               quality_mode, error, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              payload_json = excluded.payload_json,
              status = excluded.status,
              build_id = excluded.build_id,
              source_fingerprint = excluded.source_fingerprint,
              quality_mode = excluded.quality_mode,
              error = excluded.error,
              updated_at = excluded.updated_at
            """,
            (
                payload["id"],
                payload["conversation_id"],
                json.dumps(payload, ensure_ascii=False),
                payload.get("status", "building"),
                build_id,
                payload.get("source_fingerprint", ""),
                quality_mode,
                error,
                now,
                now,
            ),
        )

    def _save_job(
        self,
        job_id: str,
        status: str,
        conversation_id: str,
        build_id: str,
        fingerprint: str,
        stage: str,
        extra: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        payload = {
            "conversation_id": conversation_id,
            "build_id": build_id,
            "source_fingerprint": fingerprint,
            "stage": stage,
            **(extra or {}),
        }
        now = utc_now()
        get_store().execute(
            """
            INSERT INTO background_jobs (id, job_type, status, payload_json, error, created_at, updated_at)
            VALUES (?, 'coursebuilder', ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
              status = excluded.status,
              payload_json = excluded.payload_json,
              error = excluded.error,
              updated_at = excluded.updated_at
            """,
            (job_id, status, json.dumps(payload), error, now, now),
        )

    def _load_progress(self, conversation_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        row = get_store().one(
            "SELECT progress_json FROM coursebuilder_progress WHERE conversation_id = ?",
            (conversation_id,),
        )
        if row is None:
            return _empty_progress(payload)
        try:
            return {**_empty_progress(payload), **json.loads(row["progress_json"])}
        except json.JSONDecodeError:
            return _empty_progress(payload)

    def _save_progress(self, conversation_id: str, payload: dict[str, Any], progress: dict[str, Any]) -> None:
        progress["chapter_fingerprints"] = [chapter.get("content_fingerprint", "") for chapter in payload.get("chapters", [])]
        final_quiz = payload.get("final_quiz") or {}
        progress["course_completed"] = final_quiz.get("id") in set(progress.get("passed_quiz_ids", []))
        get_store().execute(
            """
            INSERT INTO coursebuilder_progress
              (conversation_id, course_id, source_fingerprint, progress_json, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(conversation_id) DO UPDATE SET
              course_id = excluded.course_id,
              source_fingerprint = excluded.source_fingerprint,
              progress_json = excluded.progress_json,
              updated_at = excluded.updated_at
            """,
            (
                conversation_id,
                payload["id"],
                payload.get("source_fingerprint", ""),
                json.dumps(progress, ensure_ascii=False),
                utc_now(),
            ),
        )

    def _reconcile_progress(self, payload: dict[str, Any]) -> None:
        progress = self._load_progress(payload["conversation_id"], payload)
        old_fingerprints = list(progress.get("chapter_fingerprints", []))
        new_fingerprints = [chapter.get("content_fingerprint", "") for chapter in payload.get("chapters", [])]
        prefix = 0
        for old, new in zip(old_fingerprints, new_fingerprints):
            if not old or old != new:
                break
            prefix += 1
        allowed_chapters = payload.get("chapters", [])[:prefix] if old_fingerprints else []
        allowed_lessons = {lesson["id"] for chapter in allowed_chapters for lesson in chapter.get("lessons", [])}
        allowed_quizzes = {chapter["quiz"]["id"] for chapter in allowed_chapters if chapter.get("quiz")}
        if old_fingerprints:
            progress["completed_lesson_ids"] = [
                item for item in progress.get("completed_lesson_ids", []) if item in allowed_lessons
            ]
            progress["passed_quiz_ids"] = [item for item in progress.get("passed_quiz_ids", []) if item in allowed_quizzes]
            progress["quiz_scores"] = {
                key: value for key, value in progress.get("quiz_scores", {}).items() if key in allowed_quizzes
            }
        self._save_progress(payload["conversation_id"], payload, progress)

    def _public_course(self, payload: dict[str, Any]) -> dict[str, Any]:
        public = json.loads(json.dumps(payload))
        progress = self._load_progress(payload["conversation_id"], payload)
        completed_lessons = set(progress.get("completed_lesson_ids", []))
        passed_quizzes = set(progress.get("passed_quiz_ids", []))
        attempt_counts = progress.get("quiz_attempt_counts", {})
        previous_passed = True
        for chapter in public.get("chapters", []):
            chapter_ready = chapter.get("generation_status", "ready") == "ready"
            chapter_unlocked = previous_passed and chapter_ready
            chapter["is_locked"] = not chapter_unlocked
            prior_lessons_complete = chapter_unlocked
            for lesson in chapter.get("lessons", []):
                lesson["is_completed"] = lesson["id"] in completed_lessons
                lesson["is_locked"] = not prior_lessons_complete
                prior_lessons_complete = prior_lessons_complete and lesson["is_completed"]
            quiz = chapter.get("quiz")
            if quiz:
                quiz["is_locked"] = not (chapter_unlocked and prior_lessons_complete)
                quiz["is_passed"] = quiz["id"] in passed_quizzes
                quiz["attempt_count"] = int(attempt_counts.get(quiz["id"], 0))
                _sanitize_and_shuffle_quiz(quiz)
                chapter["is_complete"] = quiz["is_passed"]
                previous_passed = quiz["is_passed"]
            else:
                chapter["is_complete"] = False
                previous_passed = False
        final_quiz = public.get("final_quiz")
        if final_quiz:
            final_quiz["is_locked"] = not (
                public.get("status") == "ready"
                and all(chapter.get("is_complete") for chapter in public.get("chapters", []))
            )
            final_quiz["is_passed"] = final_quiz["id"] in passed_quizzes
            final_quiz["attempt_count"] = int(attempt_counts.get(final_quiz["id"], 0))
            _sanitize_and_shuffle_quiz(final_quiz)
        public["progress"] = {
            **progress,
            "completed_lesson_count": len(completed_lessons),
            "passed_chapter_count": sum(chapter.get("is_complete", False) for chapter in public.get("chapters", [])),
            "course_completed": bool(final_quiz and final_quiz.get("is_passed")),
        }
        return public


async def _build_outline_with_llm(
    provider: Any,
    chunks: list[dict[str, Any]],
    fallback: dict[str, Any],
    *,
    graph: dict[str, Any] | None = None,
) -> CourseOutline:
    allowed_ids = {chunk["id"] for chunk in chunks}
    evidence = _outline_evidence(chunks)
    architecture_hint = _infer_course_architecture(chunks, graph)
    architecture_policy = _architecture_policy(architecture_hint)
    graph_context = _graph_planning_context(graph, allowed_ids)
    system = (
        "You are TeacherLM CourseBuilder. Infer the course architecture from the actual subject before sequencing it. "
        "Do not force a computer-science curriculum onto history, chemistry, physics, mathematics, life science, or a "
        "procedural course. Use the supplied knowledge graph to detect prerequisites, components, variants, causal links, "
        "chronology, formulas, and evaluation relationships. Definitions and prerequisites must precede concepts that "
        "depend on them; a hybrid must follow its components; metrics must follow the systems they evaluate. Historical "
        "material should follow context, causes, chronology, events, consequences, and interpretation. Chemistry should "
        "follow particles/structure, principles, reactions, mechanisms/calculation, procedure, application, and safety. "
        "Physics should follow quantities/foundations, mathematics, laws/models, derivations, experiments, and applications. "
        "Assign architecture_type, architecture_rationale, pedagogical_role, sequencing_reason, and prerequisite titles. "
        "Use every supplied chunk ID at least once across the lessons. Merge duplicate teaching points, but never drop their "
        "chunk IDs. Never add outside facts. Return JSON matching the schema with 1-12 chapters and 1-8 lessons per chapter."
    )
    user = (
        f"DETERMINISTIC ARCHITECTURE HINT: {architecture_hint}\n"
        f"ARCHITECTURE POLICY: {architecture_policy}\n\n"
        f"ALL SOURCE PLANNING UNITS:\n{evidence}\n\n"
        f"KNOWLEDGE GRAPH:\n{graph_context or 'No graph relationships were available.'}\n\n"
        f"FALLBACK TITLE: {fallback.get('title', 'Generated course')}"
    )
    raw = await complete_text(
        provider,
        [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
        json_schema=CourseOutline.model_json_schema(),
        temperature=0.25,
    )
    decoded = json.loads(raw)
    if "architecture_type" not in decoded:
        decoded["architecture_type"] = architecture_hint
    if "architecture_rationale" not in decoded:
        decoded["architecture_rationale"] = _architecture_rationale(decoded["architecture_type"])
    outline = CourseOutline.model_validate(decoded)
    for chapter in outline.chapters:
        chapter.source_chunk_ids = _valid_ids(chapter.source_chunk_ids, allowed_ids)
        for lesson in chapter.lessons:
            lesson.source_chunk_ids = _valid_ids(lesson.source_chunk_ids, allowed_ids)
        lesson_ids = _dedupe(item for lesson in chapter.lessons for item in lesson.source_chunk_ids)
        chapter.source_chunk_ids = _dedupe([*chapter.source_chunk_ids, *lesson_ids])
        if not chapter.source_chunk_ids or any(not lesson.source_chunk_ids for lesson in chapter.lessons):
            raise ValueError("outline contains an ungrounded chapter or lesson")
    outline = _ensure_outline_coverage(outline, chunks, graph)
    return _sequence_outline(outline)


async def _build_chapter_with_llm(
    provider: Any,
    conversation_id: str,
    chapter_index: int,
    outline: OutlineChapter,
    all_chunks: list[dict[str, Any]],
    previous_summary: str,
) -> dict[str, Any]:
    chunk_by_id = {chunk["id"]: chunk for chunk in all_chunks}
    chapter_chunks = [chunk_by_id[item] for item in outline.source_chunk_ids if item in chunk_by_id]
    if not chapter_chunks:
        raise ValueError("chapter has no valid evidence")
    allowed_ids = {chunk["id"] for chunk in chapter_chunks}
    lesson_plan = "\n".join(
        f"- {lesson.title}: role={lesson.pedagogical_role}; prerequisites={lesson.prerequisite_lesson_titles}; "
        f"chunk_ids={lesson.source_chunk_ids}; objectives={lesson.learning_objectives}"
        for lesson in outline.lessons
    )
    evidence = _chapter_evidence(chapter_chunks)
    system = (
        "Write a warm, rigorous chapter from only the supplied evidence. Preserve equations and technical notation. "
        "Each block must cite one or more supplied chunk IDs. Use markdown for explanations, example for grounded worked "
        "examples, and summary for takeaways. Do not invent examples, dates, equations, or claims. Return valid JSON."
    )
    user = (
        f"CHAPTER: {outline.title}\nPEDAGOGICAL ROLE: {outline.pedagogical_role}\n"
        f"SEQUENCING REASON: {outline.sequencing_reason}\n"
        f"PREVIOUS FOUNDATION: {previous_summary or 'This is the first chapter.'}\n"
        f"LESSON PLAN:\n{lesson_plan}\n\nEVIDENCE:\n{evidence}"
    )
    raw = await complete_text(
        provider,
        [LLMMessage(role="system", content=system), LLMMessage(role="user", content=user)],
        json_schema=ChapterDraft.model_json_schema(),
        temperature=0.25,
    )
    draft = ChapterDraft.model_validate(json.loads(raw))
    lessons: list[dict[str, Any]] = []
    for lesson_index, lesson in enumerate(draft.lessons[:MAX_LESSONS_PER_CHAPTER]):
        lesson_ids = _valid_ids(lesson.source_chunk_ids, allowed_ids)
        lesson_chunks = [chunk_by_id[item] for item in lesson_ids]
        if not lesson_chunks:
            raise ValueError("lesson synthesis is not grounded")
        blocks = []
        for block_index, block in enumerate(lesson.blocks):
            source_ids = _valid_ids(block.source_chunk_ids, set(lesson_ids))
            if not source_ids:
                raise ValueError("lesson block is not grounded")
            block_chunks = [chunk_by_id[item] for item in source_ids]
            blocks.append(
                _block(
                    lesson.title,
                    block_index,
                    block.block_type,
                    block.title or lesson.title,
                    block.content,
                    block_chunks,
                )
            )
        blocks.extend(_special_blocks(lesson.title, lesson_chunks, start_index=len(blocks)))
        lesson_id = _stable_id(conversation_id, "lesson", chapter_index, lesson_index, lesson.title)
        lessons.append(
            {
                "id": lesson_id,
                "title": lesson.title,
                "order_index": lesson_index,
                "summary": lesson.summary,
                "learning_objectives": lesson.learning_objectives,
                "pedagogical_role": outline.lessons[min(lesson_index, len(outline.lessons) - 1)].pedagogical_role,
                "sequencing_reason": outline.lessons[min(lesson_index, len(outline.lessons) - 1)].sequencing_reason,
                "prerequisite_lesson_ids": [lessons[-1]["id"]] if lessons else [],
                "source_chunk_ids": lesson_ids,
                "citations": _citations(lesson_chunks, lesson_ids),
                "blocks": blocks[:MAX_BLOCKS_PER_LESSON],
                "content_fingerprint": _content_fingerprint(lesson_ids),
                "generation_status": "ready",
            }
        )
    if not lessons:
        raise ValueError("chapter synthesis returned no lessons")
    _ensure_chapter_chunk_coverage(lessons, chapter_chunks)
    quiz_count = min(10, max(4, len(lessons) + 2))
    quiz = await _build_quiz_with_llm(provider, outline.title, chapter_chunks, quiz_count, "chapter")
    chapter_id = _stable_id(conversation_id, "chapter", chapter_index, outline.title)
    return {
        "id": chapter_id,
        "title": outline.title,
        "description": outline.description,
        "order_index": chapter_index,
        "summary": draft.summary or outline.description,
        "learning_objectives": outline.learning_objectives,
        "pedagogical_role": outline.pedagogical_role,
        "sequencing_reason": outline.sequencing_reason,
        "prerequisite_chapter_ids": [],
        "source_chunk_ids": outline.source_chunk_ids,
        "citations": _citations(chapter_chunks, outline.source_chunk_ids),
        "lessons": lessons,
        "quiz": quiz,
        "content_fingerprint": _content_fingerprint(outline.source_chunk_ids),
        "generation_status": "ready",
    }


def _ensure_chapter_chunk_coverage(
    lessons: list[dict[str, Any]],
    chapter_chunks: list[dict[str, Any]],
) -> None:
    chunk_by_id = {chunk["id"]: chunk for chunk in chapter_chunks}
    covered = {chunk_id for lesson in lessons for chunk_id in lesson.get("source_chunk_ids", [])}
    assignments: dict[str, list[dict[str, Any]]] = {lesson["id"]: [] for lesson in lessons}
    for chunk in chapter_chunks:
        if chunk["id"] in covered:
            continue
        metadata = chunk.get("metadata", {})
        chunk_terms = set(_norm(
            f"{metadata.get('heading_path', '')} {' '.join(metadata.get('key_concepts') or [])} {chunk.get('text', '')[:600]}"
        ).split())
        lesson = max(
            lessons,
            key=lambda item: len(chunk_terms & set(_norm(f"{item['title']} {item.get('summary', '')}").split())),
        )
        assignments[lesson["id"]].append(chunk)
    for lesson in lessons:
        extra_chunks = assignments[lesson["id"]]
        if not extra_chunks:
            continue
        extra_ids = [chunk["id"] for chunk in extra_chunks]
        lesson["source_chunk_ids"] = _dedupe([*lesson.get("source_chunk_ids", []), *extra_ids])
        lesson["citations"] = _citations(
            [chunk_by_id[item] for item in lesson["source_chunk_ids"] if item in chunk_by_id],
            lesson["source_chunk_ids"],
        )
        extension = _source_paragraphs(extra_chunks, sentence_limit=8, max_chars=3600)
        if extension:
            lesson["blocks"].append(
                _block(
                    lesson["title"],
                    len(lesson["blocks"]),
                    "markdown",
                    "Connected source material",
                    extension,
                    extra_chunks,
                )
            )
        lesson["blocks"].extend(_special_blocks(lesson["title"], extra_chunks, start_index=len(lesson["blocks"])))
        lesson["blocks"] = lesson["blocks"][:MAX_BLOCKS_PER_LESSON]
        lesson["content_fingerprint"] = _content_fingerprint(lesson["source_chunk_ids"])


async def _build_quiz_with_llm(
    provider: Any,
    title: str,
    chunks: list[dict[str, Any]],
    count: int,
    scope: Literal["chapter", "course"],
) -> dict[str, Any]:
    fallback = _course_quiz(title, chunks, count=count, scope=scope)
    if not chunks:
        return fallback
    allowed_ids = {chunk["id"] for chunk in chunks}
    evidence = _chapter_evidence(chunks, max_chars=42_000)
    prompt = (
        f"Create exactly {count} rigorous four-option MCQs for the {scope} assessment '{title}'. Test understanding, "
        "relationships, mechanisms, and application. Every answer must be supported by its source_chunk_id. Avoid asking "
        "about documents, authors, pages, or uploaded files. Exactly one option is correct. Return JSON.\n\n"
        f"EVIDENCE:\n{evidence}"
    )
    try:
        raw = await complete_text(
            provider,
            [
                LLMMessage(role="system", content="You create source-grounded educational assessments only."),
                LLMMessage(role="user", content=prompt),
            ],
            json_schema=QuizDraft.model_json_schema(),
            temperature=0.35,
        )
        draft = QuizDraft.model_validate(json.loads(raw))
        if len(draft.questions) != count:
            raise ValueError("quiz question count mismatch")
        questions = []
        for index, question in enumerate(draft.questions):
            if question.source_chunk_id not in allowed_ids:
                raise ValueError("quiz contains an unknown source chunk")
            if len({_norm(option) for option in question.options}) != 4:
                raise ValueError("quiz options are not distinct")
            questions.append(
                _quiz_question(
                    title,
                    index,
                    question.prompt,
                    question.options,
                    question.correct_index,
                    question.explanation,
                    next(chunk for chunk in chunks if chunk["id"] == question.source_chunk_id),
                )
            )
        return {
            "id": _stable_id(title, scope, "quiz"),
            "title": title,
            "scope": scope,
            "questions": questions,
            "pass_score": PASS_SCORE,
        }
    except (ValidationError, ValueError, json.JSONDecodeError, StopIteration):
        return fallback


def _build_fallback_course(
    conversation_id: str,
    chunks: list[dict[str, Any]],
    fingerprint: str,
    *,
    graph: dict[str, Any] | None = None,
) -> dict[str, Any]:
    architecture = _infer_course_architecture(chunks, graph)
    groups = _chapter_groups(chunks, architecture=architecture, graph=graph)
    chapters = []
    for chapter_index, group in enumerate(groups[:MAX_CHAPTERS]):
        lessons = _lessons_for_group(conversation_id, chapter_index, group)
        source_chunk_ids = _dedupe(chunk["id"] for chunk in group["chunks"])
        chapter_id = _stable_id(conversation_id, "chapter", chapter_index, group["title"])
        chapters.append(
            {
                "id": chapter_id,
                "title": group["title"],
                "description": _first_sentence(" ".join(chunk["text"] for chunk in group["chunks"][:2])),
                "order_index": chapter_index,
                "summary": _source_paragraphs(group["chunks"], sentence_limit=5, max_chars=900),
                "learning_objectives": [f"Explain {group['title']}", f"Apply the key ideas in {group['title']}"],
                "pedagogical_role": group["pedagogical_role"],
                "sequencing_reason": group["sequencing_reason"],
                "prerequisite_chapter_ids": [chapters[-1]["id"]] if chapters else [],
                "source_chunk_ids": source_chunk_ids,
                "citations": _citations(group["chunks"], source_chunk_ids),
                "lessons": lessons,
                "quiz": _course_quiz(
                    group["title"],
                    group["chunks"],
                    count=min(10, max(4, len(lessons) + 2)),
                    scope="chapter",
                ),
                "content_fingerprint": _content_fingerprint(source_chunk_ids),
                "generation_status": "ready",
            }
        )
    title = _course_title(chapters, chunks)
    return {
        "id": f"coursebuilder_{conversation_id}",
        "conversation_id": conversation_id,
        "contract_version": "2.1.0",
        "source_fingerprint": fingerprint,
        "status": "ready" if chapters else "empty",
        "title": title,
        "description": "A cumulative course generated and cited from all uploaded materials.",
        "learning_objectives": [f"Master {chapter['title']}" for chapter in chapters[:10]],
        "prerequisites": [],
        "language": "auto",
        "chapters": chapters,
        "final_quiz": _course_quiz(
            f"{title} final assessment",
            chunks,
            count=min(30, max(10, len(chapters) * 2)),
            scope="course",
        ) if chapters else None,
        "metadata": {
            "context_pack_version": COURSEBUILDER_VERSION,
            "stage": "complete",
            "chunk_count": len(chunks),
            "chapter_count": len(chapters),
            "ready_chapter_count": len(chapters),
            "total_chapter_count": len(chapters),
            "lesson_count": sum(len(chapter["lessons"]) for chapter in chapters),
            "source_file_count": len({chunk["source_file_id"] for chunk in chunks}),
            "quality_mode": "fallback",
            "architecture_type": architecture,
            "architecture_rationale": _architecture_rationale(architecture),
            "chunk_coverage_count": len({chunk["id"] for chunk in chunks}),
            "chunk_coverage_total": len({chunk["id"] for chunk in chunks}),
            "chunk_coverage_ratio": 1.0,
            "graph_node_count": len((graph or {}).get("nodes", [])),
            "graph_edge_count": len((graph or {}).get("edges", [])),
        },
    }


def _course_shell(
    conversation_id: str,
    fingerprint: str,
    outline: CourseOutline,
    build_id: str,
) -> dict[str, Any]:
    return {
        "id": f"coursebuilder_{conversation_id}",
        "conversation_id": conversation_id,
        "contract_version": "2.1.0",
        "source_fingerprint": fingerprint,
        "build_id": build_id,
        "status": "building",
        "title": outline.title,
        "description": outline.description,
        "learning_objectives": outline.learning_objectives,
        "prerequisites": [],
        "language": "auto",
        "chapters": [],
        "final_quiz": None,
        "metadata": {
            "context_pack_version": COURSEBUILDER_VERSION,
            "stage": "generating_chapter",
            "ready_chapter_count": 0,
            "total_chapter_count": len(outline.chapters),
            "quality_mode": "llm",
            "architecture_type": outline.architecture_type,
            "architecture_rationale": outline.architecture_rationale,
        },
    }


def _outline_from_course(course: dict[str, Any]) -> CourseOutline:
    return CourseOutline(
        title=course.get("title") or "Generated course",
        description=course.get("description") or "",
        learning_objectives=course.get("learning_objectives") or [],
        architecture_type=course.get("metadata", {}).get("architecture_type", "conceptual"),
        architecture_rationale=course.get("metadata", {}).get("architecture_rationale", ""),
        chapters=[
            OutlineChapter(
                title=chapter["title"],
                description=chapter.get("description", ""),
                learning_objectives=chapter.get("learning_objectives", []),
                source_chunk_ids=chapter.get("source_chunk_ids", []),
                pedagogical_role=chapter.get("pedagogical_role", "core_concept"),
                sequencing_reason=chapter.get("sequencing_reason", ""),
                lessons=[
                    OutlineLesson(
                        title=lesson["title"],
                        summary=lesson.get("summary", ""),
                        learning_objectives=lesson.get("learning_objectives", []),
                        source_chunk_ids=lesson.get("source_chunk_ids", []),
                        pedagogical_role=lesson.get("pedagogical_role", "core_concept"),
                        sequencing_reason=lesson.get("sequencing_reason", ""),
                    )
                    for lesson in chapter.get("lessons", [])
                ],
            )
            for chapter in course.get("chapters", [])
        ],
    )


def _build_course_from_outline_fallback(
    conversation_id: str,
    chunks: list[dict[str, Any]],
    fingerprint: str,
    outline: CourseOutline,
    build_id: str,
    graph: dict[str, Any] | None,
) -> dict[str, Any]:
    source_fallback = _build_fallback_course(conversation_id, chunks, fingerprint, graph=graph)
    payload = _course_shell(conversation_id, fingerprint, outline, build_id)
    chapters: list[dict[str, Any]] = []
    for index, chapter_outline in enumerate(outline.chapters):
        chapter = _fallback_chapter_for_outline(
            conversation_id,
            index,
            chapter_outline,
            chunks,
            _fallback_chapters_by_sources(source_fallback),
        )
        chapter["prerequisite_chapter_ids"] = [chapters[-1]["id"]] if chapters else []
        chapters.append(chapter)
    payload["chapters"] = chapters
    payload["final_quiz"] = _course_quiz(
        f"{outline.title} final assessment",
        chunks,
        count=min(30, max(10, len(chapters) * 2)),
        scope="course",
    ) if chapters else None
    payload["status"] = "ready" if chapters else "empty"
    payload["metadata"].update(
        {
            "stage": "complete",
            "ready_chapter_count": len(chapters),
            "total_chapter_count": len(chapters),
            "chapter_count": len(chapters),
            "lesson_count": sum(len(chapter.get("lessons", [])) for chapter in chapters),
            "quality_mode": "fallback",
            **_coverage_metadata(outline, chunks, graph),
        }
    )
    return payload


def _course_graph(conversation_id: str) -> dict[str, Any]:
    service = get_knowledge_graph_service()
    graph = service.get_graph(conversation_id)
    return graph if graph.get("nodes") else service.rebuild_graph(conversation_id)


def _infer_course_architecture(
    chunks: list[dict[str, Any]],
    graph: dict[str, Any] | None = None,
) -> ArchitectureType:
    graph_labels = " ".join(
        f"{node.get('label', '')} {node.get('description', '')}"
        for node in (graph or {}).get("nodes", [])
        if node.get("node_type") in {"concept", "formula", "procedure", "objective", "section"}
    )
    source = " ".join(
        " ".join(
            [
                str(chunk.get("metadata", {}).get("heading_path") or ""),
                " ".join(str(item) for item in chunk.get("metadata", {}).get("key_concepts") or []),
                str(chunk.get("text", ""))[:1200],
            ]
        )
        for chunk in chunks
    )
    haystack = _norm(f"{source} {graph_labels}")
    keyword_sets: dict[str, tuple[str, ...]] = {
        "historical": (
            "history", "historical", "century", "bce", "dynasty", "empire", "revolution", "war",
            "treaty", "reign", "colonial", "chronology", "timeline", "aftermath", "historiography",
        ),
        "chemistry": (
            "chemistry", "chemical", "atom", "molecule", "molar", "stoichiometry", "reaction", "compound",
            "acid", "base", "oxidation", "reduction", "organic", "inorganic", "equilibrium", "periodic table",
        ),
        "physics": (
            "physics", "force", "velocity", "acceleration", "momentum", "thermodynamics", "quantum",
            "electromagnetic", "optics", "mechanics", "newton", "energy conservation", "wave", "relativity",
        ),
        "mathematics": (
            "mathematics", "theorem", "proof", "lemma", "calculus", "algebra", "geometry", "topology",
            "derivative", "integral", "matrix", "probability distribution", "vector space",
        ),
        "life_science": (
            "biology", "biological", "cell", "genetics", "organism", "protein", "enzyme", "anatomy",
            "physiology", "ecology", "evolution", "metabolism", "dna", "rna",
        ),
        "procedural": (
            "procedure", "step by step", "workflow", "installation", "configuration", "protocol", "tutorial",
            "operating procedure", "laboratory method", "troubleshooting",
        ),
    }
    scores = {
        architecture: sum(len(re.findall(rf"(?<!\w){re.escape(keyword)}(?!\w)", haystack)) for keyword in keywords)
        for architecture, keywords in keyword_sets.items()
    }
    dated_lines = sum(bool(re.search(r"\b(?:\d{3,4}(?:\s*(?:BCE|BC|CE|AD))?)\b", chunk.get("text", ""), re.I)) for chunk in chunks)
    if dated_lines >= max(3, len(chunks) // 3):
        scores["historical"] += 4
    chemical_equations = sum(any(_looks_chemical(line) for line in str(chunk.get("text", "")).splitlines()) for chunk in chunks)
    scores["chemistry"] += chemical_equations * 3
    formula_chunks = sum(int(chunk.get("metadata", {}).get("equation_count") or 0) > 0 for chunk in chunks)
    if scores["physics"] > 0:
        scores["physics"] += min(4, formula_chunks)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    if not ranked or ranked[0][1] < 3:
        return "conceptual"
    if len(ranked) > 1 and ranked[1][1] >= max(3, ranked[0][1] * 0.8):
        return "mixed"
    return ranked[0][0]  # type: ignore[return-value]


def _architecture_policy(architecture: str) -> str:
    return " -> ".join(ARCHITECTURE_ROLE_ORDER.get(architecture, ARCHITECTURE_ROLE_ORDER["conceptual"]))


def _architecture_rationale(architecture: str) -> str:
    descriptions = {
        "conceptual": "Definitions and foundations lead into concepts, methods, variants, integrations, advanced material, applications, and evaluation.",
        "historical": "Scope and context lead into causes, chronology, events, consequences, and evidence-based interpretation.",
        "chemistry": "Atomic and molecular foundations lead into principles, reactions, mechanisms, calculations, laboratory procedure, applications, and safety.",
        "physics": "Physical quantities and mathematical foundations lead into laws, models, derivations, experiments, and applications.",
        "mathematics": "Definitions and axiomatic foundations lead into structures, results, derivations or proofs, methods, and applications.",
        "life_science": "Biological structures lead into functions and mechanisms, experimental evidence, systems, applications, and evaluation.",
        "procedural": "Orientation and prerequisites lead into safe ordered procedures, standard and specialized workflows, application, and verification.",
        "mixed": "The material combines multiple disciplines, so sequencing follows explicit prerequisites and graph relationships before domain-specific conventions.",
    }
    return descriptions.get(architecture, descriptions["conceptual"])


def _pedagogical_role_rank(role: str, architecture: str = "conceptual") -> int:
    order = ARCHITECTURE_ROLE_ORDER.get(architecture, ARCHITECTURE_ROLE_ORDER["conceptual"])
    try:
        return order.index(role)
    except ValueError:
        return len(order)


def _sequencing_reason(role: str, architecture: str = "conceptual") -> str:
    reasons = {
        "definition": "Introduces the vocabulary and scope required by later material.",
        "context": "Establishes the setting needed to understand subsequent developments.",
        "foundation": "Supplies prerequisite knowledge used throughout the course.",
        "structure": "Introduces the components whose behavior is explained later.",
        "law": "Establishes a governing principle before derivation and application.",
        "chronology": "Places developments in time before causal or interpretive analysis.",
        "cause": "Explains the conditions that lead to the central events or changes.",
        "event": "Presents the central development after its context and causes.",
        "consequence": "Examines outcomes after the relevant event or mechanism.",
        "interpretation": "Compares evidence and interpretations after the factual sequence is established.",
        "core_concept": "Develops a central idea after its prerequisites.",
        "mechanism": "Explains how previously introduced components interact.",
        "mathematical_formulation": "Formalizes concepts after their meaning and assumptions are clear.",
        "derivation": "Derives a result from laws and mathematical foundations introduced earlier.",
        "standard_method": "Introduces a principal method after its underlying concepts.",
        "specialized_method": "Builds a specialized approach on a standard method.",
        "reaction": "Applies chemical structure and principles to transformations.",
        "procedure": "Turns prior concepts into an ordered, reproducible process.",
        "experiment": "Connects theory to observation and measurement.",
        "integration": "Combines components only after each component is understood.",
        "advanced": "Extends established methods and concepts into advanced material.",
        "application": "Applies the established foundation to supported cases.",
        "evaluation": "Evaluates systems or claims only after they have been introduced.",
        "safety": "Adds constraints and safe practice after the relevant procedures are known.",
        "synthesis": "Connects and reviews the completed learning sequence.",
    }
    return reasons.get(role, f"Follows the {architecture} course architecture and its prerequisite relationships.")


def _infer_pedagogical_role(
    title: str,
    text: str,
    *,
    course_title: str = "",
    architecture: str = "conceptual",
) -> PedagogicalRole:
    normalized_title = _norm(title)
    normalized = _norm(f"{title} {text[:4000]}")
    if any(term in normalized_title for term in ("what is", "definition", "terminology", "overview", "introduction")):
        return "definition"
    if architecture == "historical":
        if any(term in normalized for term in ("background", "setting", "society", "political context", "economic context")):
            return "context"
        if any(term in normalized_title for term in ("cause", "origin", "factor", "why")):
            return "cause"
        if any(term in normalized_title for term in ("aftermath", "consequence", "impact", "legacy", "effect")):
            return "consequence"
        if any(term in normalized_title for term in ("historiography", "interpretation", "debate", "sources", "evidence")):
            return "interpretation"
        if re.search(r"\b(?:\d{3,4}|century|era|period|timeline|chronology)\b", normalized):
            return "chronology"
        return "event"
    if architecture == "chemistry":
        if any(term in normalized_title for term in ("safety", "hazard", "handling", "disposal")):
            return "safety"
        if any(term in normalized_title for term in ("atom", "molecule", "bond", "periodic", "structure", "orbital")):
            return "structure"
        if any(term in normalized_title for term in ("law", "principle", "mole", "conservation")):
            return "law"
        if any(term in normalized_title for term in ("mechanism", "kinetics", "pathway")):
            return "mechanism"
        if any(term in normalized_title for term in ("reaction", "redox", "acid", "base", "equilibrium", "combustion")):
            return "reaction"
        if any(term in normalized_title for term in ("calculation", "stoichiometry", "equation", "formula", "concentration")):
            return "mathematical_formulation"
        if any(term in normalized_title for term in ("lab", "procedure", "synthesis", "titration", "method")):
            return "procedure"
    if architecture == "physics":
        if any(term in normalized_title for term in ("experiment", "laboratory")):
            return "experiment"
        if any(term in normalized for term in ("unit", "units", "quantity", "quantities", "vector", "measurement", "foundation")):
            return "foundation"
        if any(term in normalized_title for term in ("law", "principle", "conservation")):
            return "law"
        if any(term in normalized_title for term in ("derive", "derivation", "proof")):
            return "derivation"
        if any(term in normalized_title for term in ("equation", "mathematical", "calculus", "formula")):
            return "mathematical_formulation"
    if architecture == "mathematics":
        if any(term in normalized_title for term in ("axiom", "notation", "set", "structure")):
            return "structure"
        if any(term in normalized_title for term in ("theorem", "lemma", "property", "law")):
            return "law"
        if any(term in normalized_title for term in ("proof", "derive", "derivation")):
            return "derivation"
    if any(term in normalized_title for term in ("fundamental", "foundation", "prerequisite", "basic", "background")):
        return "foundation"
    if any(term in normalized_title for term in ("equation", "formula", "matrix", "mathematical", "derivation")):
        return "mathematical_formulation"
    if any(term in normalized_title for term in ("collaborative filtering", "baseline", "standard method", "nearest neighbor")):
        return "standard_method"
    if any(term in normalized_title for term in ("content based", "content-based", "specialized", "variant")):
        return "specialized_method"
    if any(term in normalized_title for term in ("hybrid", "integration", "combined", "fusion", "ensemble")):
        return "integration"
    if any(term in normalized_title for term in ("deep learning", "neural", "advanced", "transformer", "representation learning")):
        return "advanced"
    if any(term in normalized_title for term in ("metric", "evaluation", "validation", "benchmark", "precision", "recall", "rmse", "mae")):
        return "evaluation"
    if any(term in normalized_title for term in ("application", "case study", "use case")):
        return "application"
    if any(term in normalized_title for term in ("method", "algorithm", "model", "filtering")):
        return "standard_method"
    if any(term in normalized_title for term in ("mechanism", "architecture", "process", "workflow")):
        return "mechanism"
    if any(term in normalized_title for term in ("summary", "review", "synthesis", "conclusion")):
        return "synthesis"
    if course_title and _norm(course_title) == normalized_title:
        return "definition"
    return "core_concept"


def _sequence_outline(outline: CourseOutline) -> CourseOutline:
    architecture = outline.architecture_type

    def order_items(items: list[Any], prerequisite_field: str) -> list[Any]:
        by_title = {_norm(item.title): item for item in items}
        original = {_norm(item.title): index for index, item in enumerate(items)}
        dependencies: dict[str, set[str]] = {}
        for item in items:
            key = _norm(item.title)
            dependencies[key] = {
                _norm(title)
                for title in getattr(item, prerequisite_field, [])
                if _norm(title) in by_title and _norm(title) != key
            }
        ordered: list[Any] = []
        remaining = set(by_title)
        while remaining:
            available = [key for key in remaining if not (dependencies[key] & remaining)]
            candidates = available or list(remaining)
            selected = min(
                candidates,
                key=lambda key: (
                    _pedagogical_role_rank(by_title[key].pedagogical_role, architecture),
                    _historical_year_hint(by_title[key].title, getattr(by_title[key], "description", ""), architecture),
                    original[key],
                ),
            )
            ordered.append(by_title[selected])
            remaining.remove(selected)
        return ordered

    outline.chapters = order_items(outline.chapters, "prerequisite_chapter_titles")
    for chapter_index, chapter in enumerate(outline.chapters):
        chapter.lessons = order_items(chapter.lessons, "prerequisite_lesson_titles")
        if chapter_index > 0:
            previous = outline.chapters[chapter_index - 1].title
            chapter.prerequisite_chapter_titles = _dedupe([previous, *chapter.prerequisite_chapter_titles])
        for lesson_index, lesson in enumerate(chapter.lessons):
            if lesson_index > 0:
                previous_lesson = chapter.lessons[lesson_index - 1].title
                lesson.prerequisite_lesson_titles = _dedupe([previous_lesson, *lesson.prerequisite_lesson_titles])
    return outline


def _validate_plan_with_graph(
    outline: CourseOutline,
    chunks: list[dict[str, Any]],
    graph: dict[str, Any] | None,
) -> CourseOutline:
    """Repair coverage and add graph-derived prerequisites without replacing the saved plan."""
    outline = CourseOutline.model_validate(outline.model_dump(mode="json"))
    allowed_ids = {chunk["id"] for chunk in chunks}
    for chapter in outline.chapters:
        chapter.source_chunk_ids = _valid_ids(chapter.source_chunk_ids, allowed_ids)
        for lesson in chapter.lessons:
            lesson.source_chunk_ids = _valid_ids(lesson.source_chunk_ids, allowed_ids)
        chapter.source_chunk_ids = _dedupe(
            [*chapter.source_chunk_ids, *(item for lesson in chapter.lessons for item in lesson.source_chunk_ids)]
        )
    outline = _ensure_outline_coverage(outline, chunks, graph)
    _add_graph_prerequisites(outline, graph)
    return _sequence_outline(outline)


def _add_graph_prerequisites(outline: CourseOutline, graph: dict[str, Any] | None) -> None:
    nodes = {str(node.get("id")): node for node in (graph or {}).get("nodes", [])}
    chapter_by_chunk = {
        chunk_id: chapter
        for chapter in outline.chapters
        for chunk_id in chapter.source_chunk_ids
    }
    lesson_by_chunk = {
        chunk_id: (chapter, lesson)
        for chapter in outline.chapters
        for lesson in chapter.lessons
        for chunk_id in lesson.source_chunk_ids
    }

    def node_chunks(node_id: Any) -> set[str]:
        node = nodes.get(str(node_id), {})
        ids = {str(item) for item in node.get("source_chunk_ids") or []}
        if node.get("node_type") == "chunk" and node.get("ref_id"):
            ids.add(str(node["ref_id"]))
        return ids

    for edge in (graph or {}).get("edges", []):
        if float(edge.get("confidence") or 0.0) < 0.6:
            continue
        relation = str(edge.get("relation_type") or edge.get("edge_type") or "")
        source_ids = node_chunks(edge.get("source_node_id"))
        target_ids = node_chunks(edge.get("target_node_id"))
        if relation in {"requires", "depends_on"}:
            dependent_ids, prerequisite_ids = source_ids, target_ids
        elif relation in {"prerequisite_of", "precedes", "causes", "leads_to", "foundation_for"}:
            dependent_ids, prerequisite_ids = target_ids, source_ids
        else:
            continue
        for dependent_id in dependent_ids:
            dependent_chapter = chapter_by_chunk.get(dependent_id)
            if dependent_chapter is None:
                continue
            for prerequisite_id in prerequisite_ids:
                prerequisite_chapter = chapter_by_chunk.get(prerequisite_id)
                if prerequisite_chapter is not None and prerequisite_chapter is not dependent_chapter:
                    dependent_chapter.prerequisite_chapter_titles = _dedupe(
                        [*dependent_chapter.prerequisite_chapter_titles, prerequisite_chapter.title]
                    )
                dependent_lesson_ref = lesson_by_chunk.get(dependent_id)
                prerequisite_lesson_ref = lesson_by_chunk.get(prerequisite_id)
                if (
                    dependent_lesson_ref is not None
                    and prerequisite_lesson_ref is not None
                    and dependent_lesson_ref[0] is prerequisite_lesson_ref[0]
                    and dependent_lesson_ref[1] is not prerequisite_lesson_ref[1]
                ):
                    dependent_lesson = dependent_lesson_ref[1]
                    dependent_lesson.prerequisite_lesson_titles = _dedupe(
                        [*dependent_lesson.prerequisite_lesson_titles, prerequisite_lesson_ref[1].title]
                    )


def _historical_year_hint(title: str, description: str, architecture: str) -> int:
    if architecture != "historical":
        return 0
    match = re.search(r"\b(\d{3,4})\b", f"{title} {description}")
    return int(match.group(1)) if match else 99999


def _ensure_outline_coverage(
    outline: CourseOutline,
    chunks: list[dict[str, Any]],
    graph: dict[str, Any] | None,
) -> CourseOutline:
    chunk_by_id = {chunk["id"]: chunk for chunk in chunks}
    assigned = {
        chunk_id
        for chapter in outline.chapters
        for lesson in chapter.lessons
        for chunk_id in lesson.source_chunk_ids
    }
    neighbors = _graph_chunk_neighbors(graph)
    for chunk_id in [item for item in chunk_by_id if item not in assigned]:
        chunk = chunk_by_id[chunk_id]
        best: tuple[float, OutlineChapter, OutlineLesson] | None = None
        for chapter in outline.chapters:
            for lesson in chapter.lessons:
                score = _chunk_lesson_affinity(chunk, lesson, chunk_by_id, neighbors.get(chunk_id, set()))
                if best is None or score > best[0]:
                    best = (score, chapter, lesson)
        if best is None:
            continue
        _, chapter, lesson = best
        lesson.source_chunk_ids.append(chunk_id)
        chapter.source_chunk_ids.append(chunk_id)
    for chapter in outline.chapters:
        chapter.source_chunk_ids = _dedupe(
            [*chapter.source_chunk_ids, *(chunk_id for lesson in chapter.lessons for chunk_id in lesson.source_chunk_ids)]
        )
        for lesson in chapter.lessons:
            lesson.source_chunk_ids = _dedupe(lesson.source_chunk_ids)
    covered = {
        chunk_id
        for chapter in outline.chapters
        for lesson in chapter.lessons
        for chunk_id in lesson.source_chunk_ids
    }
    if covered != set(chunk_by_id):
        raise ValueError("course outline does not cover every source chunk")
    return outline


def _chunk_lesson_affinity(
    chunk: dict[str, Any],
    lesson: OutlineLesson,
    chunk_by_id: dict[str, dict[str, Any]],
    graph_neighbors: set[str],
) -> float:
    metadata = chunk.get("metadata", {})
    chunk_terms = set(_norm(
        f"{metadata.get('heading_path', '')} {' '.join(metadata.get('key_concepts') or [])} {chunk.get('text', '')[:500]}"
    ).split())
    lesson_terms = set(_norm(f"{lesson.title} {lesson.summary} {' '.join(lesson.learning_objectives)}").split())
    score = float(len(chunk_terms & lesson_terms) * 2)
    assigned_chunks = [chunk_by_id[item] for item in lesson.source_chunk_ids if item in chunk_by_id]
    if any(item.get("source_file_id") == chunk.get("source_file_id") for item in assigned_chunks):
        score += 2
    chunk_root = (metadata.get("heading_path_list") or [""])[0]
    if any((item.get("metadata", {}).get("heading_path_list") or [None])[0] == chunk_root for item in assigned_chunks):
        score += 3
    score += len(graph_neighbors.intersection(lesson.source_chunk_ids)) * 5
    return score


def _graph_chunk_neighbors(graph: dict[str, Any] | None) -> dict[str, set[str]]:
    nodes = (graph or {}).get("nodes", [])
    node_by_id = {str(node.get("id")): node for node in nodes}
    neighbors: dict[str, set[str]] = {}

    def chunk_ids(node: dict[str, Any] | None) -> set[str]:
        if not node:
            return set()
        ids = {str(item) for item in node.get("source_chunk_ids") or []}
        if node.get("node_type") == "chunk" and node.get("ref_id"):
            ids.add(str(node["ref_id"]))
        return ids

    for edge in (graph or {}).get("edges", []):
        left = chunk_ids(node_by_id.get(str(edge.get("source_node_id"))))
        right = chunk_ids(node_by_id.get(str(edge.get("target_node_id"))))
        for chunk_id in left:
            neighbors.setdefault(chunk_id, set()).update(right)
        for chunk_id in right:
            neighbors.setdefault(chunk_id, set()).update(left)
    return neighbors


def _graph_planning_context(graph: dict[str, Any] | None, allowed_chunk_ids: set[str]) -> str:
    if not graph:
        return ""
    nodes = [
        node for node in graph.get("nodes", [])
        if node.get("node_type") not in {"course", "file", "document", "chunk"}
    ]
    node_by_id = {str(node.get("id")): node for node in graph.get("nodes", [])}
    lines = [f"nodes={len(graph.get('nodes', []))}; edges={len(graph.get('edges', []))}"]
    for node in nodes:
        ids = [str(item) for item in node.get("source_chunk_ids") or [] if str(item) in allowed_chunk_ids]
        lines.append(
            f"NODE {node.get('id')} | {node.get('node_type')} | {node.get('label')} | chunks={','.join(ids)}"
            + (f" | {str(node.get('description'))[:180]}" if node.get("description") else "")
        )
    for edge in graph.get("edges", []):
        relation = str(edge.get("relation_type") or edge.get("edge_type") or "related_to")
        if relation in {"part_of", "supports"}:
            continue
        source = node_by_id.get(str(edge.get("source_node_id")), {})
        target = node_by_id.get(str(edge.get("target_node_id")), {})
        lines.append(
            f"EDGE {source.get('label', edge.get('source_node_id'))} --{relation}--> "
            f"{target.get('label', edge.get('target_node_id'))} | confidence={edge.get('confidence', 0)}"
        )
    return "\n".join(lines)


def _coverage_metadata(
    outline: CourseOutline,
    chunks: list[dict[str, Any]],
    graph: dict[str, Any] | None,
) -> dict[str, Any]:
    covered = {
        chunk_id
        for chapter in outline.chapters
        for lesson in chapter.lessons
        for chunk_id in lesson.source_chunk_ids
    }
    total = {chunk["id"] for chunk in chunks}
    return {
        "architecture_type": outline.architecture_type,
        "architecture_rationale": outline.architecture_rationale,
        "chunk_coverage_count": len(covered & total),
        "chunk_coverage_total": len(total),
        "chunk_coverage_ratio": len(covered & total) / max(1, len(total)),
        "graph_node_count": len((graph or {}).get("nodes", [])),
        "graph_edge_count": len((graph or {}).get("edges", [])),
    }


def _chapter_groups(
    chunks: list[dict[str, Any]],
    *,
    architecture: str | None = None,
    graph: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    architecture = architecture or _infer_course_architecture(chunks, graph)
    root_titles = {
        _norm(str(path[0]))
        for chunk in chunks
        if isinstance((path := chunk.get("metadata", {}).get("heading_path_list")), list) and path
    }
    use_section_level = len(root_titles) <= 1
    grouped: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        path = metadata.get("heading_path_list") or []
        title = str(path[1] if use_section_level and len(path) > 1 else path[0]) if isinstance(path, list) and path else str(
            metadata.get("section_title") or metadata.get("heading_path") or chunk["source_filename"]
        ).split(">")[-1]
        title = _clean_title(title) or chunk["source_filename"]
        key = title.casefold() if use_section_level else f"{chunk['source_file_id']}::{title.casefold()}"
        grouped.setdefault(key, {"title": title, "chunks": []})["chunks"].append(chunk)
    groups = list(grouped.values())
    for group in groups:
        role = _infer_pedagogical_role(
            group["title"],
            " ".join(str(chunk.get("text", ""))[:800] for chunk in group["chunks"][:3]),
            architecture=architecture,
        )
        group["pedagogical_role"] = role
        group["sequencing_reason"] = _sequencing_reason(role, architecture)
        group["architecture_type"] = architecture
    groups = _sort_groups_with_graph(groups, graph, architecture)
    if len(groups) <= MAX_CHAPTERS:
        return groups
    # Merge overflow chapters into the nearest preceding source group so no evidence is discarded.
    kept = groups[:MAX_CHAPTERS]
    for index, group in enumerate(groups[MAX_CHAPTERS:]):
        kept[(MAX_CHAPTERS - 1) - (index % min(3, MAX_CHAPTERS))]["chunks"].extend(group["chunks"])
    return kept


def _sort_groups_with_graph(
    groups: list[dict[str, Any]],
    graph: dict[str, Any] | None,
    architecture: str,
) -> list[dict[str, Any]]:
    chunk_to_group = {
        str(chunk["id"]): index
        for index, group in enumerate(groups)
        for chunk in group["chunks"]
    }
    nodes = {str(node.get("id")): node for node in (graph or {}).get("nodes", [])}

    def node_groups(node: dict[str, Any] | None) -> set[int]:
        if not node:
            return set()
        ids = {str(item) for item in node.get("source_chunk_ids") or []}
        if node.get("node_type") == "chunk" and node.get("ref_id"):
            ids.add(str(node["ref_id"]))
        return {chunk_to_group[item] for item in ids if item in chunk_to_group}

    dependencies: dict[int, set[int]] = {index: set() for index in range(len(groups))}
    for edge in (graph or {}).get("edges", []):
        confidence = float(edge.get("confidence") or 0.0)
        relation = str(edge.get("relation_type") or edge.get("edge_type") or "")
        if confidence < 0.6:
            continue
        source_groups = node_groups(nodes.get(str(edge.get("source_node_id"))))
        target_groups = node_groups(nodes.get(str(edge.get("target_node_id"))))
        if relation in {"requires", "depends_on", "prerequisite_of"}:
            for source_group in source_groups:
                dependencies[source_group].update(target_groups - {source_group})
        elif relation in {"precedes", "causes", "leads_to", "foundation_for"}:
            for target_group in target_groups:
                dependencies[target_group].update(source_groups - {target_group})

    remaining = set(range(len(groups)))
    ordered: list[dict[str, Any]] = []
    while remaining:
        available = [index for index in remaining if not (dependencies[index] & remaining)]
        candidates = available or list(remaining)
        selected = min(
            candidates,
            key=lambda index: (
                _pedagogical_role_rank(groups[index]["pedagogical_role"], architecture),
                _group_year_hint(groups[index], architecture),
                index,
            ),
        )
        ordered.append(groups[selected])
        remaining.remove(selected)
    return ordered


def _group_year_hint(group: dict[str, Any], architecture: str) -> int:
    if architecture != "historical" or group.get("pedagogical_role") not in {"chronology", "event"}:
        return 0
    text = f"{group.get('title', '')} " + " ".join(str(chunk.get("text", ""))[:800] for chunk in group.get("chunks", []))
    match = re.search(r"\b(\d{3,4})\b", text)
    return int(match.group(1)) if match else 99999


def _lessons_for_group(conversation_id: str, chapter_index: int, group: dict[str, Any]) -> list[dict[str, Any]]:
    architecture = group.get("architecture_type") or _infer_course_architecture(group["chunks"])
    section_groups: dict[str, list[dict[str, Any]]] = {}
    for chunk in group["chunks"]:
        metadata = chunk.get("metadata", {})
        title = str(metadata.get("section_title") or metadata.get("heading_path") or group["title"]).split(">")[-1]
        title = _clean_title(title) or group["title"]
        section_groups.setdefault(title.casefold(), []).append(chunk)
    items = list(section_groups.items())
    items = [
        item
        for _, item in sorted(
            enumerate(items),
            key=lambda indexed: (
                _pedagogical_role_rank(
                    _infer_pedagogical_role(
                        _clean_title(indexed[1][1][0].get("metadata", {}).get("section_title") or indexed[1][0]),
                        " ".join(str(chunk.get("text", ""))[:700] for chunk in indexed[1][1][:2]),
                        course_title=group["title"],
                        architecture=architecture,
                    ),
                    architecture,
                ),
                indexed[0],
            ),
        )
    ]
    if len(items) > MAX_LESSONS_PER_CHAPTER:
        base = items[:MAX_LESSONS_PER_CHAPTER]
        for index, (_, overflow_chunks) in enumerate(items[MAX_LESSONS_PER_CHAPTER:]):
            base[(MAX_LESSONS_PER_CHAPTER - 1) - (index % 2)][1].extend(overflow_chunks)
        items = base
    lessons = []
    for lesson_index, (section_key, section_chunks) in enumerate(items):
        title = _clean_title(section_chunks[0].get("metadata", {}).get("section_title") or section_key) or group["title"]
        role = _infer_pedagogical_role(
            title,
            " ".join(str(chunk.get("text", ""))[:900] for chunk in section_chunks[:2]),
            course_title=group["title"],
            architecture=architecture,
        )
        source_chunk_ids = _dedupe(chunk["id"] for chunk in section_chunks)
        lesson_id = _stable_id(conversation_id, "lesson", chapter_index, lesson_index, title)
        lessons.append(
            {
                "id": lesson_id,
                "title": title,
                "order_index": lesson_index,
                "summary": _source_paragraphs(section_chunks, sentence_limit=3, max_chars=600),
                "learning_objectives": [f"Explain {title}", f"Connect {title} to the course foundations"],
                "pedagogical_role": role,
                "sequencing_reason": _sequencing_reason(role, architecture),
                "prerequisite_lesson_ids": [lessons[-1]["id"]] if lessons else [],
                "source_chunk_ids": source_chunk_ids,
                "citations": _citations(section_chunks, source_chunk_ids),
                "blocks": _lesson_blocks(title, section_chunks),
                "content_fingerprint": _content_fingerprint(source_chunk_ids),
                "generation_status": "ready",
            }
        )
    return lessons or [_single_fallback_lesson(conversation_id, chapter_index, group)]


def _single_fallback_lesson(conversation_id: str, chapter_index: int, group: dict[str, Any]) -> dict[str, Any]:
    ids = _dedupe(chunk["id"] for chunk in group["chunks"])
    architecture = group.get("architecture_type") or _infer_course_architecture(group["chunks"])
    role = _infer_pedagogical_role(
        group["title"],
        " ".join(chunk["text"][:900] for chunk in group["chunks"][:2]),
        architecture=architecture,
    )
    return {
        "id": _stable_id(conversation_id, "lesson", chapter_index, 0, group["title"]),
        "title": group["title"],
        "order_index": 0,
        "summary": _source_paragraphs(group["chunks"], sentence_limit=3, max_chars=600),
        "learning_objectives": [f"Understand {group['title']}"],
        "pedagogical_role": role,
        "sequencing_reason": _sequencing_reason(role, architecture),
        "prerequisite_lesson_ids": [],
        "source_chunk_ids": ids,
        "citations": _citations(group["chunks"], ids),
        "blocks": _lesson_blocks(group["title"], group["chunks"]),
        "content_fingerprint": _content_fingerprint(ids),
        "generation_status": "ready",
    }


def _lesson_blocks(title: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    source_ids = _dedupe(chunk["id"] for chunk in chunks)[:8]
    explanation = _source_paragraphs(chunks, sentence_limit=10)
    example = _source_paragraphs(_example_first(chunks), sentence_limit=5)
    takeaway = _source_paragraphs(chunks, sentence_limit=4)
    blocks = [
        _block(title, 0, "markdown", title, explanation, chunks),
        _block(title, 1, "example", "Grounded example", example or explanation, chunks),
        *_special_blocks(title, chunks, start_index=2),
        _block(title, 8, "summary", "Key takeaway", takeaway or explanation, chunks),
    ]
    return [block for block in blocks if block.get("content")][:MAX_BLOCKS_PER_LESSON]


def _special_blocks(title: str, chunks: list[dict[str, Any]], *, start_index: int) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        text = str(chunk.get("text", ""))
        for expression in _equations(text):
            key = _norm(expression)
            if not key or key in seen:
                continue
            seen.add(key)
            is_matrix = bool(re.search(r"\\begin\{[pbvBV]?matrix\}", expression))
            is_chemical = _looks_chemical(expression)
            block_type = "matrix" if is_matrix else "chemical_equation" if is_chemical else "equation"
            content = f"$$\\ce{{{expression}}}$$" if is_chemical else f"$$\n{expression}\n$$"
            blocks.append(
                {
                    "id": _stable_id(title, "special", start_index + len(blocks), block_type),
                    "block_type": block_type,
                    "title": "Chemical equation" if is_chemical else "Matrix" if is_matrix else "Equation",
                    "content": content,
                    "data_json": {"expression": expression},
                    "source_chunk_ids": [chunk["id"]],
                    "citations": _citations([chunk], [chunk["id"]]),
                }
            )
        for table in _markdown_tables(text):
            key = json.dumps(table, sort_keys=True)
            if key in seen:
                continue
            seen.add(key)
            blocks.append(
                {
                    "id": _stable_id(title, "special", start_index + len(blocks), "table"),
                    "block_type": "table",
                    "title": table.get("caption") or "Source table",
                    "content": table["markdown"],
                    "data_json": table,
                    "source_chunk_ids": [chunk["id"]],
                    "citations": _citations([chunk], [chunk["id"]]),
                }
            )
        events = _timeline_events(text)
        if events:
            blocks.append(
                {
                    "id": _stable_id(title, "special", start_index + len(blocks), "timeline"),
                    "block_type": "timeline",
                    "title": "Timeline",
                    "content": "\n".join(f"- **{event['date']}** — {event['description']}" for event in events),
                    "data_json": {"events": events},
                    "source_chunk_ids": [chunk["id"]],
                    "citations": _citations([chunk], [chunk["id"]]),
                }
            )
        if len(blocks) >= 5:
            break
    return blocks[:5]


def _block(
    title: str,
    index: int,
    block_type: str,
    block_title: str,
    content: str,
    chunks: list[dict[str, Any]],
) -> dict[str, Any]:
    ids = _dedupe(chunk["id"] for chunk in chunks)[:8]
    return {
        "id": _stable_id(title, "block", index, block_type),
        "block_type": block_type,
        "title": block_title,
        "content": content,
        "data_json": {},
        "source_chunk_ids": ids,
        "citations": _citations(chunks, ids),
    }


def _course_quiz(title: str, chunks: list[dict[str, Any]], *, count: int, scope: str) -> dict[str, Any]:
    knowledge = []
    for chunk in chunks:
        for sentence in _source_sentences(chunk.get("text", "")):
            if len(sentence) >= 45:
                knowledge.append((sentence, chunk))
                break
    if not knowledge and chunks:
        knowledge = [(str(chunks[0].get("text", ""))[:300], chunks[0])]
    questions = []
    for index in range(count):
        sentence, chunk = knowledge[index % len(knowledge)] if knowledge else ("The sources contain this topic.", {"id": "", "source_filename": "", "metadata": {}, "text": ""})
        concept = _concept_for_chunk(chunk) or title
        prompts = (
            f"Which explanation best captures the role of {concept}?",
            f"Which statement best connects {concept} to this chapter's foundations?",
            f"Which relationship involving {concept} is supported by the course evidence?",
            f"Which description of {concept} should guide later applications?",
        )
        distractors = [
            item[0][:220]
            for item in knowledge
            if item[1].get("id") != chunk.get("id") and _norm(item[0]) != _norm(sentence)
        ][:3]
        generic = [
            f"{concept} is unrelated to the mechanisms described in this course.",
            f"{concept} has the opposite relationship to the one established by the evidence.",
            f"{concept} can be applied without any of the stated conditions or constraints.",
        ]
        options = [sentence[:220], *distractors, *generic][:4]
        while len(options) < 4:
            options.append(generic[len(options) - 1])
        questions.append(
            _quiz_question(
                title,
                index,
                prompts[(index // max(1, len(knowledge))) % len(prompts)],
                options,
                0,
                f"The evidence supports this explanation: {sentence[:320]}",
                chunk,
            )
        )
    return {
        "id": _stable_id(title, scope, "quiz"),
        "title": title,
        "scope": scope,
        "questions": questions,
        "pass_score": PASS_SCORE,
    }


def _quiz_question(
    title: str,
    index: int,
    prompt: str,
    options: list[str],
    correct_index: int,
    explanation: str,
    chunk: dict[str, Any],
) -> dict[str, Any]:
    question_id = _stable_id(title, "question", index)
    option_items = [
        {"id": _stable_id(question_id, "option", option_index), "text": option}
        for option_index, option in enumerate(options)
    ]
    source_id = str(chunk.get("id", ""))
    return {
        "id": question_id,
        "type": "mcq",
        "prompt": prompt,
        "options": option_items,
        "correct_option_id": option_items[correct_index]["id"],
        "explanation": explanation,
        "source_chunk_ids": [source_id] if source_id else [],
        "citations": _citations([chunk], [source_id]) if source_id else [],
    }


def _sanitize_and_shuffle_quiz(quiz: dict[str, Any]) -> None:
    attempt_count = int(quiz.get("attempt_count", 0))
    rng = random.Random(f"{quiz.get('id')}:{attempt_count}")
    for question in quiz.get("questions", []):
        question.pop("correct_option_id", None)
        question.pop("explanation", None)
        rng.shuffle(question.get("options", []))
    rng.shuffle(quiz.get("questions", []))


def _find_lesson(course: dict[str, Any], lesson_id: str) -> dict[str, Any] | None:
    for chapter in course.get("chapters", []):
        for lesson in chapter.get("lessons", []):
            if lesson.get("id") == lesson_id:
                return lesson
    return None


def _find_quiz(course: dict[str, Any], quiz_id: str) -> dict[str, Any] | None:
    for chapter in course.get("chapters", []):
        quiz = chapter.get("quiz")
        if quiz and quiz.get("id") == quiz_id:
            return quiz
    final = course.get("final_quiz")
    return final if final and final.get("id") == quiz_id else None


def _review_lessons(course: dict[str, Any], source_ids: set[str]) -> list[str]:
    return [
        lesson["id"]
        for chapter in course.get("chapters", [])
        for lesson in chapter.get("lessons", [])
        if source_ids.intersection(lesson.get("source_chunk_ids", []))
    ]


def _quiz_concept(course: dict[str, Any], quiz_id: str) -> str:
    for chapter in course.get("chapters", []):
        if chapter.get("quiz", {}).get("id") == quiz_id:
            return chapter.get("title") or "Course chapter"
    return course.get("title") or "Course mastery"


def _empty_progress(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "completed_lesson_ids": [],
        "passed_quiz_ids": [],
        "quiz_scores": {},
        "quiz_attempt_counts": {},
        "chapter_fingerprints": [chapter.get("content_fingerprint", "") for chapter in payload.get("chapters", [])],
        "course_completed": False,
    }


def _fallback_chapters_by_sources(course: dict[str, Any]) -> list[dict[str, Any]]:
    return list(course.get("chapters", []))


def _fallback_chapter_for_outline(
    conversation_id: str,
    index: int,
    outline: OutlineChapter,
    chunks: list[dict[str, Any]],
    fallbacks: list[dict[str, Any]],
) -> dict[str, Any]:
    wanted = set(outline.source_chunk_ids)
    selected = [chunk for chunk in chunks if chunk["id"] in wanted] or chunks[:1]
    chunk_by_id = {chunk["id"]: chunk for chunk in selected}
    lessons: list[dict[str, Any]] = []
    for lesson_index, lesson_outline in enumerate(outline.lessons):
        lesson_chunks = [chunk_by_id[item] for item in lesson_outline.source_chunk_ids if item in chunk_by_id]
        if not lesson_chunks:
            continue
        lesson = _single_fallback_lesson(
            conversation_id,
            index,
            {
                "title": lesson_outline.title,
                "chunks": lesson_chunks,
            },
        )
        lesson.update(
            {
                "id": _stable_id(conversation_id, "lesson", index, lesson_index, lesson_outline.title),
                "title": lesson_outline.title,
                "order_index": lesson_index,
                "summary": lesson_outline.summary or lesson.get("summary", ""),
                "learning_objectives": lesson_outline.learning_objectives,
                "pedagogical_role": lesson_outline.pedagogical_role,
                "sequencing_reason": lesson_outline.sequencing_reason,
                "prerequisite_lesson_ids": [lessons[-1]["id"]] if lessons else [],
            }
        )
        lessons.append(lesson)
    if not lessons:
        lessons = _lessons_for_group(conversation_id, index, {"title": outline.title, "chunks": selected})
    return {
        "id": _stable_id(conversation_id, "chapter", index, outline.title),
        "title": outline.title,
        "description": outline.description,
        "order_index": index,
        "summary": _source_paragraphs(selected, sentence_limit=5, max_chars=900),
        "learning_objectives": outline.learning_objectives,
        "pedagogical_role": outline.pedagogical_role,
        "sequencing_reason": outline.sequencing_reason,
        "prerequisite_chapter_ids": [],
        "source_chunk_ids": _dedupe(chunk["id"] for chunk in selected),
        "citations": _citations(selected, None),
        "lessons": lessons,
        "quiz": _course_quiz(outline.title, selected, count=min(10, max(4, len(lessons) + 2)), scope="chapter"),
        "content_fingerprint": _content_fingerprint(chunk["id"] for chunk in selected),
        "generation_status": "ready",
    }


def _citations(chunks: list[dict[str, Any]], source_chunk_ids: list[str] | None) -> list[dict[str, Any]]:
    wanted = set(source_chunk_ids or [])
    selected = [chunk for chunk in chunks if not wanted or chunk.get("id") in wanted] or chunks[:2]
    citations = []
    seen = set()
    for chunk in selected[:8]:
        if not chunk or chunk.get("id") in seen:
            continue
        seen.add(chunk.get("id"))
        metadata = chunk.get("metadata", {})
        citations.append(
            {
                "chunk_id": chunk.get("id", ""),
                "source": chunk.get("source_filename", ""),
                "section": metadata.get("heading_path") or metadata.get("section_title") or "",
                "snippet": " ".join(str(chunk.get("text", "")).split())[:360],
            }
        )
    return citations


def _source_paragraphs(chunks: list[dict[str, Any]], *, sentence_limit: int, max_chars: int = 5000) -> str:
    sentences = []
    seen = set()
    for chunk in chunks[:10]:
        for sentence in _source_sentences(chunk.get("text", "")):
            key = sentence.casefold()
            if key in seen:
                continue
            seen.add(key)
            sentences.append(sentence)
            if len(sentences) >= sentence_limit:
                break
        if len(sentences) >= sentence_limit:
            break
    return "\n\n".join(
        " ".join(sentences[index : index + 3]).strip()
        for index in range(0, len(sentences), 3)
        if sentences[index : index + 3]
    )[:max_chars].strip()


def _source_sentences(text: str) -> list[str]:
    prose = re.sub(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]", " ", str(text or ""))
    prose = "\n".join(
        line
        for line in prose.replace("\r", "\n").splitlines()
        if not (line.strip().startswith("|") and line.strip().endswith("|"))
    )
    clean = re.sub(r"\s+", " ", prose).strip()
    pieces = re.split(r"(?<=[.!?。！？])\s+|\s+[•*-]\s+", clean)
    out = []
    for piece in pieces:
        sentence = piece.strip(" -•*\t")
        if len(sentence) >= 25 and len(re.findall(r"\w+", sentence)) >= 5:
            out.append(sentence[:700])
    return out or ([clean[:1200]] if clean else [])


def _equations(text: str) -> list[str]:
    display = [" ".join(part for part in match if part).strip() for match in re.findall(r"\$\$(.+?)\$\$|\\\[(.+?)\\\]", text, re.DOTALL)]
    matrix = re.findall(r"(?:[A-Za-z][A-Za-z0-9_{}^]*\s*=\s*)?\\begin\{[pbvBV]?matrix\}[\s\S]+?\\end\{[pbvBV]?matrix\}", text)
    chemical = [line.strip() for line in text.splitlines() if _looks_chemical(line)]
    return _dedupe([*display, *matrix, *chemical])[:8]


def _looks_chemical(value: str) -> bool:
    text = str(value).strip().strip("$")
    return bool(
        re.search(r"(?:->|<-|<=>|=>|→|⇌|\\rightarrow|\\leftrightarrow)", text)
        and len(re.findall(r"(?:\d*[A-Z][a-z]?\d*)+", text)) >= 2
    )


def _markdown_tables(text: str) -> list[dict[str, Any]]:
    lines = text.splitlines()
    tables = []
    index = 0
    while index + 1 < len(lines):
        if "|" not in lines[index] or not re.match(r"^\s*\|?\s*:?-{3,}", lines[index + 1]):
            index += 1
            continue
        block = [lines[index], lines[index + 1]]
        index += 2
        while index < len(lines) and "|" in lines[index] and lines[index].strip():
            block.append(lines[index])
            index += 1
        headers = _table_cells(block[0])
        rows = [_table_cells(line) for line in block[2:]]
        tables.append({"caption": "Source table", "headers": headers, "rows": rows, "markdown": "\n".join(block)})
    return tables[:4]


def _table_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _timeline_events(text: str) -> list[dict[str, str]]:
    event_re = re.compile(r"\b(?:\d{3,4}(?:\s*(?:BCE|BC|CE|AD))?|Q[1-4]\s+\d{4})\b", re.IGNORECASE)
    events = []
    for line in text.splitlines():
        match = event_re.search(line)
        clean = " ".join(line.split())
        if match and 10 <= len(clean) <= 500:
            events.append({"date": match.group(0), "description": clean})
    return events[:12]


def _outline_evidence(chunks: list[dict[str, Any]], max_chars: int = 58_000) -> str:
    units: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        path = metadata.get("heading_path") or " > ".join(metadata.get("heading_path_list") or [])
        key = f"{chunk['source_filename']}::{path or metadata.get('section_title') or 'Untitled section'}"
        units.setdefault(key, []).append(chunk)
    summary_budget = max(90, min(650, max_chars // max(1, len(units)) - 180))
    parts = []
    for key, members in units.items():
        filename, section = key.split("::", 1)
        chunk_ids = [str(member["id"]) for member in members]
        concepts = _dedupe(
            str(concept)
            for member in members
            for concept in member.get("metadata", {}).get("key_concepts") or []
        )
        summary = _summary(" ".join(str(member.get("text", "")) for member in members), summary_budget)
        parts.append(
            f"UNIT file={filename} | section={section} | chunk_ids={','.join(chunk_ids)} | "
            f"concepts={','.join(concepts[:12])}\n{summary}"
        )
    return "\n\n".join(parts)


def _chapter_evidence(chunks: list[dict[str, Any]], max_chars: int = 34_000) -> str:
    parts = []
    size = 0
    for chunk in chunks:
        item = f"[{chunk['id']}] {chunk['source_filename']}\n{chunk['text'][:4500]}"
        if size + len(item) > max_chars:
            break
        parts.append(item)
        size += len(item)
    return "\n\n".join(parts)


def _source_fingerprint(files: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    digest.update(COURSEBUILDER_VERSION.encode())
    for file in sorted(files, key=lambda item: item["id"]):
        digest.update(f"{file['id']}:{file.get('chunk_count', 0)}:{file.get('size_bytes', 0)}".encode())
    for chunk in sorted(chunks, key=lambda item: item["id"]):
        digest.update(chunk["id"].encode())
        digest.update(hashlib.sha256(str(chunk.get("text", "")).encode()).digest())
    return digest.hexdigest()[:24]


def _sort_chunks_by_file_order(files: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> None:
    file_order = {
        file["id"]: index
        for index, file in enumerate(sorted(files, key=lambda item: (item.get("created_at", ""), item["id"])))
    }
    chunks.sort(key=lambda item: (file_order.get(item["source_file_id"], len(file_order)), item.get("chunk_index", 0)))


def _content_fingerprint(source_ids: Any) -> str:
    return hashlib.sha256("|".join(sorted(str(item) for item in source_ids)).encode()).hexdigest()[:20]


def _valid_ids(values: list[str], allowed: set[str]) -> list[str]:
    return _dedupe(value for value in values if value in allowed)


def _example_first(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    terms = ("example", "exemple", "par exemple", "e.g.", "application", "case study", "cas ")
    return [
        chunk
        for _, chunk in sorted(
            enumerate(chunks),
            key=lambda item: (0 if any(term in item[1].get("text", "").casefold() for term in terms) else 1, item[0]),
        )
    ]


def _concept_for_chunk(chunk: dict[str, Any]) -> str:
    metadata = chunk.get("metadata", {})
    section_title = _clean_title(metadata.get("section_title") or "")
    if section_title:
        return section_title
    concepts = metadata.get("key_concepts") or []
    if concepts:
        return str(concepts[0])
    return _clean_title(metadata.get("section_title") or metadata.get("heading_path") or "")


def _course_title(chapters: list[dict[str, Any]], chunks: list[dict[str, Any]]) -> str:
    metadata_titles = [
        str(chunk.get("metadata", {}).get("document_title") or "").strip()
        for chunk in chunks
        if chunk.get("metadata", {}).get("document_title")
    ]
    if metadata_titles:
        return Counter(metadata_titles).most_common(1)[0][0]
    heading_titles = [
        str(path[0]).strip()
        for chunk in chunks
        if isinstance((path := chunk.get("metadata", {}).get("heading_path_list")), list) and path and str(path[0]).strip()
    ]
    if heading_titles:
        return Counter(heading_titles).most_common(1)[0][0]
    return chapters[0]["title"] if chapters else "Generated Course"


def _summary(text: str, max_chars: int) -> str:
    clean = " ".join(str(text or "").split())
    return clean if len(clean) <= max_chars else clean[:max_chars].rsplit(" ", 1)[0].strip()


def _first_sentence(text: str) -> str:
    clean = " ".join(str(text or "").split())
    return re.split(r"(?<=[.!?])\s+", clean, maxsplit=1)[0][:700] if clean else ""


def _clean_title(value: Any) -> str:
    title = re.sub(r"\s+", " ", str(value or "")).strip(" -:;,.#*")
    return "" if not title or len(title) > 140 or re.fullmatch(r"[\d\W_]+", title) else title


def _pedagogical_title_rank(title: str) -> int:
    normalized = _norm(title)
    if any(term in normalized for term in ("foundation", "introduction", "basics", "fundamentals", "prerequisite")):
        return -2
    if any(term in normalized for term in ("overview", "concept", "principle")):
        return -1
    if any(term in normalized for term in ("application", "advanced", "case study", "synthesis")):
        return 2
    if any(term in normalized for term in ("conclusion", "review", "summary")):
        return 3
    return 0


def _dedupe(values: Any) -> list[Any]:
    out = []
    seen = set()
    for value in values:
        key = str(value).casefold()
        if key and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _norm(text: str) -> str:
    return re.sub(r"\W+", " ", str(text).casefold()).strip()


def _stable_id(*parts: Any) -> str:
    raw = "::".join(str(part) for part in parts)
    compact = re.sub(r"[^a-zA-Z0-9]+", "_", raw).strip("_").lower()
    suffix = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"{compact[:82]}_{suffix}" if compact else f"coursebuilder_item_{suffix}"


_coursebuilder_service: LocalCourseBuilderService | None = None


def get_coursebuilder_service() -> LocalCourseBuilderService:
    global _coursebuilder_service
    if _coursebuilder_service is None:
        _coursebuilder_service = LocalCourseBuilderService()
    return _coursebuilder_service
