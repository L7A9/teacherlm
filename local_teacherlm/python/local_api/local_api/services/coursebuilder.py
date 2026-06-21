from __future__ import annotations

import asyncio
import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError
from teacherlm_core.llm.providers import LLMMessage, complete_text
from teacherlm_core.schemas.generator_io import LearnerUpdates

from local_api.config import get_settings
from local_api.db import get_store, new_id, utc_now
from local_api.services.coursebuilder_structure import (
    MarkdownPlanningContext,
    SourceChapter,
    SourceLesson,
    SourceStructure,
    decode_course_rows,
    extract_source_structure,
    load_markdown_planning_context,
    select_representative_chunks,
)
from local_api.services.knowledge_graph import get_knowledge_graph_service
from local_api.services.learner import get_learner_service
from local_api.services.retrieval import get_retrieval_service
from local_api.services.settings import get_settings_service
from local_api.services.source_markup import extract_markdown_tables, strip_tables_for_prose


MAX_LESSONS_PER_CHAPTER = 14
MAX_BLOCKS_PER_LESSON = 9
MAX_LESSON_EVIDENCE_CHUNKS = 12
MIN_TEACHING_SOURCE_WORDS = 70
MIN_RICH_BLOCK_WORDS = 85
PASS_SCORE = 0.70
COURSEBUILDER_VERSION = "local-coursebuilder-v9-clean-structured-content"
COURSE_PLAN_CONTRACT_VERSION = "1.3.0"
QUALITY_PIPELINE_VERSION = "local-platform-quality-v1"
PARSER_PENDING_STATUSES = {"uploaded", "parsing"}


class CourseBuildStopped(Exception):
    """Internal cooperative-cancellation boundary for CourseBuilder work."""

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

LessonStage = Literal["introduction", "content", "conclusion"]

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
    source_chunk_ids: list[str] = Field(default_factory=list)
    pedagogical_role: PedagogicalRole = "core_concept"
    sequencing_reason: str = Field(default="", max_length=400)
    prerequisite_lesson_titles: list[str] = Field(default_factory=list, max_length=8)
    lesson_stage: LessonStage = "content"
    source_queries: list[str] = Field(default_factory=list, max_length=8)


class OutlineChapter(BaseModel):
    title: str = Field(min_length=2, max_length=140)
    description: str = Field(default="", max_length=700)
    learning_objectives: list[str] = Field(default_factory=list, max_length=6)
    source_chunk_ids: list[str] = Field(default_factory=list)
    pedagogical_role: PedagogicalRole = "core_concept"
    sequencing_reason: str = Field(default="", max_length=500)
    prerequisite_chapter_titles: list[str] = Field(default_factory=list, max_length=8)
    source_queries: list[str] = Field(default_factory=list, max_length=10)
    lessons: list[OutlineLesson] = Field(min_length=1)


class CourseOutline(BaseModel):
    title: str = Field(min_length=2, max_length=180)
    description: str = Field(default="", max_length=900)
    learning_objectives: list[str] = Field(default_factory=list, max_length=10)
    architecture_type: ArchitectureType = "conceptual"
    architecture_rationale: str = Field(default="", max_length=700)
    chapters: list[OutlineChapter] = Field(min_length=1)


class MarkdownPlanChapter(BaseModel):
    title: str = Field(min_length=2, max_length=140)
    description: str = Field(default="", max_length=700)
    subchapters: list[str] = Field(min_length=1, max_length=MAX_LESSONS_PER_CHAPTER)


class MarkdownCoursePlan(BaseModel):
    title: str = Field(min_length=2, max_length=180)
    description: str = Field(default="", max_length=900)
    chapters: list[MarkdownPlanChapter] = Field(min_length=1)


class DraftBlock(BaseModel):
    block_type: Literal[
        "markdown", "definition", "example", "procedure", "warning", "summary",
        "table", "equation", "diagram",
    ] = "markdown"
    title: str = Field(default="", max_length=140)
    content: str = Field(min_length=20, max_length=7000)
    source_chunk_ids: list[str] = Field(default_factory=list)
    source_query: str = Field(default="", max_length=500)


class LessonBlockPlan(BaseModel):
    block_type: Literal[
        "markdown", "definition", "example", "procedure", "warning", "summary", "equation", "table",
    ] = "markdown"
    title: str = Field(default="", max_length=140)
    source_query: str = Field(min_length=2, max_length=500)


class LessonBlockPlanBatch(BaseModel):
    blocks: list[LessonBlockPlan] = Field(min_length=1, max_length=5)


class DraftLesson(BaseModel):
    title: str = Field(min_length=2, max_length=140)
    summary: str = Field(default="", max_length=700)
    learning_objectives: list[str] = Field(default_factory=list, max_length=5)
    source_chunk_ids: list[str] = Field(default_factory=list, min_length=1)
    blocks: list[DraftBlock] = Field(min_length=1, max_length=6)
    lesson_stage: LessonStage = "content"


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


@dataclass(slots=True)
class CoursePlanningContext:
    structure: SourceStructure | None
    markdown: MarkdownPlanningContext
    representative_chunks: list[dict[str, Any]]
    document_count: int
    section_count: int


class LocalCourseBuilderService:
    """Grounded, staged course synthesis with a deterministic recovery path."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._plan_locks: dict[str, asyncio.Lock] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._active_build_tasks: dict[str, asyncio.Task[Any]] = {}
        self._cancel_events: dict[str, asyncio.Event] = {}
        self._requested_force: dict[str, bool] = {}
        self._pending_resume_payloads: dict[str, dict[str, Any]] = {}

    def begin_build(self, conversation_id: str, *, force: bool) -> asyncio.Event:
        row = self._course_row(conversation_id)
        if not force and row is not None and row.get("status") == "stopped":
            self._pending_resume_payloads[conversation_id] = json.loads(row["payload_json"])
        else:
            self._pending_resume_payloads.pop(conversation_id, None)
        event = asyncio.Event()
        self._cancel_events[conversation_id] = event
        self._requested_force[conversation_id] = force
        return event

    def mark_build_queued(
        self,
        conversation_id: str,
        *,
        resuming: bool,
        improved_quality: bool = False,
    ) -> dict[str, Any]:
        row = self._course_row(conversation_id)
        if row is None:
            return {"conversation_id": conversation_id, "status": "building", "chapters": [], "metadata": {"stage": "queued"}}
        payload = json.loads(row["payload_json"])
        previous_stage = str(payload.get("metadata", {}).get("stage") or payload.get("status") or "ready")
        payload["status"] = "building"
        payload.setdefault("metadata", {}).update(
            {
                "stage": "queued",
                "queued_from_stage": previous_stage,
                "resuming": resuming,
                "build_profile": "improved" if improved_quality else "fast",
            }
        )
        self._save_course(
            payload,
            build_id=str(row.get("build_id") or payload.get("build_id") or new_id("course_build")),
            quality_mode=str(row.get("quality_mode") or payload.get("metadata", {}).get("quality_mode") or "fallback"),
        )
        return self._public_course(payload)

    def stop_build(self, conversation_id: str) -> dict[str, Any]:
        event = self._cancel_events.setdefault(conversation_id, asyncio.Event())
        event.set()
        task = self._active_build_tasks.get(conversation_id)
        try:
            current = asyncio.current_task()
        except RuntimeError:
            current = None
        if task is not None and task is not current and not task.done():
            task.cancel("CourseBuilder stopped by the student")
        return self._mark_build_stopped(conversation_id)

    def _mark_build_stopped(self, conversation_id: str) -> dict[str, Any]:
        row = self._course_row(conversation_id)
        if row is None:
            return {
                "conversation_id": conversation_id,
                "status": "stopped",
                "chapters": [],
                "metadata": {"stage": "stopped", "stopped_at": utc_now()},
            }
        payload = json.loads(row["payload_json"])
        if (
            payload.get("status") == "ready"
            and conversation_id not in self._active_build_tasks
            and conversation_id not in self._requested_force
        ):
            return self._public_course(payload)
        if payload.get("status") == "stopped":
            self._mark_running_jobs_stopped(conversation_id)
            return self._public_course(payload)
        metadata = payload.setdefault("metadata", {})
        previous_stage = str(metadata.get("stage") or payload.get("status") or "building")
        stopped_from_stage = previous_stage
        if previous_stage == "queued":
            queued_from_stage = str(metadata.get("queued_from_stage") or "")
            stopped_from_stage = (
                str(metadata.get("stopped_from_stage") or "queued")
                if queued_from_stage == "stopped"
                else queued_from_stage or "queued"
            )
        payload["status"] = "stopped"
        metadata.update(
            {
                "stage": "stopped",
                "stopped_at": utc_now(),
                "stopped_from_stage": stopped_from_stage,
                "resume_requires_fresh_plan": bool(
                    self._requested_force.get(conversation_id) and stopped_from_stage in {"complete", "ready"}
                ),
            }
        )
        self._save_course(
            payload,
            build_id=str(row.get("build_id") or payload.get("build_id") or new_id("course_build")),
            quality_mode=str(row.get("quality_mode") or payload.get("metadata", {}).get("quality_mode") or "fallback"),
        )
        self._mark_running_jobs_stopped(conversation_id)
        return self._public_course(payload)

    def _mark_running_jobs_stopped(self, conversation_id: str) -> None:
        for row in get_store().query(
            "SELECT id, payload_json FROM background_jobs WHERE job_type = 'coursebuilder' AND status = 'running'"
        ):
            try:
                payload = json.loads(row.get("payload_json") or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            if payload.get("conversation_id") != conversation_id:
                continue
            get_store().execute(
                "UPDATE background_jobs SET status = 'stopped', updated_at = ? WHERE id = ?",
                (utc_now(), row["id"]),
            )

    @staticmethod
    def _raise_if_stopped(cancel_event: asyncio.Event) -> None:
        if cancel_event.is_set():
            raise CourseBuildStopped

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

    async def prepare_plan_async(
        self,
        conversation_id: str,
        *,
        force: bool = False,
        improved_quality: bool = False,
    ) -> dict[str, Any]:
        """Create the authoritative course skeleton from parser Markdown before chunking."""
        while True:
            files = get_store().list_files(conversation_id)
            if not files:
                return {"status": "empty", "chapters": []}
            if not any(file.get("status") in PARSER_PENDING_STATUSES for file in files):
                break
            await asyncio.sleep(0.05)

        lock = self._plan_locks.setdefault(conversation_id, asyncio.Lock())
        async with lock:
            files = get_store().list_files(conversation_id)
            context = _course_planning_context(
                conversation_id,
                files,
                [],
                structure_only=True,
            )
            if not context.markdown.text.strip():
                return {"status": "empty", "chapters": []}
            structural_fingerprint = _markdown_source_fingerprint(files, context.markdown.text)
            existing = self._plan_row(conversation_id)
            if not force and existing is not None and existing.get("status") in {"draft", "validated"}:
                existing_payload = json.loads(existing["payload_json"])
                existing_structural_fingerprint = str(
                    existing_payload.get("metadata", {}).get("structural_fingerprint")
                    or existing_payload.get("source_fingerprint")
                    or ""
                )
                if existing_structural_fingerprint == structural_fingerprint:
                    return existing_payload

            plan_id = new_id("course_plan")
            planning_payload = {
                "id": f"courseplan_{conversation_id}",
                "plan_id": plan_id,
                "conversation_id": conversation_id,
                "contract_version": COURSE_PLAN_CONTRACT_VERSION,
                "source_fingerprint": structural_fingerprint,
                "status": "planning",
                "outline": {},
                "metadata": {
                    "stage": "planning_from_parser_markdown",
                    "planning_basis": "parser_markdown",
                    "structural_fingerprint": structural_fingerprint,
                    "chunk_count": 0,
                    "source_file_count": context.markdown.source_count,
                    "quality_pipeline_version": QUALITY_PIPELINE_VERSION,
                    "build_profile": "improved" if improved_quality else "fast",
                    "markdown_source_count": context.markdown.source_count,
                    "markdown_raw_chars": context.markdown.raw_chars,
                    "markdown_planning_chars": len(context.markdown.text),
                    "representative_chunk_count": 0,
                    "document_count": context.document_count,
                    "section_count": 0,
                },
            }
            self._save_plan(planning_payload, quality_mode="llm")
            provider = get_settings_service().get_default_chat_provider_config()
            deterministic_outline = (
                _outline_from_source_structure(context.structure, [])
                if context.structure is not None
                else None
            )
            quality_mode = "source_exact" if deterministic_outline is not None else "fallback"
            error: str | None = None
            structure_mode = "source_exact" if deterministic_outline is not None else "inferred"
            if provider is not None:
                try:
                    outline = await asyncio.wait_for(
                        _build_outline_from_markdown_with_llm(
                            provider,
                            context.markdown.text,
                            context.structure,
                        ),
                        timeout=max(20.0, float(provider.timeout_s)),
                    )
                    quality_mode = "llm"
                    structure_mode = "markdown_llm"
                except Exception as exc:  # noqa: BLE001 - exact source headings remain a safe recovery plan.
                    error = str(exc)[:500]
                    if deterministic_outline is None:
                        raise ValueError(f"parser Markdown planning failed: {error}") from exc
                    outline = deterministic_outline
            elif deterministic_outline is not None:
                outline = deterministic_outline
            else:
                raise ValueError("no chat model is configured and parser Markdown has no usable structure")
            outline = _ensure_outline_source_queries(outline, [], None)
            plan_payload = {
                **planning_payload,
                "status": "draft",
                "title": outline.title,
                "architecture_type": outline.architecture_type,
                "outline": outline.model_dump(mode="json"),
                "metadata": {
                    **planning_payload["metadata"],
                    "stage": "draft_ready_before_chunking",
                    "quality_mode": quality_mode,
                    "structure_mode": structure_mode,
                    "source_structure_origin": context.structure.origin if context.structure else None,
                },
            }
            self._save_plan(plan_payload, quality_mode=quality_mode, error=error)
            return plan_payload

    async def replan_and_rebuild_async(self, conversation_id: str) -> dict[str, Any]:
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
        plan_payload = json.loads(plan["payload_json"]) if plan is not None else {}
        markdown_plan = plan_payload.get("metadata", {}).get("planning_basis") == "parser_markdown"
        if (
            plan is not None
            and plan.get("status") in {"draft", "validated"}
            and (plan.get("source_fingerprint") == fingerprint or markdown_plan)
        ):
            try:
                outline = _validate_plan_with_graph(
                    CourseOutline.model_validate(plan_payload.get("outline")),
                    chunks,
                    graph,
                    preserve_structure=plan_payload.get("metadata", {}).get("structure_mode") in {
                        "source_exact",
                        "markdown_llm",
                    },
                )
                validated_plan = {
                    **plan_payload,
                    "source_fingerprint": fingerprint,
                    "status": "validated",
                    "outline": outline.model_dump(mode="json"),
                    "metadata": {
                        **plan_payload.get("metadata", {}),
                        "stage": "validated_with_knowledge_graph",
                        "evidence_bound_after_chunking": True,
                        **_coverage_metadata(outline, chunks, graph),
                    },
                }
                self._save_plan(
                    validated_plan,
                    quality_mode=str(validated_plan.get("metadata", {}).get("quality_mode") or "fallback"),
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

    async def rebuild_async(
        self,
        conversation_id: str,
        *,
        force: bool = False,
        improved_quality: bool = False,
        _cancel_event: asyncio.Event | None = None,
    ) -> dict[str, Any]:
        cancel_event = _cancel_event or self.begin_build(conversation_id, force=force)
        task = asyncio.current_task()
        if task is not None:
            self._active_build_tasks[conversation_id] = task
        try:
            self._raise_if_stopped(cancel_event)
            return await self._rebuild_async_impl(
                conversation_id,
                force=force,
                improved_quality=improved_quality,
                cancel_event=cancel_event,
            )
        except (CourseBuildStopped, asyncio.CancelledError):
            if task is not None and hasattr(task, "uncancel"):
                task.uncancel()
            return self._mark_build_stopped(conversation_id)
        finally:
            if task is not None and self._active_build_tasks.get(conversation_id) is task:
                self._active_build_tasks.pop(conversation_id, None)
            if self._cancel_events.get(conversation_id) is cancel_event:
                self._cancel_events.pop(conversation_id, None)
            self._requested_force.pop(conversation_id, None)
            self._pending_resume_payloads.pop(conversation_id, None)

    async def _rebuild_async_impl(
        self,
        conversation_id: str,
        *,
        force: bool,
        improved_quality: bool,
        cancel_event: asyncio.Event,
    ) -> dict[str, Any]:
        lock = self._locks.setdefault(conversation_id, asyncio.Lock())
        async with lock:
            self._raise_if_stopped(cancel_event)
            ready = self._ready_material(conversation_id)
            if isinstance(ready, dict):
                return ready
            files, chunks, fingerprint = ready
            graph = _course_graph(conversation_id)
            existing = self._course_row(conversation_id)
            resume_payload: dict[str, Any] | None = self._pending_resume_payloads.get(conversation_id)
            if (
                resume_payload is None
                and not force
                and existing is not None
                and existing.get("source_fingerprint") == fingerprint
                and existing.get("status") == "stopped"
            ):
                stopped_payload = json.loads(existing["payload_json"])
                stopped_from_stage = str(stopped_payload.get("metadata", {}).get("stopped_from_stage") or "")
                if stopped_from_stage in {"using_validated_plan", "generating_chapter", "generating_final_quiz"}:
                    resume_payload = stopped_payload
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
            plan_payload = await self.prepare_plan_async(
                conversation_id,
                force=force,
                improved_quality=improved_quality,
            )
            self._raise_if_stopped(cancel_event)
            try:
                markdown_plan = plan_payload.get("metadata", {}).get("planning_basis") == "parser_markdown"
                if plan_payload.get("source_fingerprint") != fingerprint and not markdown_plan:
                    raise ValueError("saved course plan does not match the ready source set")
                outline = CourseOutline.model_validate(plan_payload.get("outline"))
                outline_mode = str(plan_payload.get("metadata", {}).get("quality_mode") or "fallback")
            except (ValidationError, ValueError, TypeError):
                outline = _outline_from_course(fallback)
                outline_mode = "fallback"
            preserve_structure = plan_payload.get("metadata", {}).get("structure_mode") in {
                "source_exact",
                "markdown_llm",
            }
            outline = _validate_plan_with_graph(
                outline,
                chunks,
                graph,
                preserve_structure=preserve_structure,
            )
            validated_plan = {
                **plan_payload,
                "source_fingerprint": fingerprint,
                "status": "validated",
                "outline": outline.model_dump(mode="json"),
                "metadata": {
                    **plan_payload.get("metadata", {}),
                    "stage": "validated_with_knowledge_graph",
                    "evidence_bound_after_chunking": True,
                    **_coverage_metadata(outline, chunks, graph),
                },
            }
            self._save_plan(validated_plan, quality_mode=outline_mode)
            self._raise_if_stopped(cancel_event)
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
                planned_fallback["metadata"]["build_profile"] = "source_only"
                self._save_course(planned_fallback, build_id=build_id, quality_mode="fallback")
                self._reconcile_progress(planned_fallback)
                self._save_job(job_id, "completed", conversation_id, build_id, fingerprint, "complete")
                return self._public_course(planned_fallback)

            planning_payload = _course_shell(conversation_id, fingerprint, outline, build_id)
            build_profile = "improved" if improved_quality else "standard"
            resumed_chapters = _resumable_chapter_prefix(
                conversation_id,
                outline,
                resume_payload.get("chapters", []) if resume_payload else [],
            )
            planning_payload["chapters"] = resumed_chapters
            planning_payload["metadata"].update(
                {
                    "stage": "using_validated_plan",
                    "ready_chapter_count": len(resumed_chapters),
                    "total_chapter_count": len(outline.chapters),
                    "quality_mode": outline_mode,
                    "plan_id": validated_plan.get("plan_id"),
                    "plan_contract_version": validated_plan.get("contract_version"),
                    "build_profile": build_profile,
                    "content_strategy": (
                        "lesson_scoped_multi_block" if improved_quality else "lesson_scoped_single_pass"
                    ),
                }
            )
            self._save_course(planning_payload, build_id=build_id, quality_mode=outline_mode)

            try:
                payload = _course_shell(conversation_id, fingerprint, outline, build_id)
                payload["metadata"].update(_coverage_metadata(outline, chunks, graph))
                payload["metadata"]["plan_id"] = validated_plan.get("plan_id")
                payload["metadata"]["plan_contract_version"] = validated_plan.get("contract_version")
                payload["metadata"]["build_profile"] = build_profile
                payload["metadata"]["content_strategy"] = (
                    "lesson_scoped_multi_block" if improved_quality else "lesson_scoped_single_pass"
                )
                payload["chapters"] = list(resumed_chapters)
                self._save_course(payload, build_id=build_id, quality_mode=outline_mode)
                ready_chapters: list[dict[str, Any]] = list(resumed_chapters)
                fallback_chapters = 0
                previous_summary = ready_chapters[-1].get("summary", "") if ready_chapters else ""
                fallback_by_sources = _fallback_chapters_by_sources(fallback)
                retrieval_totals = {
                    "chapter_retrieval_count": 0,
                    "lesson_retrieval_count": 0,
                    "block_retrieval_count": 0,
                    "block_retry_count": 0,
                    "fallback_block_count": 0,
                    "unsupported_block_count": 0,
                    "weak_support_lesson_count": 0,
                }
                for resumed in ready_chapters:
                    for key in retrieval_totals:
                        retrieval_totals[key] += int(resumed.get("generation_metadata", {}).get(key, 0) or 0)
                for index, chapter_outline in enumerate(outline.chapters):
                    self._raise_if_stopped(cancel_event)
                    if index < len(ready_chapters):
                        continue
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
                        chapter = await _build_chapter_with_quality(
                            provider,
                            conversation_id,
                            index,
                            chapter_outline,
                            chunks,
                            previous_summary,
                            detailed_blocks=improved_quality,
                        )
                    except Exception:  # noqa: BLE001
                        chapter = _fallback_chapter_for_outline(
                            conversation_id,
                            index,
                            chapter_outline,
                            chunks,
                            fallback_by_sources,
                        )
                        chapter["generation_metadata"] = _fallback_quality_counters([chapter])
                        fallback_chapters += 1
                    self._raise_if_stopped(cancel_event)
                    chapter["prerequisite_chapter_ids"] = [ready_chapters[-1]["id"]] if ready_chapters else []
                    chapter_generation = chapter.get("generation_metadata", {})
                    for key in retrieval_totals:
                        retrieval_totals[key] += int(chapter_generation.get(key, 0) or 0)
                    ready_chapters.append(chapter)
                    previous_summary = chapter.get("summary", "")
                    payload["chapters"] = ready_chapters
                    payload["metadata"].update(
                        {
                            "stage": "generating_chapter",
                            "ready_chapter_count": len(ready_chapters),
                            "total_chapter_count": len(outline.chapters),
                            **retrieval_totals,
                        }
                    )
                    self._save_course(
                        payload,
                        build_id=build_id,
                        quality_mode=(
                            "mixed"
                            if fallback_chapters or retrieval_totals["fallback_block_count"] or outline_mode == "fallback"
                            else "llm"
                        ),
                    )

                final_count = min(30, max(10, len(ready_chapters) * 2))
                self._raise_if_stopped(cancel_event)
                payload["metadata"]["stage"] = "generating_final_quiz"
                has_fallback = bool(
                    fallback_chapters or retrieval_totals["fallback_block_count"] or outline_mode == "fallback"
                )
                self._save_course(payload, build_id=build_id, quality_mode="mixed" if has_fallback else "llm")
                payload["final_quiz"] = await _build_quiz_with_llm(
                    provider,
                    title=f"{outline.title} final assessment",
                    chunks=chunks,
                    count=final_count,
                    scope="course",
                )
                self._raise_if_stopped(cancel_event)
                payload["status"] = "ready"
                payload["metadata"].update(
                    {
                        "stage": "complete",
                        "ready_chapter_count": len(ready_chapters),
                        "quality_mode": "mixed" if has_fallback else "llm",
                        "fallback_chapter_count": fallback_chapters,
                        "quality_pipeline_version": QUALITY_PIPELINE_VERSION,
                        **retrieval_totals,
                    }
                )
                quality_mode = payload["metadata"]["quality_mode"]
                self._save_course(payload, build_id=build_id, quality_mode=quality_mode)
                self._reconcile_progress(payload)
                self._save_job(job_id, "completed", conversation_id, build_id, fingerprint, "complete")
                return self._public_course(payload)
            except CourseBuildStopped:
                raise
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
                planned_fallback["metadata"]["build_profile"] = build_profile
                planned_fallback["metadata"]["warnings"] = [
                    "Course synthesis failed validation; a grounded source-extracted course was kept.",
                    str(exc)[:300],
                ]
                self._save_course(planned_fallback, build_id=build_id, quality_mode="fallback", error=str(exc))
                self._reconcile_progress(planned_fallback)
                self._save_job(job_id, "failed", conversation_id, build_id, fingerprint, "failed", error=str(exc))
                return self._public_course(planned_fallback)

    def schedule_rebuild(
        self,
        conversation_id: str,
        *,
        force: bool = False,
        improved_quality: bool = False,
    ) -> None:
        cancel_event = self.begin_build(conversation_id, force=force)
        task = asyncio.create_task(
            self.rebuild_async(
                conversation_id,
                force=force,
                improved_quality=improved_quality,
                _cancel_event=cancel_event,
            )
        )
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
            if (
                current is not None
                and current.get("status") == "stopped"
                and current.get("source_fingerprint") == fingerprint
            ):
                continue
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
        progress["lesson_fingerprints"] = {
            lesson["id"]: lesson.get("content_fingerprint", "")
            for chapter in payload.get("chapters", [])
            for lesson in chapter.get("lessons", [])
        }
        progress["quiz_fingerprints"] = {
            quiz["id"]: _quiz_fingerprint(quiz)
            for quiz in [
                *(chapter.get("quiz") for chapter in payload.get("chapters", []) if chapter.get("quiz")),
                *([payload.get("final_quiz")] if payload.get("final_quiz") else []),
            ]
        }
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
        old_lesson_fingerprints = dict(progress.get("lesson_fingerprints") or {})
        old_quiz_fingerprints = dict(progress.get("quiz_fingerprints") or {})
        lessons = {
            lesson["id"]: lesson.get("content_fingerprint", "")
            for chapter in payload.get("chapters", [])
            for lesson in chapter.get("lessons", [])
        }
        quizzes = {
            quiz["id"]: _quiz_fingerprint(quiz)
            for quiz in [
                *(chapter.get("quiz") for chapter in payload.get("chapters", []) if chapter.get("quiz")),
                *([payload.get("final_quiz")] if payload.get("final_quiz") else []),
            ]
        }

        def unchanged(item_id: str, current: dict[str, str], previous: dict[str, str]) -> bool:
            return item_id in current and (not previous.get(item_id) or previous[item_id] == current[item_id])

        progress["completed_lesson_ids"] = [
            item
            for item in progress.get("completed_lesson_ids", [])
            if unchanged(item, lessons, old_lesson_fingerprints)
        ]
        progress["passed_quiz_ids"] = [
            item
            for item in progress.get("passed_quiz_ids", [])
            if unchanged(item, quizzes, old_quiz_fingerprints)
        ]
        progress["quiz_scores"] = {
            key: value
            for key, value in progress.get("quiz_scores", {}).items()
            if unchanged(key, quizzes, old_quiz_fingerprints)
        }
        progress["quiz_attempt_counts"] = {
            key: value
            for key, value in progress.get("quiz_attempt_counts", {}).items()
            if unchanged(key, quizzes, old_quiz_fingerprints)
        }
        self._save_progress(payload["conversation_id"], payload, progress)

    def _public_course(self, payload: dict[str, Any]) -> dict[str, Any]:
        public = json.loads(json.dumps(payload))
        progress = self._load_progress(payload["conversation_id"], payload)
        completed_lessons = set(progress.get("completed_lesson_ids", []))
        passed_quizzes = set(progress.get("passed_quiz_ids", []))
        attempt_counts = progress.get("quiz_attempt_counts", {})
        sequential_unlocking = get_settings_service().get_coursebuilder_settings().sequential_unlocking_enabled
        previous_passed = True
        for chapter in public.get("chapters", []):
            chapter_ready = chapter.get("generation_status", "ready") == "ready"
            quiz = chapter.get("quiz")
            quiz_passed = bool(quiz and quiz["id"] in passed_quizzes)
            chapter_completed_lessons = {
                lesson["id"]
                for lesson in chapter.get("lessons", [])
                if lesson["id"] in completed_lessons
            }
            can_advance_in_chapter = chapter_ready and (previous_passed or not sequential_unlocking)
            has_reviewable_progress = bool(chapter_completed_lessons) or quiz_passed
            chapter["is_locked"] = not (
                chapter_ready and (can_advance_in_chapter or has_reviewable_progress)
            )
            prior_lessons_complete = can_advance_in_chapter
            for lesson in chapter.get("lessons", []):
                lesson["is_completed"] = lesson["id"] in completed_lessons
                lesson_ready = lesson.get("generation_status", "ready") == "ready"
                lesson["is_locked"] = not (
                    lesson_ready
                    and (
                        lesson["is_completed"]
                        or not sequential_unlocking
                        or prior_lessons_complete
                    )
                )
                prior_lessons_complete = prior_lessons_complete and lesson["is_completed"]
            if quiz:
                all_lessons_complete = all(
                    lesson["id"] in completed_lessons
                    for lesson in chapter.get("lessons", [])
                )
                quiz["is_locked"] = not (
                    quiz_passed
                    or (chapter_ready and all_lessons_complete and (previous_passed or not sequential_unlocking))
                )
                quiz["is_passed"] = quiz_passed
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


def _course_planning_context(
    conversation_id: str,
    files: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    *,
    structure_only: bool = False,
) -> CoursePlanningContext:
    store = get_store()
    documents = decode_course_rows(
        store.query(
            "SELECT * FROM course_documents WHERE conversation_id = ? ORDER BY created_at, id",
            (conversation_id,),
        ),
        metadata_key="metadata_json",
    )
    sections = decode_course_rows(
        store.query(
            "SELECT * FROM course_sections WHERE conversation_id = ? ORDER BY order_index, id",
            (conversation_id,),
        ),
        metadata_key="metadata_json",
        heading_key="heading_path_json",
    ) if _table_has_column("course_sections", "metadata_json") else [
        {
            **row,
            "heading_path": _decode_json_list(row.get("heading_path_json")),
            "metadata": {},
        }
        for row in store.query(
            "SELECT * FROM course_sections WHERE conversation_id = ? ORDER BY order_index, id",
            (conversation_id,),
        )
    ]
    if structure_only:
        sections = []
        chunks = []
    markdown = load_markdown_planning_context(files, get_settings().data_dir)
    structure = extract_source_structure(
        chunks=chunks,
        sections=sections,
        documents=documents,
        markdown=markdown.text,
    )
    return CoursePlanningContext(
        structure=structure,
        markdown=markdown,
        representative_chunks=select_representative_chunks(chunks),
        document_count=len(documents),
        section_count=len(sections),
    )


def _table_has_column(table: str, column: str) -> bool:
    return any(row.get("name") == column for row in get_store().query(f"PRAGMA table_info({table})"))


def _decode_json_list(value: Any) -> list[str]:
    try:
        decoded = json.loads(value or "[]")
    except (json.JSONDecodeError, TypeError):
        return []
    return [str(item) for item in decoded] if isinstance(decoded, list) else []


def _outline_from_source_structure(structure: SourceStructure, chunks: list[dict[str, Any]]) -> CourseOutline:
    """Create an executable outline without changing the source's titles, counts, or order."""
    architecture = _infer_course_architecture(chunks)
    assignments: dict[tuple[int, int], list[dict[str, Any]]] = {
        (chapter_index, lesson_index): []
        for chapter_index, chapter in enumerate(structure.chapters)
        for lesson_index, _lesson in enumerate(chapter.lessons)
    }
    for chunk_index, chunk in enumerate(chunks):
        chapter_scores = [
            _source_structure_affinity(chunk, chapter.title, chapter.source_queries)
            for chapter in structure.chapters
        ]
        best_chapter = max(range(len(structure.chapters)), key=lambda index: (chapter_scores[index], -index))
        chapter = structure.chapters[best_chapter]
        lesson_scores = [
            _source_structure_affinity(chunk, lesson.title, [*chapter.source_queries, *lesson.source_queries])
            for lesson in chapter.lessons
        ]
        best_lesson = max(range(len(chapter.lessons)), key=lambda index: (lesson_scores[index], -index))
        if max(chapter_scores) <= 0 and len(structure.chapters) > 1:
            best_chapter = min(len(structure.chapters) - 1, chunk_index * len(structure.chapters) // max(1, len(chunks)))
            chapter = structure.chapters[best_chapter]
            best_lesson = min(len(chapter.lessons) - 1, chunk_index * len(chapter.lessons) // max(1, len(chunks)))
        assignments[(best_chapter, best_lesson)].append(chunk)

    outline_chapters: list[OutlineChapter] = []
    for chapter_index, source_chapter in enumerate(structure.chapters):
        chapter_chunks = [
            chunk
            for lesson_index in range(len(source_chapter.lessons))
            for chunk in assignments[(chapter_index, lesson_index)]
        ]
        if not chapter_chunks:
            chapter_chunks = sorted(
                chunks,
                key=lambda chunk: -_source_structure_affinity(chunk, source_chapter.title, source_chapter.source_queries),
            )[:1]
        lessons: list[OutlineLesson] = []
        for lesson_index, source_lesson in enumerate(source_chapter.lessons):
            lesson_chunks = assignments[(chapter_index, lesson_index)]
            if not lesson_chunks:
                lesson_chunks = sorted(
                    chapter_chunks,
                    key=lambda chunk: -_source_structure_affinity(
                        chunk,
                        source_lesson.title,
                        [*source_chapter.source_queries, *source_lesson.source_queries],
                    ),
                )[:1]
            lesson_text = " ".join(str(chunk.get("text") or "")[:1200] for chunk in lesson_chunks)
            role = _infer_pedagogical_role(
                source_lesson.title,
                lesson_text,
                course_title=source_chapter.title,
                architecture=architecture,
            )
            lessons.append(
                OutlineLesson(
                    title=_clean_planned_title(source_lesson.title),
                    summary=_first_sentence(lesson_text)[:500],
                    learning_objectives=[f"Explain {_clean_planned_title(source_lesson.title)}"],
                    source_chunk_ids=_dedupe(str(chunk["id"]) for chunk in lesson_chunks),
                    pedagogical_role=role,
                    sequencing_reason="Preserves the sequence defined by the uploaded course source.",
                    prerequisite_lesson_titles=[
                        _clean_planned_title(source_chapter.lessons[lesson_index - 1].title)
                    ] if lesson_index else [],
                    lesson_stage="content",
                    source_queries=_clean_queries(
                        [source_lesson.title, *source_lesson.source_queries, source_chapter.title],
                        limit=8,
                    ),
                )
            )
        source_ids = _dedupe(
            str(chunk["id"])
            for lesson_index in range(len(source_chapter.lessons))
            for chunk in assignments[(chapter_index, lesson_index)]
        ) or _dedupe(str(chunk["id"]) for chunk in chapter_chunks)
        chapter_text = " ".join(str(chunk.get("text") or "")[:800] for chunk in chapter_chunks[:4])
        role = _infer_pedagogical_role(source_chapter.title, chapter_text, architecture=architecture)
        outline_chapters.append(
            OutlineChapter(
                title=_clean_planned_title(source_chapter.title),
                description=source_chapter.description or _first_sentence(chapter_text),
                learning_objectives=[f"Master {lesson.title}" for lesson in lessons[:6]],
                source_chunk_ids=source_ids,
                pedagogical_role=role,
                sequencing_reason="Preserves the chapter order defined by the uploaded course source.",
                prerequisite_chapter_titles=[
                    _clean_planned_title(structure.chapters[chapter_index - 1].title)
                ] if chapter_index else [],
                source_queries=_clean_queries([source_chapter.title, *source_chapter.source_queries], limit=10),
                lessons=lessons,
            )
        )
    return CourseOutline(
        title=_clean_planned_title(structure.title),
        description="A source-grounded course that preserves the uploaded chapter and subchapter structure.",
        learning_objectives=[f"Master {chapter.title}" for chapter in outline_chapters[:10]],
        architecture_type=architecture,
        architecture_rationale=(
            "The source-defined structure is authoritative; domain architecture is used only inside each lesson."
        ),
        chapters=outline_chapters,
    )


def _merge_source_outline_enrichment(exact: CourseOutline, enrichment: CourseOutline) -> CourseOutline:
    """Accept pedagogical prose from the model while keeping the source skeleton immutable."""
    if not _same_source_skeleton(exact, enrichment):
        raise ValueError("source outline enrichment changed the authoritative chapter or lesson skeleton")
    exact.description = enrichment.description or exact.description
    exact.learning_objectives = enrichment.learning_objectives or exact.learning_objectives
    by_chapter_title = {_norm(chapter.title): chapter for chapter in enrichment.chapters}
    for chapter_index, chapter in enumerate(exact.chapters):
        enriched = by_chapter_title.get(_norm(chapter.title))
        if enriched is None and chapter_index < len(enrichment.chapters):
            enriched = enrichment.chapters[chapter_index]
        if enriched is None:
            continue
        chapter.description = enriched.description or chapter.description
        chapter.learning_objectives = enriched.learning_objectives or chapter.learning_objectives
        chapter.source_queries = _clean_queries(
            [*chapter.source_queries, *enriched.source_queries],
            limit=10,
        )
        enriched_lessons = {_norm(lesson.title): lesson for lesson in enriched.lessons}
        for lesson_index, lesson in enumerate(chapter.lessons):
            enriched_lesson = enriched_lessons.get(_norm(lesson.title))
            if enriched_lesson is None and len(enriched.lessons) == len(chapter.lessons):
                enriched_lesson = enriched.lessons[lesson_index]
            if enriched_lesson is None:
                continue
            lesson.summary = enriched_lesson.summary or lesson.summary
            lesson.learning_objectives = enriched_lesson.learning_objectives or lesson.learning_objectives
            lesson.source_queries = _clean_queries(
                [*lesson.source_queries, *enriched_lesson.source_queries],
                limit=8,
            )
    return exact


async def _enrich_source_outline_with_llm(
    provider: Any,
    exact: CourseOutline,
    chunks: list[dict[str, Any]],
) -> CourseOutline:
    raw = await complete_text(
        provider,
        [
            LLMMessage(
                role="system",
                content=(
                    "Enrich a source-defined course outline without restructuring it. Return the exact same course title, "
                    "chapter count, chapter titles, chapter order, lesson count, lesson titles, and lesson order. Improve "
                    "only descriptions, summaries, learning objectives, roles, sequencing reasons, and retrieval queries. "
                    "Use only the supplied evidence and existing chunk IDs. Do not add introduction or conclusion lessons. "
                    "Return JSON matching the schema."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f"AUTHORITATIVE OUTLINE:\n{json.dumps(exact.model_dump(mode='json'), ensure_ascii=False)}\n\n"
                    f"SOURCE EVIDENCE:\n{_outline_evidence(chunks, max_chars=48_000)}"
                ),
            ),
        ],
        json_schema=CourseOutline.model_json_schema(),
        temperature=0.15,
    )
    enrichment = CourseOutline.model_validate(json.loads(raw))
    if not _same_source_skeleton(exact, enrichment):
        raise ValueError("model changed the authoritative source outline")
    return enrichment


def _same_source_skeleton(exact: CourseOutline, candidate: CourseOutline) -> bool:
    if exact.title != candidate.title or len(exact.chapters) != len(candidate.chapters):
        return False
    for exact_chapter, candidate_chapter in zip(exact.chapters, candidate.chapters, strict=True):
        if exact_chapter.title != candidate_chapter.title or len(exact_chapter.lessons) != len(candidate_chapter.lessons):
            return False
        if [lesson.title for lesson in exact_chapter.lessons] != [lesson.title for lesson in candidate_chapter.lessons]:
            return False
    return True


def _source_structure_affinity(chunk: dict[str, Any], title: str, queries: list[str]) -> float:
    metadata = chunk.get("metadata") or {}
    explicit_unit = _norm(str(metadata.get("course_unit_title") or ""))
    explicit_lesson = _norm(str(metadata.get("subchapter_title") or ""))
    target = _norm(title)
    score = 0.0
    if target and target == explicit_unit:
        score += 40.0
    if target and target == explicit_lesson:
        score += 50.0
    haystack = _norm(
        " ".join(
            [
                str(metadata.get("heading_path") or ""),
                " ".join(str(item) for item in metadata.get("heading_path_list") or []),
                " ".join(str(item) for item in metadata.get("subchapter_titles") or []),
                str(chunk.get("text") or "")[:2400],
            ]
        )
    )
    target_terms = set(_norm(" ".join([title, *queries])).split())
    score += len(target_terms.intersection(haystack.split())) * 2.0
    if target and target in haystack:
        score += 12.0
    if str(metadata.get("course_unit_role") or "primary") == "supplemental":
        score -= 1.0
    return score


async def _build_outline_from_markdown_with_llm(
    provider: Any,
    markdown: str,
    detected_structure: SourceStructure | None,
) -> CourseOutline:
    """Plan only the course skeleton from parser Markdown; evidence IDs are attached later."""
    detected_hint = _source_structure_prompt_hint(detected_structure)
    raw = await complete_text(
        provider,
        [
            LLMMessage(
                role="system",
                content=(
                    "Create the authoritative chapter and subchapter plan for a course from parser-produced Markdown. "
                    "This happens before chunking, embeddings, retrieval, or a knowledge graph exist. Read every Markdown "
                    "source. For a multi-file course, preserve explicit lecture, week, module, unit, part, or chapter "
                    "boundaries as chapters in their declared numeric/source order. Use the numbered items under a source "
                    "Plan de la séance, agenda, outline, contents, or table of contents as that chapter's subchapters, "
                    "preserving their source language and order. Nested bullets may inform a parent subchapter but should "
                    "not create duplicate lessons. Merge a supplementary guide into the matching primary chapter when it "
                    "covers the same material; create a separate chapter only when it is genuinely a distinct course unit. "
                    "Never use filenames, title slides, Plan de la séance, Source material, references, page headers, or "
                    "page footers as chapter or subchapter titles. Do not invent generic introduction, summary, or transition "
                    "lessons unless the source plan explicitly names them. Return only a course title, optional course and "
                    "chapter descriptions, chapter titles, and each chapter's ordered subchapter title strings. Do not write "
                    "lesson content, objectives, quizzes, blocks, citations, or chunk IDs. Return JSON matching the schema."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f"SOURCE-DETECTED STRUCTURAL HINTS (use as evidence, repair omissions from the Markdown):\n"
                    f"{detected_hint or 'No reliable deterministic hints were found.'}\n\n"
                    f"PARSER MARKDOWN SOURCES:\n{markdown}"
                ),
            ),
        ],
        json_schema=MarkdownCoursePlan.model_json_schema(),
        temperature=0.1,
    )
    decoded = json.loads(raw)
    _normalize_markdown_plan_payload(decoded, detected_structure)
    plan = MarkdownCoursePlan.model_validate(decoded)
    outline = _outline_from_markdown_plan(plan)
    outline = _reconcile_markdown_outline(outline, detected_structure)
    _sanitize_prechunk_outline(outline)
    _validate_markdown_outline(outline, detected_structure)
    return outline


def _normalize_markdown_plan_payload(
    payload: Any,
    detected_structure: SourceStructure | None,
) -> None:
    if not isinstance(payload, dict):
        return
    if not payload.get("title") and detected_structure is not None:
        payload["title"] = detected_structure.title
    payload["title"] = str(payload.get("title") or "Generated course")[:180]
    payload["description"] = str(payload.get("description") or "")[:900]
    chapters = payload.get("chapters")
    if not isinstance(chapters, list):
        chapters = payload.get("units") if isinstance(payload.get("units"), list) else []
        payload["chapters"] = chapters
    source_chapters = detected_structure.chapters if detected_structure is not None else []
    normalized_chapters: list[dict[str, Any]] = []
    for chapter_index, chapter in enumerate(chapters):
        if isinstance(chapter, str):
            chapter = {"title": chapter}
        if not isinstance(chapter, dict):
            continue
        title = str(
            chapter.get("title")
            or chapter.get("name")
            or (source_chapters[chapter_index].title if chapter_index < len(source_chapters) else "Chapter")
        )[:140]
        values = next(
            (
                chapter.get(alias)
                for alias in ("subchapters", "lessons", "sub_chapters", "sections", "topics")
                if isinstance(chapter.get(alias), list)
            ),
            [],
        )
        subchapters: list[str] = []
        for value in values:
            if isinstance(value, str):
                subchapter = value
            elif isinstance(value, dict):
                subchapter = str(value.get("title") or value.get("name") or value.get("heading") or "")
            else:
                continue
            subchapter = subchapter[:140].strip()
            if subchapter:
                subchapters.append(subchapter)
        if not subchapters and chapter_index < len(source_chapters):
            subchapters = [lesson.title[:140] for lesson in source_chapters[chapter_index].lessons]
        normalized_chapters.append(
            {
                "title": title,
                "description": str(chapter.get("description") or chapter.get("summary") or "")[:700],
                "subchapters": subchapters[:MAX_LESSONS_PER_CHAPTER],
            }
        )
    payload["chapters"] = normalized_chapters


def _outline_from_markdown_plan(plan: MarkdownCoursePlan) -> CourseOutline:
    structure = SourceStructure(
        title=plan.title,
        origin="markdown_llm",
        chapters=[
            SourceChapter(
                title=chapter.title,
                description=chapter.description,
                source_queries=[chapter.title],
                lessons=[SourceLesson(title=title, source_queries=[chapter.title, title]) for title in chapter.subchapters],
            )
            for chapter in plan.chapters
        ],
    )
    outline = _outline_from_source_structure(structure, [])
    outline.description = plan.description or outline.description
    return outline


def _normalize_markdown_outline_payload(
    payload: Any,
    detected_structure: SourceStructure | None,
) -> None:
    """Normalize common structured-output aliases before Pydantic enforces the internal contract."""
    if not isinstance(payload, dict):
        return
    if not payload.get("title") and detected_structure is not None:
        payload["title"] = detected_structure.title
    chapters = payload.get("chapters")
    if not isinstance(chapters, list):
        return
    source_chapters = detected_structure.chapters if detected_structure is not None else []
    for chapter_index, chapter in enumerate(chapters):
        if isinstance(chapter, str):
            chapter = {"title": chapter}
            chapters[chapter_index] = chapter
        if not isinstance(chapter, dict):
            continue
        lessons = chapter.get("lessons")
        if not isinstance(lessons, list):
            lessons = next(
                (
                    chapter.get(alias)
                    for alias in ("subchapters", "sub_chapters", "sections", "topics")
                    if isinstance(chapter.get(alias), list)
                ),
                None,
            )
        if not lessons and chapter_index < len(source_chapters):
            lessons = [{"title": lesson.title} for lesson in source_chapters[chapter_index].lessons]
        normalized_lessons: list[dict[str, Any]] = []
        for lesson in lessons or []:
            if isinstance(lesson, str):
                normalized_lessons.append({"title": lesson})
            elif isinstance(lesson, dict):
                if not lesson.get("title"):
                    lesson["title"] = lesson.get("name") or lesson.get("heading") or "Subchapter"
                normalized_lessons.append(lesson)
        chapter["lessons"] = normalized_lessons


def _reconcile_markdown_outline(
    generated: CourseOutline,
    detected_structure: SourceStructure | None,
) -> CourseOutline:
    """Keep model-authored pedagogy while making explicit source plans structurally authoritative."""
    if detected_structure is None:
        return generated
    exact = _outline_from_source_structure(detected_structure, [])
    exact.description = generated.description or exact.description
    exact.learning_objectives = generated.learning_objectives or exact.learning_objectives
    exact.architecture_type = generated.architecture_type
    exact.architecture_rationale = generated.architecture_rationale or exact.architecture_rationale
    unused = list(generated.chapters)
    for chapter in exact.chapters:
        matched = _best_outline_title_match(chapter.title, unused)
        if matched is None:
            continue
        unused.remove(matched)
        chapter.description = matched.description or chapter.description
        chapter.learning_objectives = matched.learning_objectives or chapter.learning_objectives
        chapter.pedagogical_role = matched.pedagogical_role
        chapter.sequencing_reason = matched.sequencing_reason or chapter.sequencing_reason
        chapter.source_queries = _clean_queries([*chapter.source_queries, *matched.source_queries], limit=10)
        available_lessons = list(matched.lessons)
        for lesson in chapter.lessons:
            enriched = _best_outline_title_match(lesson.title, available_lessons)
            if enriched is None:
                continue
            available_lessons.remove(enriched)
            lesson.summary = enriched.summary or lesson.summary
            lesson.learning_objectives = enriched.learning_objectives or lesson.learning_objectives
            lesson.pedagogical_role = enriched.pedagogical_role
            lesson.sequencing_reason = enriched.sequencing_reason or lesson.sequencing_reason
            lesson.source_queries = _clean_queries([*lesson.source_queries, *enriched.source_queries], limit=8)
    return exact


def _best_outline_title_match(title: str, candidates: list[Any]) -> Any | None:
    target_terms = set(_norm(title).split())
    if not target_terms or not candidates:
        return None
    ranked: list[tuple[float, Any]] = []
    for candidate in candidates:
        candidate_terms = set(_norm(str(getattr(candidate, "title", ""))).split())
        if not candidate_terms:
            continue
        overlap = len(target_terms.intersection(candidate_terms)) / max(1, min(len(target_terms), len(candidate_terms)))
        ranked.append((overlap, candidate))
    if not ranked:
        return None
    score, candidate = max(ranked, key=lambda item: item[0])
    return candidate if score >= 0.45 else None


def _source_structure_prompt_hint(structure: SourceStructure | None) -> str:
    if structure is None:
        return ""
    return json.dumps(
        {
            "course_title_hint": structure.title,
            "origin": structure.origin,
            "chapters": [
                {
                    "title": chapter.title,
                    "subchapters": [lesson.title for lesson in chapter.lessons],
                }
                for chapter in structure.chapters
            ],
        },
        ensure_ascii=False,
    )


def _clamp_outline_payload(payload: Any) -> None:
    if not isinstance(payload, dict):
        return
    payload["title"] = str(payload.get("title") or "Generated course")[:180]
    payload["description"] = str(payload.get("description") or "")[:900]
    payload["architecture_rationale"] = str(payload.get("architecture_rationale") or "")[:700]
    for chapter in payload.get("chapters") or []:
        if not isinstance(chapter, dict):
            continue
        chapter["title"] = str(chapter.get("title") or "Chapter")[:140]
        chapter["description"] = str(chapter.get("description") or "")[:700]
        chapter["sequencing_reason"] = str(chapter.get("sequencing_reason") or "")[:500]
        chapter["source_chunk_ids"] = []
        for lesson in chapter.get("lessons") or []:
            if not isinstance(lesson, dict):
                continue
            lesson["title"] = str(lesson.get("title") or "Lesson")[:140]
            lesson["summary"] = str(lesson.get("summary") or "")[:500]
            lesson["sequencing_reason"] = str(lesson.get("sequencing_reason") or "")[:400]
            lesson["source_chunk_ids"] = []


def _sanitize_prechunk_outline(outline: CourseOutline) -> None:
    outline.title = _clean_planned_title(outline.title)
    for chapter_index, chapter in enumerate(outline.chapters):
        chapter.title = _clean_planned_title(chapter.title)
        chapter.source_chunk_ids = []
        chapter.source_queries = _clean_queries([chapter.title, *chapter.source_queries], limit=10)
        chapter.prerequisite_chapter_titles = [
            outline.chapters[chapter_index - 1].title
        ] if chapter_index else []
        for lesson_index, lesson in enumerate(chapter.lessons):
            lesson.title = _clean_planned_title(lesson.title)
            lesson.summary = lesson.summary[:500]
            lesson.source_chunk_ids = []
            lesson.lesson_stage = "content"
            lesson.source_queries = _clean_queries(
                [lesson.title, *lesson.source_queries, chapter.title],
                limit=8,
            )
            lesson.prerequisite_lesson_titles = [
                chapter.lessons[lesson_index - 1].title
            ] if lesson_index else []


def _validate_markdown_outline(
    outline: CourseOutline,
    detected_structure: SourceStructure | None,
) -> None:
    chapter_keys = [_norm(chapter.title) for chapter in outline.chapters]
    if len(set(chapter_keys)) != len(chapter_keys):
        raise ValueError("Markdown plan contains duplicate chapters")
    forbidden = ("source material", "plan de la seance", "table of contents", "table des matieres")
    for chapter in outline.chapters:
        if any(term in _norm(chapter.title) for term in forbidden) or re.search(r"\.(?:pdf|docx?|pptx?)\b", chapter.title, re.I):
            raise ValueError(f"Markdown plan contains a parser heading as a chapter: {chapter.title}")
        if not chapter.lessons:
            raise ValueError(f"Markdown plan chapter has no subchapters: {chapter.title}")
        lesson_keys = [_norm(lesson.title) for lesson in chapter.lessons]
        if len(set(lesson_keys)) != len(lesson_keys):
            raise ValueError(f"Markdown plan contains duplicate subchapters in {chapter.title}")
        if any(any(term in key for term in forbidden) for key in lesson_keys):
            raise ValueError(f"Markdown plan contains parser headings in {chapter.title}")
    if detected_structure is None or len(detected_structure.chapters) <= 1:
        return
    if len(outline.chapters) < len(detected_structure.chapters):
        raise ValueError("Markdown plan dropped one or more explicit source course units")
    candidate_titles = [_norm(chapter.title) for chapter in outline.chapters]
    for source_chapter in detected_structure.chapters:
        source_terms = set(_norm(source_chapter.title).split())
        if not source_terms:
            continue
        best_coverage = max(
            len(source_terms.intersection(candidate.split())) / len(source_terms)
            for candidate in candidate_titles
        )
        if best_coverage < 0.55:
            raise ValueError(f"Markdown plan dropped the source unit {source_chapter.title}")


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
        "There is no maximum chapter count: create as many coherent chapters as the evidence requires and never merge "
        "distinct topics merely to reduce chapter count. Every chapter must contain at least three ordered subchapters: "
        "one lesson_stage=introduction for its purpose and prerequisites, one or more lesson_stage=content subchapters, "
        "and one lesson_stage=conclusion that synthesizes the chapter and prepares the next foundation. "
        "Assign architecture_type, architecture_rationale, pedagogical_role, sequencing_reason, and prerequisite titles. "
        "For every chapter and subchapter, add concise source_queries containing the original source headings, canonical "
        "concepts, formulas, dates, or named processes that should retrieve its evidence after embedding. "
        "Use every supplied chunk ID at least once across the lessons. Merge duplicate teaching points, but never drop their "
        "chunk IDs. Never add outside facts. Return JSON matching the schema with an unbounded chapter count and 3-14 "
        "subchapters per chapter, including its introduction and conclusion."
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
    outline = _ensure_outline_chapter_arcs(outline)
    outline = _ensure_outline_source_queries(outline, chunks, graph)
    return _sequence_outline(outline)


async def _retrieve_lesson_evidence(
    conversation_id: str,
    chapter: OutlineChapter,
    lesson: OutlineLesson,
    chapter_chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    """Use the normal hybrid+graph retriever, constrained to the planned chapter."""
    chapter_by_id = {str(chunk["id"]): chunk for chunk in chapter_chunks}
    planned = [chapter_by_id[item] for item in lesson.source_chunk_ids if item in chapter_by_id]
    pool = _dedupe_chunk_rows([*planned, *chapter_chunks])
    query = "\n".join(
        _clean_queries(
            [
                lesson.title,
                *lesson.source_queries,
                *lesson.learning_objectives,
                chapter.title,
                *chapter.source_queries,
                "definition detailed explanation mechanism example contrast application",
            ],
            limit=16,
        )
    )
    retrieved_ids: list[str] = []
    retrieval_count = 0
    try:
        hits = await get_retrieval_service().retrieve_for(
            conversation_id=conversation_id,
            user_message=query,
            output_type="text",
            source_file_ids=_dedupe(chunk.get("source_file_id", "") for chunk in pool),
            options={"top_k": MAX_LESSON_EVIDENCE_CHUNKS * 2, "hyde_enabled": False},
        )
        retrieval_count = 1
        retrieved_ids = [str(hit.chunk_id) for hit in hits if str(hit.chunk_id) in chapter_by_id]
    except Exception:  # noqa: BLE001 - deterministic lexical selection remains available offline.
        retrieval_count = 1

    retrieved = [chapter_by_id[item] for item in _dedupe(retrieved_ids) if item in chapter_by_id]
    planned_teaching = [
        chunk
        for chunk in _dedupe_chunk_rows(planned)
        if not _looks_like_structure_only_chunk(chunk)
    ]
    retrieved_candidates = [
        chunk
        for chunk in _dedupe_chunk_rows(retrieved)
        if not _looks_like_structure_only_chunk(chunk)
    ]
    retrieved_supported = [chunk for chunk in retrieved_candidates if _chunk_supports_lesson(chunk, lesson)]
    scoped_candidates = [
        chunk
        for chunk in _dedupe_chunk_rows([*planned_teaching, *retrieved_supported])
        if not _looks_like_structure_only_chunk(chunk)
    ]
    ranked_supported = _rank_lesson_chunks(scoped_candidates, chapter, lesson)
    if _has_rich_teaching_material(ranked_supported):
        return ranked_supported[:MAX_LESSON_EVIDENCE_CHUNKS], retrieval_count

    candidates = [
        chunk
        for chunk in _dedupe_chunk_rows([*scoped_candidates, *chapter_chunks])
        if not _looks_like_structure_only_chunk(chunk)
    ]
    supported = [chunk for chunk in candidates if _chunk_supports_lesson(chunk, lesson)]
    ranked_supported = _rank_lesson_chunks(supported, chapter, lesson)
    # Do not turn a retrieval miss into content from a neighboring subchapter. The
    # platform treats unsupported lessons explicitly; the local app must do the same.
    return ranked_supported[:MAX_LESSON_EVIDENCE_CHUNKS], retrieval_count


def _chunk_supports_lesson(chunk: dict[str, Any], lesson: OutlineLesson) -> bool:
    """Platform-parity lexical gate before semantic ranking and lesson generation."""
    metadata = chunk.get("metadata", {})
    haystack = _fold_text(
        " ".join(
            [
                str(metadata.get("heading_path") or ""),
                " ".join(str(item) for item in metadata.get("heading_path_list") or []),
                str(metadata.get("course_unit_title") or ""),
                str(metadata.get("subchapter_title") or ""),
                " ".join(str(item) for item in metadata.get("subchapter_titles") or []),
                str(chunk.get("text") or ""),
            ]
        )
    )
    title_terms = _meaningful_title_terms(lesson.title)
    source_groups = [
        group
        for group in (_meaningful_title_terms(part) for part in lesson.source_queries)
        if group and group.intersection(title_terms)
    ]
    groups = [
        title_terms,
        *source_groups,
        *(_meaningful_title_terms(part) for part in lesson.learning_objectives),
    ]
    for group in (item for item in groups if item):
        matches = sum(term in haystack for term in group)
        if matches >= min(2, len(group)):
            return True
    lesson_key = _fold_text(lesson.title)
    heading_key = _fold_text(str(metadata.get("section_title") or metadata.get("heading_path") or ""))
    if (
        (
            re.search(r"\bother\b.*\barchitectures?\b", lesson_key)
            or re.search(r"\bautres\b.*\barchitectures?\b", lesson_key)
        )
        and re.search(
            r"\b(?:auto[ -]?encodeurs?|auto[ -]?encoders?|cnn|transformers?|gan|graph neural)\b",
            heading_key,
        )
    ):
        return True
    return False


def _rank_lesson_chunks(
    chunks: list[dict[str, Any]],
    chapter: OutlineChapter,
    lesson: OutlineLesson,
) -> list[dict[str, Any]]:
    terms = set(
        _fold_text(
            " ".join(
                [
                    lesson.title,
                    *lesson.source_queries,
                    *lesson.learning_objectives,
                    chapter.title,
                    *chapter.source_queries,
                ]
            )
        ).split()
    )
    planned_ids = set(lesson.source_chunk_ids)
    title_terms = _meaningful_title_terms(lesson.title)
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        metadata = chunk.get("metadata", {})
        haystack = _fold_text(
            " ".join(
                [
                    str(metadata.get("heading_path") or ""),
                    " ".join(str(item) for item in metadata.get("heading_path_list") or []),
                    " ".join(str(item) for item in metadata.get("key_concepts") or []),
                    str(chunk.get("text") or "")[:2400],
                ]
            )
        )
        overlap = len(terms.intersection(haystack.split()))
        score = float(overlap * 2)
        score += float(len(title_terms.intersection(haystack.split())) * 8)
        explicit_heading = str(
            metadata.get("subchapter_title")
            or metadata.get("section_title")
            or metadata.get("heading_path")
            or ""
        )
        score += _title_term_coverage(explicit_heading, lesson.title) * 40.0
        if str(chunk["id"]) in planned_ids:
            score += 30.0
        score += min(4.0, _word_count(str(chunk.get("text") or "")) / 90.0)
        if _looks_like_structure_only_chunk(chunk):
            score -= 20.0
        scored.append((score, index, chunk))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [chunk for _, _, chunk in scored]


def _planned_lesson_evidence(
    outline: OutlineChapter,
    evidence_by_lesson: dict[str, list[dict[str, Any]]],
    *,
    max_chars: int = 46_000,
) -> str:
    sections: list[str] = []
    remaining = max_chars
    lesson_count = max(1, len(outline.lessons))
    per_lesson = max(1800, max_chars // lesson_count)
    for lesson in outline.lessons:
        chunks = evidence_by_lesson.get(_norm(lesson.title), [])
        header = (
            f"LESSON: {lesson.title}\n"
            f"STAGE: {lesson.lesson_stage}\n"
            f"SOURCE QUERIES: {' | '.join(lesson.source_queries)}\n"
        )
        body = _chapter_evidence(chunks, max_chars=min(per_lesson, max(0, remaining - len(header))))
        section = f"{header}{body}".strip()
        if not section or len(section) > remaining:
            break
        sections.append(section)
        remaining -= len(section) + 2
    return "\n\n".join(sections)


def _dedupe_chunk_rows(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        if not chunk_id or chunk_id in seen:
            continue
        seen.add(chunk_id)
        rows.append(chunk)
    return rows


def _word_count(text: str) -> int:
    return len(re.findall(r"\w+", str(text or ""), flags=re.UNICODE))


def _looks_like_structure_only_chunk(chunk: dict[str, Any]) -> bool:
    text = " ".join(str(chunk.get("text") or "").split()).strip()
    if not text or _word_count(text) <= 4:
        return True
    metadata = chunk.get("metadata", {})
    if any(int(metadata.get(key) or 0) > 0 for key in ("equation_count", "table_count", "timeline_event_count")):
        return False
    folded = _norm(text)
    if "source plan item" in folded:
        return True
    navigation_terms = (
        "previous page", "next page", "back to top", "return to contents", "download pdf",
        "navigation controls", "slide navigation", "home menu", "page navigation",
    )
    if any(term in folded for term in navigation_terms) and _word_count(text) < 120:
        return True
    heading = _fold_text(
        f"{metadata.get('heading_path', '')} {' '.join(str(item) for item in metadata.get('heading_path_list') or [])}"
    )
    structure_terms = (
        "plan de la seance",
        "source material",
        "agenda",
        "contents",
        "outline",
        "table of contents",
        "table des matieres",
    )
    if any(term in heading for term in structure_terms):
        return True
    if any(term in folded[:160] for term in structure_terms) and _word_count(text) < 90:
        return True
    lines = [line.strip(" -\t") for line in str(chunk.get("text") or "").splitlines() if line.strip()]
    if len(lines) >= 3 and _word_count(text) < 45:
        title_like = sum(len(line.split()) <= 9 and not re.search(r"[.!?:;]$", line) for line in lines)
        if title_like / len(lines) >= 0.7:
            return True
    return False


def _has_rich_teaching_material(chunks: list[dict[str, Any]]) -> bool:
    teaching = [chunk for chunk in chunks if not _looks_like_structure_only_chunk(chunk)]
    words = sum(min(_word_count(str(chunk.get("text") or "")), 320) for chunk in teaching[:MAX_LESSON_EVIDENCE_CHUNKS])
    return words >= MIN_TEACHING_SOURCE_WORDS


def _is_thin_teaching_block(block_type: str, content: str, title: str) -> bool:
    if block_type not in {"markdown", "definition", "example", "procedure", "summary"}:
        return not str(content or "").strip()
    text = " ".join(str(content or "").split()).strip()
    if not text or "source plan item" in _norm(text):
        return True
    normalized_text = _norm(text)
    normalized_title = _norm(title)
    if normalized_text == normalized_title:
        return True
    words = _word_count(text)
    if words < MIN_RICH_BLOCK_WORDS:
        return True
    sentences = [item for item in re.split(r"(?<=[.!?])\s+|\n{2,}", text) if _word_count(item) >= 5]
    return len(sentences) < 3 and words < 130


def _fallback_content_for_block(
    block_type: str,
    chunks: list[dict[str, Any]],
    *,
    focus: str = "",
    primary_focus: str = "",
) -> str:
    teaching = [chunk for chunk in chunks if not _looks_like_structure_only_chunk(chunk)]
    ordered = _example_first(teaching) if block_type == "example" else teaching
    if focus.strip():
        focused = _focused_source_paragraphs(
            ordered,
            focus=focus,
            primary_focus=primary_focus,
            sentence_limit=10,
            max_chars=3000,
        )
        if focused:
            return focused
    return _source_paragraphs(ordered, sentence_limit=10, max_chars=3000)


def _insufficient_source_message(topic: str = "") -> str:
    if topic.strip():
        return (
            "The uploaded sources do not contain enough detailed material to teach "
            f"**{topic.strip()}** reliably."
        )
    return "The uploaded sources do not contain enough detailed material to teach this point reliably."


async def _build_chapter_with_quality(
    provider: Any,
    conversation_id: str,
    chapter_index: int,
    outline: OutlineChapter,
    all_chunks: list[dict[str, Any]],
    previous_summary: str,
    *,
    detailed_blocks: bool = True,
) -> dict[str, Any]:
    chunk_by_id = {str(chunk["id"]): chunk for chunk in all_chunks}
    chapter_chunks = [chunk_by_id[item] for item in outline.source_chunk_ids if item in chunk_by_id]
    if not chapter_chunks:
        raise ValueError("chapter has no valid evidence")

    lessons: list[dict[str, Any]] = []
    totals = {
        "chapter_retrieval_count": 1,
        "lesson_retrieval_count": 0,
        "block_retrieval_count": 0,
        "block_retry_count": 0,
        "fallback_block_count": 0,
        "unsupported_block_count": 0,
        "weak_support_lesson_count": 0,
    }
    for lesson_index, planned in enumerate(outline.lessons):
        lesson_chunks, retrieval_count = await _retrieve_lesson_evidence(
            conversation_id,
            outline,
            planned,
            chapter_chunks,
        )
        totals["lesson_retrieval_count"] += retrieval_count
        rich_support = _has_rich_teaching_material(lesson_chunks)
        if not rich_support:
            totals["weak_support_lesson_count"] += 1
        if detailed_blocks:
            try:
                plans = await _plan_lesson_blocks_with_llm(
                    provider,
                    outline,
                    planned,
                    lesson_chunks,
                    previous_summary,
                )
            except Exception:  # noqa: BLE001 - deterministic block planning is the bounded recovery path.
                plans = _fallback_block_plans(planned, lesson_chunks)
        else:
            plans = [
                LessonBlockPlan(
                    block_type="markdown",
                    title=planned.title,
                    source_query=" ".join(
                        _clean_queries([planned.title, *planned.source_queries], limit=3)
                    )[:500],
                )
            ]

        blocks: list[dict[str, Any]] = []
        used_plan_keys: set[tuple[str, str]] = set()
        for plan in plans[:5]:
            plan_key = (plan.block_type, _norm(plan.source_query))
            if plan_key in used_plan_keys:
                continue
            used_plan_keys.add(plan_key)
            block_chunks, block_retrievals = await _retrieve_block_evidence(
                conversation_id,
                outline,
                planned,
                plan,
                lesson_chunks,
                chapter_chunks,
                expanded=False,
            )
            totals["block_retrieval_count"] += block_retrievals
            try:
                generated = await _generate_lesson_block(
                    provider,
                    outline,
                    planned,
                    plan,
                    block_chunks,
                )
                validation_status = "supported"
            except Exception:  # noqa: BLE001 - exactly one expanded retry is intentional.
                totals["block_retry_count"] += 1
                expanded_chunks, expanded_retrievals = await _retrieve_block_evidence(
                    conversation_id,
                    outline,
                    planned,
                    plan,
                    lesson_chunks,
                    chapter_chunks,
                    expanded=True,
                )
                totals["block_retrieval_count"] += expanded_retrievals
                block_chunks = expanded_chunks or block_chunks
                try:
                    generated = await _generate_lesson_block(
                        provider,
                        outline,
                        planned,
                        plan,
                        block_chunks,
                    )
                    validation_status = "supported"
                except Exception:  # noqa: BLE001 - keep the valid course and replace only this block.
                    totals["fallback_block_count"] += 1
                    generated, validation_status = _fallback_generated_block(plan, block_chunks)
                    if validation_status != "supported":
                        totals["unsupported_block_count"] += 1
            generated_ids = _valid_ids(generated.source_chunk_ids, {str(chunk["id"]) for chunk in block_chunks})
            grounded_chunks = [chunk_by_id[item] for item in generated_ids if item in chunk_by_id]
            if not grounded_chunks:
                grounded_chunks = block_chunks[:1]
            blocks.append(
                _block(
                    planned.title,
                    len(blocks),
                    generated.block_type,
                    generated.title or plan.title or planned.title,
                    generated.content,
                    grounded_chunks,
                    validation_status=validation_status,
                    source_query=plan.source_query,
                )
            )

        if not blocks:
            fallback_plan = _fallback_block_plans(planned, lesson_chunks)[0]
            generated, validation_status = _fallback_generated_block(fallback_plan, lesson_chunks)
            totals["fallback_block_count"] += 1
            totals["unsupported_block_count"] += int(validation_status != "supported")
            blocks.append(
                _block(
                    planned.title,
                    0,
                    generated.block_type,
                    generated.title or planned.title,
                    generated.content,
                    lesson_chunks[:8],
                    validation_status=validation_status,
                    source_query=fallback_plan.source_query,
                )
            )
        if detailed_blocks or all(
            block.get("validation_status") == "insufficient_source_material"
            for block in blocks
        ):
            blocks.extend(
                _special_blocks(
                    planned.title,
                    lesson_chunks,
                    start_index=len(blocks),
                    existing_blocks=blocks,
                )
            )
        blocks = _dedupe_lesson_blocks(blocks)
        lesson_source_ids = _dedupe(
            [
                *planned.source_chunk_ids,
                *(source_id for block in blocks for source_id in block.get("source_chunk_ids", [])),
            ]
        )
        source_chunks = [chunk_by_id[item] for item in lesson_source_ids if item in chunk_by_id]
        lesson_id = _stable_id(conversation_id, "lesson", chapter_index, lesson_index, planned.title)
        lessons.append(
            {
                "id": lesson_id,
                "title": planned.title,
                "order_index": lesson_index,
                "summary": planned.summary or _source_paragraphs(lesson_chunks, sentence_limit=3, max_chars=650),
                "learning_objectives": planned.learning_objectives,
                "pedagogical_role": planned.pedagogical_role,
                "sequencing_reason": planned.sequencing_reason,
                "lesson_stage": planned.lesson_stage,
                "prerequisite_lesson_ids": [lessons[-1]["id"]] if lessons else [],
                "source_chunk_ids": lesson_source_ids,
                "citations": _citations(source_chunks, lesson_source_ids),
                "blocks": blocks[:MAX_BLOCKS_PER_LESSON],
                "support_status": (
                    "supported"
                    if rich_support and all(block.get("validation_status") == "supported" for block in blocks)
                    else "insufficient_source_material"
                ),
                "source_queries": planned.source_queries,
                "content_fingerprint": _content_fingerprint(lesson_source_ids),
                "generation_status": "ready",
            }
        )

    if not lessons:
        raise ValueError("chapter synthesis returned no lessons")
    _ensure_chapter_chunk_coverage(lessons, chapter_chunks)
    quiz_count = min(10, max(4, len(lessons) + 2))
    quiz = await _build_quiz_with_llm(provider, outline.title, chapter_chunks, quiz_count, "chapter")
    return {
        "id": _stable_id(conversation_id, "chapter", chapter_index, outline.title),
        "title": outline.title,
        "description": outline.description,
        "order_index": chapter_index,
        "summary": _source_paragraphs(chapter_chunks, sentence_limit=5, max_chars=1000) or outline.description,
        "learning_objectives": outline.learning_objectives,
        "pedagogical_role": outline.pedagogical_role,
        "sequencing_reason": outline.sequencing_reason,
        "prerequisite_chapter_ids": [],
        "source_chunk_ids": outline.source_chunk_ids,
        "citations": _citations(chapter_chunks, outline.source_chunk_ids),
        "source_queries": outline.source_queries,
        "lessons": lessons,
        "quiz": quiz,
        "content_fingerprint": _content_fingerprint(outline.source_chunk_ids),
        "generation_status": "ready",
        "generation_metadata": totals,
    }


async def _plan_lesson_blocks_with_llm(
    provider: Any,
    chapter: OutlineChapter,
    lesson: OutlineLesson,
    chunks: list[dict[str, Any]],
    previous_summary: str,
) -> list[LessonBlockPlan]:
    if not chunks:
        return _fallback_block_plans(lesson, chunks)
    evidence = _chapter_evidence(chunks, max_chars=20_000)
    raw = await complete_text(
        provider,
        [
            LLMMessage(
                role="system",
                content=(
                    "Plan 2-5 substantial teaching blocks for exactly one source-grounded lesson. Every block must "
                    "directly teach the stated lesson title rather than summarize the chapter or a neighboring lesson. "
                    "Each block must cover a distinct supported teaching purpose and contain a precise retrieval query "
                    "using source terminology. Prefer blocks that support mechanisms, contrasts, consequences, or worked "
                    "examples. Use example or "
                    "procedure only when the evidence actually contains one. Plan an equation block when the evidence "
                    "contains a formula or matrix, and a table block when it contains structured rows or comparisons. "
                    "Do not invent content. Return JSON."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f"CHAPTER: {chapter.title}\nLESSON: {lesson.title}\nSTAGE: {lesson.lesson_stage}\n"
                    f"OBJECTIVES: {lesson.learning_objectives}\nPREVIOUS FOUNDATION: {previous_summary}\n\n"
                    f"EVIDENCE:\n{evidence}"
                ),
            ),
        ],
        json_schema=LessonBlockPlanBatch.model_json_schema(),
        temperature=0.15,
    )
    batch = LessonBlockPlanBatch.model_validate(_normalize_lesson_block_plan_payload(json.loads(raw), lesson))
    plans = [
        plan.model_copy(update={"title": plan.title or lesson.title})
        for plan in batch.blocks
        if _norm(plan.source_query)
    ]
    if not plans:
        raise ValueError("lesson block plan is empty")
    return plans


def _normalize_lesson_block_plan_payload(payload: Any, lesson: OutlineLesson) -> dict[str, Any]:
    """Accept common provider wrappers while preserving the local typed contract."""
    if not isinstance(payload, dict):
        return {"blocks": []}
    raw_blocks = next(
        (
            payload.get(key)
            for key in ("blocks", "teaching_blocks", "lesson_blocks", "block_plan", "plan")
            if isinstance(payload.get(key), list)
        ),
        [],
    )
    blocks: list[dict[str, str]] = []
    for item in raw_blocks:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or item.get("heading") or "").strip()
        source_query = str(
            item.get("source_query")
            or item.get("retrieval_query")
            or item.get("evidence_query")
            or item.get("query")
            or item.get("content_focus")
            or item.get("purpose")
            or title
            or lesson.title
        ).strip()
        blocks.append(
            {
                "block_type": _normalize_generated_block_type(
                    item.get("block_type") or item.get("type") or item.get("kind"),
                    fallback="markdown",
                ),
                "title": title or lesson.title,
                "source_query": source_query[:500] or lesson.title,
            }
        )
    return {"blocks": blocks}


def _fallback_block_plans(lesson: OutlineLesson, chunks: list[dict[str, Any]]) -> list[LessonBlockPlan]:
    base_query = " ".join(_clean_queries([lesson.title, *lesson.source_queries], limit=4)) or lesson.title
    if lesson.lesson_stage == "conclusion":
        return [LessonBlockPlan(block_type="summary", title="Key synthesis", source_query=base_query)]
    plans = [
        LessonBlockPlan(block_type="definition", title="Core explanation", source_query=f"{base_query} definition mechanism"),
        LessonBlockPlan(block_type="summary", title="Key relationships", source_query=f"{base_query} relationships conditions"),
    ]
    evidence_text = _norm(" ".join(str(chunk.get("text") or "")[:1400] for chunk in chunks))
    if any(_source_equations(chunk) for chunk in chunks):
        plans.append(LessonBlockPlan(block_type="equation", title="Source equation", source_query=f"{base_query} equation variables"))
    if any(_markdown_tables(str(chunk.get("text") or "")) for chunk in chunks):
        plans.append(LessonBlockPlan(block_type="table", title="Source table", source_query=f"{base_query} table comparison values"))
    if any(term in evidence_text for term in ("example", "worked", "case study", "for instance")):
        plans.insert(1, LessonBlockPlan(block_type="example", title="Source example", source_query=f"{base_query} example"))
    elif any(term in evidence_text for term in ("step", "procedure", "method", "algorithm")):
        plans.insert(1, LessonBlockPlan(block_type="procedure", title="Method", source_query=f"{base_query} procedure steps"))
    return plans


async def _retrieve_block_evidence(
    conversation_id: str,
    chapter: OutlineChapter,
    lesson: OutlineLesson,
    plan: LessonBlockPlan,
    lesson_chunks: list[dict[str, Any]],
    chapter_chunks: list[dict[str, Any]],
    *,
    expanded: bool,
) -> tuple[list[dict[str, Any]], int]:
    chapter_by_id = {str(chunk["id"]): chunk for chunk in chapter_chunks}
    query = " ".join(
        _clean_queries(
            [
                plan.source_query,
                plan.title,
                lesson.title,
                *lesson.source_queries,
                chapter.title,
                *("definition evidence assumptions mechanism worked example conditions contrast".split() if expanded else []),
            ],
            limit=18,
        )
    )
    retrieved_ids: list[str] = []
    try:
        hits = await get_retrieval_service().retrieve_for(
            conversation_id=conversation_id,
            user_message=query,
            output_type="text",
            source_file_ids=_dedupe(str(chunk.get("source_file_id") or "") for chunk in chapter_chunks),
            options={"top_k": 30 if expanded else 16, "hyde_enabled": False},
        )
        retrieved_ids = [str(hit.chunk_id) for hit in hits if str(hit.chunk_id) in chapter_by_id]
    except Exception:  # noqa: BLE001 - lexical ranking below is always available offline.
        pass
    pool = _dedupe_chunk_rows([*lesson_chunks, *(chapter_chunks if expanded else [])])
    pool_ids = {str(chunk["id"]) for chunk in pool}
    terms = set(_norm(query).split())
    ranked = sorted(
        (
            chunk
            for chunk in pool
            if not _looks_like_structure_only_chunk(chunk)
            and (not expanded or _chunk_supports_lesson(chunk, lesson))
        ),
        key=lambda chunk: (
            str(chunk["id"]) not in set(retrieved_ids),
            -len(terms.intersection(_norm(
                f"{chunk.get('metadata', {}).get('heading_path', '')} {chunk.get('text', '')[:2600]}"
            ).split())),
            int(chunk.get("chunk_index") or 0),
        ),
    )
    retrieved_rows = [
        chapter_by_id[item]
        for item in retrieved_ids
        if item in chapter_by_id
        and item in pool_ids
        and not _looks_like_structure_only_chunk(chapter_by_id[item])
        and (not expanded or _chunk_supports_lesson(chapter_by_id[item], lesson))
    ]
    selected = _dedupe_chunk_rows([*retrieved_rows, *ranked])
    return selected[:MAX_LESSON_EVIDENCE_CHUNKS], 1


async def _generate_lesson_block(
    provider: Any,
    chapter: OutlineChapter,
    lesson: OutlineLesson,
    plan: LessonBlockPlan,
    chunks: list[dict[str, Any]],
) -> DraftBlock:
    if not chunks:
        raise ValueError("block has no evidence")
    allowed_ids = {str(chunk["id"]) for chunk in chunks}
    evidence = _chapter_evidence(chunks, max_chars=24_000)
    raw = await complete_text(
        provider,
        [
            LLMMessage(
                role="system",
                content=(
                    "Write exactly one warm, rigorous teaching block using only the supplied evidence. The block must "
                    "directly teach the stated LESSON and BLOCK TITLE; do not summarize the whole chapter or drift into "
                    "another planned lesson. For prose blocks, write 2-4 connected paragraphs, usually 120-260 words, "
                    "explaining what the idea means, how it works, why it matters here, and a supported mechanism, "
                    "contrast, consequence, or example. Preserve equations, matrices, notation, dates, constraints, and caveats. "
                    "For an equation block, reproduce a source equation in $$...$$ and explain its supported symbols. "
                    "For a table block, produce a valid Markdown table using only source values. "
                    "Never use a parser plan marker as teaching content. Do not mention uploads or invent examples. "
                    "If the evidence does not support this exact lesson, return a warning block saying the support is weak. "
                    "Cite only supplied chunk IDs and return JSON."
                ),
            ),
            LLMMessage(
                role="user",
                content=(
                    f"CHAPTER: {chapter.title}\nLESSON: {lesson.title}\nBLOCK TYPE: {plan.block_type}\n"
                    f"BLOCK TITLE: {plan.title}\nRETRIEVAL PURPOSE: {plan.source_query}\n\nEVIDENCE:\n{evidence}"
                ),
            ),
        ],
        json_schema=DraftBlock.model_json_schema(),
        temperature=0.2,
    )
    block = DraftBlock.model_validate(_normalize_generated_block_payload(json.loads(raw), plan))
    valid_ids = _valid_ids(block.source_chunk_ids, allowed_ids)
    if not valid_ids or len(valid_ids) != len(_dedupe(block.source_chunk_ids)):
        raise ValueError("block cited evidence outside its retrieval scope")
    block.source_chunk_ids = valid_ids
    block.source_query = plan.source_query
    _validate_generated_block(block, plan, chunks, lesson.title)
    return block


def _normalize_generated_block_payload(payload: Any, plan: LessonBlockPlan) -> dict[str, Any]:
    """Repair provider-specific wrappers/aliases before strict Pydantic validation."""
    if not isinstance(payload, dict):
        payload = {}
    original_payload = payload
    for wrapper in (
        "lesson_block",
        "teaching_block",
        "content_block",
        "lesson_content",
        "lesson",
        "block",
        "result",
        "output",
        "data",
    ):
        if isinstance(payload.get(wrapper), dict):
            payload = payload[wrapper]
            break
    content = next(
        (
            payload.get(key)
            for key in ("content", "text", "body", "markdown", "explanation")
            if payload.get(key) not in (None, "")
        ),
        "",
    )
    source_values = (
        payload.get("source_chunk_ids")
        or payload.get("citations")
        or payload.get("chunk_ids")
        or payload.get("sources")
        or payload.get("references")
        or []
    )
    if not isinstance(source_values, list):
        source_values = [source_values]
    source_ids: list[str] = []
    for item in source_values:
        if isinstance(item, dict):
            value = item.get("chunk_id") or item.get("source_chunk_id") or item.get("id")
        else:
            value = item
        if str(value or "").strip():
            source_ids.append(str(value).strip())
    if not source_ids:
        source_ids = _collect_generated_chunk_ids(original_payload)
    normalized_content = _coerce_generated_block_content(content)
    if not normalized_content:
        normalized_content = _longest_generated_text(original_payload)
    return {
        "block_type": _normalize_generated_block_type(
            payload.get("block_type") or payload.get("type") or payload.get("kind"),
            fallback=plan.block_type,
        ),
        "title": str(payload.get("title") or payload.get("heading") or plan.title).strip()[:140],
        "content": normalized_content,
        "source_chunk_ids": _dedupe(source_ids),
        "source_query": str(payload.get("source_query") or payload.get("retrieval_query") or plan.source_query)[:500],
    }


def _normalize_generated_block_type(value: Any, *, fallback: str) -> str:
    key = _fold_text(str(value or fallback)).replace(" ", "_")
    aliases = {
        "explanation": "markdown",
        "text": "markdown",
        "paragraph": "markdown",
        "overview": "markdown",
        "worked_example": "example",
        "case_study": "example",
        "steps": "procedure",
        "method": "procedure",
        "caution": "warning",
        "takeaway": "summary",
        "formula": "equation",
    }
    normalized = aliases.get(key, key)
    allowed = {"markdown", "definition", "example", "procedure", "warning", "summary", "table", "equation", "diagram"}
    return normalized if normalized in allowed else fallback


def _coerce_generated_block_content(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        paragraphs: list[str] = []
        for key, item in value.items():
            text = (
                _longest_generated_text(item)
                if isinstance(item, (dict, list))
                else _coerce_generated_block_content(item)
            )
            if not text:
                continue
            label = re.sub(r"[_-]+", " ", str(key)).strip().capitalize()
            paragraphs.append(f"**{label}:** {text}" if label else text)
        return "\n\n".join(paragraphs)
    if isinstance(value, list):
        return "\n\n".join(filter(None, (_coerce_generated_block_content(item) for item in value)))
    return str(value or "").strip()


def _collect_generated_chunk_ids(value: Any) -> list[str]:
    found: list[str] = []
    if isinstance(value, dict):
        for item in value.values():
            found.extend(_collect_generated_chunk_ids(item))
    elif isinstance(value, list):
        for item in value:
            found.extend(_collect_generated_chunk_ids(item))
    elif isinstance(value, str) and re.fullmatch(r"chunk[-_][\w-]+", value.strip(), flags=re.IGNORECASE):
        found.append(value.strip())
    return _dedupe(found)


def _longest_generated_text(value: Any) -> str:
    candidates: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).casefold() in {
                "title", "heading", "type", "kind", "block_type", "source_query", "retrieval_query",
                "citations", "sources", "source_chunk_ids", "chunk_ids", "references",
            }:
                continue
            text = _coerce_generated_block_content(item)
            if len(text) >= 20:
                candidates.append(text)
    elif isinstance(value, list):
        candidates.extend(_longest_generated_text(item) for item in value)
    elif isinstance(value, str) and len(value.strip()) >= 20:
        candidates.append(value.strip())
    return max((item for item in candidates if item), key=len, default="")


def _validate_generated_block(
    block: DraftBlock,
    plan: LessonBlockPlan,
    chunks: list[dict[str, Any]],
    lesson_title: str,
) -> None:
    if _is_thin_teaching_block(block.block_type, block.content, block.title or lesson_title):
        raise ValueError("block is too thin to teach reliably")
    if block.block_type == "equation":
        generated = {_canonical_math(item) for item in _equations(block.content)}
        source = {_canonical_math(item) for chunk in chunks for item in _source_equations(chunk)}
        generated.discard("")
        source.discard("")
        if not generated or not any(
            candidate == evidence or candidate in evidence or evidence in candidate
            for candidate in generated
            for evidence in source
        ):
            raise ValueError("equation block is not traceable to a retrieved source equation")
    elif block.block_type == "table":
        tables = _markdown_tables(block.content)
        evidence = _norm(" ".join(str(chunk.get("text") or "") for chunk in chunks))
        cells = [
            _norm(cell)
            for table in tables
            for cell in [*table.get("headers", []), *(value for row in table.get("rows", []) for value in row)]
            if len(_norm(cell)) >= 2
        ]
        supported = sum(cell in evidence for cell in cells)
        if not tables or not cells or supported / len(cells) < 0.7:
            raise ValueError("table block contains values not supported by retrieved source rows")


def _source_equations(chunk: dict[str, Any]) -> list[str]:
    metadata = chunk.get("metadata") or {}
    stored = metadata.get("equations") or []
    if not isinstance(stored, list):
        stored = []
    return _dedupe([*_equations(str(chunk.get("text") or "")), *(str(item) for item in stored)])


def _canonical_math(value: str) -> str:
    text = re.sub(r"\\(?:begin|end)\{(?:[pbvBV]?matrix)\}", "", str(value or ""))
    return re.sub(r"[^a-z0-9+\-*/=^]", "", _norm(text))


def _fallback_generated_block(
    plan: LessonBlockPlan,
    chunks: list[dict[str, Any]],
) -> tuple[DraftBlock, str]:
    teaching = [chunk for chunk in chunks if not _looks_like_structure_only_chunk(chunk)]
    rich = _has_rich_teaching_material(teaching)
    content = (
        _fallback_content_for_block(
            plan.block_type,
            teaching,
            focus=f"{plan.title} {plan.source_query}",
            primary_focus=plan.title,
        )
        if rich
        else _insufficient_source_message()
    )
    status = "supported" if rich and content else "insufficient_source_material"
    source_ids = _dedupe(str(chunk["id"]) for chunk in teaching[:8])
    return (
        DraftBlock(
            block_type=plan.block_type if status == "supported" else "warning",
            title=plan.title,
            content=content or _insufficient_source_message(),
            source_chunk_ids=source_ids,
            source_query=plan.source_query,
        ),
        status,
    )


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
    lesson_evidence: dict[str, list[dict[str, Any]]] = {}
    lesson_retrieval_count = 0
    weak_support_count = 0
    for planned_lesson in outline.lessons:
        selected, retrieval_count = await _retrieve_lesson_evidence(
            conversation_id,
            outline,
            planned_lesson,
            chapter_chunks,
        )
        lesson_evidence[_norm(planned_lesson.title)] = selected
        lesson_retrieval_count += retrieval_count
        if not _has_rich_teaching_material(selected):
            weak_support_count += 1
    allowed_ids = {
        chunk["id"]
        for selected in lesson_evidence.values()
        for chunk in selected
    }
    if not allowed_ids:
        raise ValueError("chapter has no teachable evidence")
    lesson_plan = "\n".join(
        f"- {lesson.title}: stage={lesson.lesson_stage}; role={lesson.pedagogical_role}; "
        f"prerequisites={lesson.prerequisite_lesson_titles}; "
        f"chunk_ids={lesson.source_chunk_ids}; objectives={lesson.learning_objectives}"
        for lesson in outline.lessons
    )
    evidence = _planned_lesson_evidence(outline, lesson_evidence)
    system = (
        "Write a warm, rigorous chapter from only the supplied, lesson-scoped evidence. Preserve equations, matrices, "
        "chemical notation, dates, tables, and technical notation. Preserve the exact lesson titles and order from the "
        "lesson plan. Every block must cite one or more chunk IDs shown under that same lesson; never cite evidence from "
        "another lesson merely because it is in the chapter. Use definition or markdown for developed explanations, "
        "example for grounded worked examples, procedure for supported steps, equation/table when the evidence contains "
        "them, warning when the evidence is insufficient, and summary for synthesis. Rich prose blocks must contain "
        "several connected explanatory sentences, not title repetition or plan markers. Put a focused retrieval phrase "
        "in source_query for each block. Do not invent examples, dates, equations, or claims. Return valid JSON."
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
        planned = next(
            (item for item in outline.lessons if _norm(item.title) == _norm(lesson.title)),
            outline.lessons[min(lesson_index, len(outline.lessons) - 1)],
        )
        planned_chunks = lesson_evidence.get(_norm(planned.title), [])
        planned_ids = {chunk["id"] for chunk in planned_chunks}
        lesson_ids = _valid_ids(lesson.source_chunk_ids, planned_ids)
        lesson_chunks = [chunk_by_id[item] for item in lesson_ids]
        if not lesson_chunks:
            raise ValueError("lesson synthesis is not grounded")
        blocks = []
        for block_index, block in enumerate(lesson.blocks):
            source_ids = _valid_ids(block.source_chunk_ids, set(lesson_ids))
            if not source_ids:
                raise ValueError("lesson block is not grounded")
            block_chunks = [chunk_by_id[item] for item in source_ids]
            content = block.content
            validation_status = "supported"
            if _is_thin_teaching_block(block.block_type, content, block.title or lesson.title):
                if _has_rich_teaching_material(block_chunks):
                    content = _fallback_content_for_block(block.block_type, block_chunks)
                else:
                    content = _insufficient_source_message()
                    validation_status = "insufficient_source_material"
            blocks.append(
                _block(
                    lesson.title,
                    block_index,
                    block.block_type,
                    block.title or lesson.title,
                    content,
                    block_chunks,
                    validation_status=validation_status,
                    source_query=block.source_query,
                )
            )
        blocks.extend(
            _special_blocks(
                lesson.title,
                lesson_chunks,
                start_index=len(blocks),
                existing_blocks=blocks,
            )
        )
        blocks = _dedupe_lesson_blocks(blocks)
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
                "lesson_stage": outline.lessons[min(lesson_index, len(outline.lessons) - 1)].lesson_stage,
                "prerequisite_lesson_ids": [lessons[-1]["id"]] if lessons else [],
                "source_chunk_ids": lesson_ids,
                "citations": _citations(lesson_chunks, lesson_ids),
                "blocks": blocks[:MAX_BLOCKS_PER_LESSON],
                "support_status": "supported" if _has_rich_teaching_material(lesson_chunks) else "insufficient_source_material",
                "source_queries": planned.source_queries,
                "content_fingerprint": _content_fingerprint(lesson_ids),
                "generation_status": "ready",
            }
        )
    if not lessons:
        raise ValueError("chapter synthesis returned no lessons")
    lessons = _align_generated_lessons_to_outline(
        lessons,
        outline,
        chapter_chunks,
        conversation_id,
        chapter_index,
    )
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
        "source_queries": outline.source_queries,
        "lessons": lessons,
        "quiz": quiz,
        "content_fingerprint": _content_fingerprint(outline.source_chunk_ids),
        "generation_status": "ready",
        "generation_metadata": {
            "chapter_retrieval_count": 1,
            "lesson_retrieval_count": lesson_retrieval_count,
            "weak_support_lesson_count": weak_support_count,
        },
    }


def _align_generated_lessons_to_outline(
    generated: list[dict[str, Any]],
    outline: OutlineChapter,
    chapter_chunks: list[dict[str, Any]],
    conversation_id: str,
    chapter_index: int,
) -> list[dict[str, Any]]:
    """Publish the exact planned subchapter arc even when synthesis omits an item."""
    chunk_by_id = {chunk["id"]: chunk for chunk in chapter_chunks}
    generated_by_title = {_norm(lesson.get("title", "")): lesson for lesson in generated}
    aligned: list[dict[str, Any]] = []
    for lesson_index, planned in enumerate(outline.lessons):
        lesson = generated_by_title.get(_norm(planned.title))
        planned_chunks = [chunk_by_id[item] for item in planned.source_chunk_ids if item in chunk_by_id]
        if lesson is None:
            lesson = _single_fallback_lesson(
                conversation_id,
                chapter_index,
                {
                    "title": planned.title,
                    "chunks": planned_chunks or chapter_chunks[:1],
                    "lesson_stage": planned.lesson_stage,
                },
            )
            if planned.lesson_stage in {"introduction", "conclusion"}:
                lesson["blocks"] = _boundary_lesson_blocks(
                    planned.title,
                    planned_chunks or chapter_chunks[:1],
                    planned.lesson_stage,
                )
        source_ids = _dedupe([*lesson.get("source_chunk_ids", []), *planned.source_chunk_ids])
        source_chunks = [chunk_by_id[item] for item in source_ids if item in chunk_by_id]
        lesson.update(
            {
                "id": _stable_id(conversation_id, "lesson", chapter_index, lesson_index, planned.title),
                "title": planned.title,
                "order_index": lesson_index,
                "summary": lesson.get("summary") or planned.summary,
                "learning_objectives": planned.learning_objectives or lesson.get("learning_objectives", []),
                "pedagogical_role": planned.pedagogical_role,
                "sequencing_reason": planned.sequencing_reason,
                "lesson_stage": planned.lesson_stage,
                "prerequisite_lesson_ids": [aligned[-1]["id"]] if aligned else [],
                "source_chunk_ids": source_ids,
                "citations": _citations(source_chunks, source_ids),
                "source_queries": planned.source_queries,
                "support_status": "supported" if _has_rich_teaching_material(source_chunks) else "insufficient_source_material",
                "content_fingerprint": _content_fingerprint(source_ids),
                "generation_status": "ready",
            }
        )
        aligned.append(lesson)
    return aligned


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
        fallback["generation_metadata"] = {"fallback_question_count": len(fallback.get("questions", []))}
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
        questions: list[dict[str, Any]] = []
        seen_prompts: set[str] = set()
        for question in draft.questions:
            if question.source_chunk_id not in allowed_ids:
                continue
            if len({_norm(option) for option in question.options}) != 4:
                continue
            prompt_key = _norm(question.prompt)
            if not prompt_key or prompt_key in seen_prompts:
                continue
            seen_prompts.add(prompt_key)
            questions.append(
                _quiz_question(
                    title,
                    len(questions),
                    question.prompt,
                    question.options,
                    question.correct_index,
                    question.explanation,
                    next(chunk for chunk in chunks if chunk["id"] == question.source_chunk_id),
                )
            )
            if len(questions) >= count:
                break
        generated_count = len(questions)
        for fallback_question in fallback.get("questions", []):
            if len(questions) >= count:
                break
            prompt_key = _norm(fallback_question.get("prompt", ""))
            if not prompt_key or prompt_key in seen_prompts:
                continue
            seen_prompts.add(prompt_key)
            questions.append(_reindex_quiz_question(fallback_question, title, len(questions)))
        if len(questions) < count:
            for fallback_question in fallback.get("questions", []):
                if len(questions) >= count:
                    break
                questions.append(_reindex_quiz_question(fallback_question, title, len(questions)))
        return {
            "id": _stable_id(title, scope, "quiz"),
            "title": title,
            "scope": scope,
            "questions": questions,
            "pass_score": PASS_SCORE,
            "generation_metadata": {"fallback_question_count": max(0, count - generated_count)},
        }
    except Exception:  # noqa: BLE001 - deterministic grounded questions are the assessment recovery boundary.
        fallback["generation_metadata"] = {"fallback_question_count": len(fallback.get("questions", []))}
        return fallback


def _reindex_quiz_question(question: dict[str, Any], title: str, index: int) -> dict[str, Any]:
    cloned = json.loads(json.dumps(question))
    old_correct = str(cloned.get("correct_option_id") or "")
    correct_index = next(
        (option_index for option_index, option in enumerate(cloned.get("options", [])) if option.get("id") == old_correct),
        0,
    )
    question_id = _stable_id(title, "question", index)
    cloned["id"] = question_id
    for option_index, option in enumerate(cloned.get("options", [])):
        option["id"] = _stable_id(question_id, "option", option_index)
    options = cloned.get("options", [])
    if options:
        cloned["correct_option_id"] = options[min(correct_index, len(options) - 1)]["id"]
    return cloned


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
    for chapter_index, group in enumerate(groups):
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
                "source_queries": _clean_queries(
                    [group["title"], *(_concept_for_chunk(chunk) for chunk in group["chunks"])],
                    limit=10,
                ),
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
            "quality_pipeline_version": QUALITY_PIPELINE_VERSION,
            "build_profile": "fast",
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
            **_fallback_quality_counters(chapters),
        },
    }


def _fallback_quality_counters(chapters: list[dict[str, Any]]) -> dict[str, int]:
    lessons = [lesson for chapter in chapters for lesson in chapter.get("lessons", [])]
    blocks = [block for lesson in lessons for block in lesson.get("blocks", [])]
    return {
        "chapter_retrieval_count": 0,
        "lesson_retrieval_count": 0,
        "block_retrieval_count": 0,
        "block_retry_count": 0,
        "fallback_block_count": len(blocks),
        "unsupported_block_count": sum(
            block.get("validation_status") == "insufficient_source_material" for block in blocks
        ),
        "weak_support_lesson_count": sum(
            lesson.get("support_status") == "insufficient_source_material" for lesson in lessons
        ),
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
            "quality_pipeline_version": QUALITY_PIPELINE_VERSION,
            "build_profile": "improved",
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
                source_queries=chapter.get("source_queries", []),
                lessons=[
                    OutlineLesson(
                        title=lesson["title"],
                        summary=str(lesson.get("summary", ""))[:500],
                        learning_objectives=lesson.get("learning_objectives", []),
                        source_chunk_ids=lesson.get("source_chunk_ids", []),
                        pedagogical_role=lesson.get("pedagogical_role", "core_concept"),
                        sequencing_reason=lesson.get("sequencing_reason", ""),
                        lesson_stage=lesson.get("lesson_stage", "content"),
                        source_queries=lesson.get("source_queries", []),
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
            **_fallback_quality_counters(chapters),
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
    haystack = _fold_text(f"{source} {graph_labels}")
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
    anchor_sets: dict[str, tuple[str, ...]] = {
        "historical": ("history", "historical", "histoire", "dynasty", "empire", "colonial", "historiography"),
        "chemistry": ("chemistry", "chimie", "chemical", "chimique", "atom", "atome", "molecule", "stoichiometry", "oxidation", "periodic table"),
        "physics": ("physics", "physique", "newton", "thermodynamics", "quantum", "electromagnetic", "relativity"),
        "mathematics": ("mathematics", "mathematique", "theorem", "theoreme", "proof", "lemma", "calculus", "algebra", "geometry", "topology"),
        "life_science": ("biology", "biologie", "biological", "cell", "genetics", "organism", "protein", "enzyme", "dna", "rna"),
        "procedural": ("step by step", "workflow", "installation", "configuration", "protocol", "tutorial", "laboratory method", "troubleshooting"),
    }
    anchor_counts = {
        architecture: sum(len(re.findall(rf"(?<!\w){re.escape(keyword)}(?!\w)", haystack)) for keyword in keywords)
        for architecture, keywords in anchor_sets.items()
    }
    conceptual_signals = (
        "recommendation", "recommender", "systeme de recommandation", "machine learning",
        "information retrieval", "software", "algorithm", "computer science",
    )
    conceptual_count = sum(
        len(re.findall(rf"(?<!\w){re.escape(keyword)}(?!\w)", haystack))
        for keyword in conceptual_signals
    )
    if conceptual_count >= max(4, max(anchor_counts.values(), default=0) * 2):
        return "conceptual"
    for architecture, anchor_count in anchor_counts.items():
        if anchor_count == 0:
            scores[architecture] = 0
        else:
            scores[architecture] += min(8, anchor_count * 2)
    dated_lines = sum(bool(re.search(r"\b(?:\d{3,4}(?:\s*(?:BCE|BC|CE|AD))?)\b", chunk.get("text", ""), re.I)) for chunk in chunks)
    if dated_lines >= max(3, len(chunks) // 3):
        scores["historical"] += 4
    chemical_equations = sum(any(_looks_chemical(line) for line in str(chunk.get("text", "")).splitlines()) for chunk in chunks)
    if anchor_counts["chemistry"]:
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
    normalized_title = _fold_text(title)
    normalized = _fold_text(f"{title} {text[:4000]}")
    if any(term in normalized_title for term in ("what is", "qu est ce", "definition", "terminology", "overview", "introduction")):
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
    if any(term in normalized_title for term in ("fundamental", "fondement", "foundation", "prerequisite", "prerequis", "basic", "bases", "background")):
        return "foundation"
    if any(term in normalized_title for term in ("equation", "formula", "matrix", "mathematical", "derivation")):
        return "mathematical_formulation"
    if any(term in normalized_title for term in ("collaborative filtering", "filtrage collaboratif", "baseline", "standard method", "nearest neighbor", "voisinage")):
        return "standard_method"
    if any(term in normalized_title for term in ("content based", "filtrage base sur le contenu", "specialized", "variant")):
        return "specialized_method"
    if any(term in normalized_title for term in ("hybrid", "hybride", "integration", "combined", "fusion", "ensemble")):
        return "integration"
    if any(term in normalized_title for term in ("deep learning", "apprentissage profond", "neural", "advanced", "avance", "transformer", "representation learning")):
        return "advanced"
    if any(term in normalized_title for term in ("metric", "metrique", "evaluation", "validation", "benchmark", "precision", "recall", "rappel", "rmse", "mae", "ndcg")):
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
                    _lesson_stage_rank(getattr(by_title[key], "lesson_stage", "content")),
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


def _lesson_stage_rank(stage: str) -> int:
    return {"introduction": 0, "content": 1, "conclusion": 2}.get(stage, 1)


def _ensure_outline_chapter_arcs(outline: CourseOutline) -> CourseOutline:
    """Guarantee introduction -> grounded content -> conclusion for every chapter."""
    for chapter in outline.chapters:
        for lesson in chapter.lessons:
            normalized = _norm(lesson.title)
            if lesson.lesson_stage == "content" and any(
                term in normalized for term in ("introduction", "chapter overview", "orientation", "learning goals")
            ):
                lesson.lesson_stage = "introduction"
            elif lesson.lesson_stage == "content" and any(
                term in normalized for term in ("conclusion", "summary and transition", "chapter summary", "chapter review", "recap")
            ):
                lesson.lesson_stage = "conclusion"

        introductions = [lesson for lesson in chapter.lessons if lesson.lesson_stage == "introduction"]
        conclusions = [lesson for lesson in chapter.lessons if lesson.lesson_stage == "conclusion"]
        content = [lesson for lesson in chapter.lessons if lesson.lesson_stage == "content"]
        if not content:
            source = next(iter(chapter.lessons), None)
            if source is not None:
                source.lesson_stage = "content"
                content = [source]
                introductions = [lesson for lesson in introductions if lesson is not source]
                conclusions = [lesson for lesson in conclusions if lesson is not source]

        source_ids = _dedupe(
            [*chapter.source_chunk_ids, *(item for lesson in chapter.lessons for item in lesson.source_chunk_ids)]
        )
        if not source_ids:
            continue
        intro_role: PedagogicalRole = "context" if outline.architecture_type == "historical" else "definition"
        introduction = introductions[0] if introductions else OutlineLesson(
            title=f"Introduction to {chapter.title}",
            summary=f"Orient the learner to the purpose, vocabulary, and prerequisites of {chapter.title}.",
            learning_objectives=[f"Identify the purpose and prerequisites of {chapter.title}"],
            source_chunk_ids=source_ids,
            pedagogical_role=intro_role,
            sequencing_reason="Introduces this chapter before its detailed content.",
            lesson_stage="introduction",
            source_queries=[chapter.title, f"{chapter.title} definitions prerequisites overview"],
        )
        conclusion = conclusions[-1] if conclusions else OutlineLesson(
            title=f"{chapter.title}: Summary and transition",
            summary=f"Synthesize {chapter.title} and connect it to the foundation needed next.",
            learning_objectives=[f"Synthesize the key ideas in {chapter.title}"],
            source_chunk_ids=source_ids,
            pedagogical_role="synthesis",
            sequencing_reason="Consolidates this chapter and prepares the next foundation.",
            lesson_stage="conclusion",
            source_queries=[chapter.title, f"{chapter.title} synthesis relationships transition"],
        )
        introduction.lesson_stage = "introduction"
        conclusion.lesson_stage = "conclusion"
        middle = [
            *[lesson for lesson in introductions[1:] if lesson is not conclusion],
            *content,
            *[lesson for lesson in conclusions[:-1] if lesson is not introduction],
        ]
        for lesson in middle:
            lesson.lesson_stage = "content"
        chapter.lessons = [introduction, *middle, conclusion]
        chapter.source_chunk_ids = source_ids
    return outline


def _validate_plan_with_graph(
    outline: CourseOutline,
    chunks: list[dict[str, Any]],
    graph: dict[str, Any] | None,
    *,
    preserve_structure: bool = False,
) -> CourseOutline:
    """Repair coverage and add graph-derived prerequisites without replacing the saved plan."""
    outline = CourseOutline.model_validate(outline.model_dump(mode="json"))
    allowed_ids = {chunk["id"] for chunk in chunks}
    for chapter in outline.chapters:
        chapter.source_chunk_ids = (
            [] if preserve_structure else _valid_ids(chapter.source_chunk_ids, allowed_ids)
        )
        plan_queries = _lesson_queries_from_source_plan(chapter, chunks)
        for lesson in chapter.lessons:
            # The Markdown plan is authoritative, but its evidence IDs are derived state.
            # Rebind them so a cached partial build cannot leave every chunk on lesson one.
            lesson.source_chunk_ids = (
                [] if preserve_structure else _valid_ids(lesson.source_chunk_ids, allowed_ids)
            )
            if preserve_structure:
                lesson.source_queries = _clean_queries(
                    [lesson.title, *plan_queries.get(_norm(lesson.title), [])],
                    limit=8,
                )
        chapter.source_chunk_ids = _dedupe(
            [*chapter.source_chunk_ids, *(item for lesson in chapter.lessons for item in lesson.source_chunk_ids)]
        )
    outline = _ensure_outline_coverage(outline, chunks, graph)
    outline = _ensure_outline_source_queries(outline, chunks, graph)
    if preserve_structure:
        return outline
    _add_graph_prerequisites(outline, graph)
    outline = _ensure_outline_chapter_arcs(outline)
    return _sequence_outline(outline)


def _lesson_queries_from_source_plan(
    chapter: OutlineChapter,
    chunks: list[dict[str, Any]],
) -> dict[str, list[str]]:
    """Recover nested agenda terms (for example, Model-Based -> SVD) for content binding."""
    segments: list[str] = []
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        heading = _fold_text(metadata.get("section_title") or metadata.get("heading_path") or "")
        if not any(
            marker in heading
            for marker in (
                "plan de la seance",
                "agenda",
                "course outline",
                "table of contents",
                "table des matieres",
            )
        ):
            continue
        segments.extend(
            match.strip()
            for match in re.findall(
                r"(?ms)^\s*\d+[.)]\s+(.+?)(?=^\s*\d+[.)]\s+|\Z)",
                str(chunk.get("text") or ""),
            )
            if match.strip()
        )

    queries: dict[str, list[str]] = {}
    for lesson in chapter.lessons:
        ranked = sorted(
            (
                (_title_term_coverage(segment.split("\n", 1)[0], lesson.title), segment)
                for segment in segments
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if ranked and ranked[0][0] >= 0.45:
            queries[_norm(lesson.title)] = [re.sub(r"\s+", " ", ranked[0][1])[:500]]
    return queries


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
    source_targets = _source_file_chapter_targets(outline, chunks)
    for chunk_id in [item for item in chunk_by_id if item not in assigned]:
        chunk = chunk_by_id[chunk_id]
        best: tuple[float, OutlineChapter, OutlineLesson] | None = None
        target_chapter = source_targets.get(str(chunk.get("source_file_id") or ""))
        candidate_chapters = [target_chapter] if target_chapter is not None else outline.chapters
        for chapter in candidate_chapters:
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


def _source_file_chapter_targets(
    outline: CourseOutline,
    chunks: list[dict[str, Any]],
) -> dict[str, OutlineChapter]:
    """Map single-unit and supplemental files to their source-matching chapter before lesson ranking."""
    by_file: dict[str, list[dict[str, Any]]] = {}
    for chunk in chunks:
        by_file.setdefault(str(chunk.get("source_file_id") or ""), []).append(chunk)
    targets: dict[str, OutlineChapter] = {}
    for source_file_id, rows in by_file.items():
        primary_titles = _dedupe(
            str(row.get("metadata", {}).get("course_unit_title") or "").strip()
            for row in rows
            if str(row.get("metadata", {}).get("course_unit_role") or "") == "primary"
            and str(row.get("metadata", {}).get("course_unit_title") or "").strip()
        )
        if len(primary_titles) == 1:
            ranked = sorted(
                (
                    (_title_term_coverage(primary_titles[0], chapter.title), chapter)
                    for chapter in outline.chapters
                ),
                key=lambda item: item[0],
                reverse=True,
            )
            if ranked and ranked[0][0] >= 0.45:
                targets[source_file_id] = ranked[0][1]
                continue
        source_filename = str(rows[0].get("source_filename") or "") if rows else ""
        if not re.search(
            r"\b(?:guide|appendix|annex|reference|references|supplement|syllabus|handbook|workbook)\b",
            _norm(source_filename.replace("_", " ")),
        ):
            continue
        source_terms = _meaningful_title_terms(
            " ".join(
                f"{row.get('metadata', {}).get('heading_path', '')} {str(row.get('text') or '')[:1600]}"
                for row in rows
            )
        )
        domain_terms = _meaningful_title_terms(outline.title)
        source_terms -= domain_terms
        root_terms = _meaningful_title_terms(
            " ".join(
                str((row.get("metadata", {}).get("heading_path_list") or [""])[0])
                for row in rows
            )
        ) - domain_terms
        chapter_terms = [
            _meaningful_title_terms(
                " ".join([chapter.title, *(lesson.title for lesson in chapter.lessons)])
            ) - domain_terms
            for chapter in outline.chapters
        ]
        term_document_frequency = Counter(term for terms in chapter_terms for term in terms)
        ranked = sorted(
            (
                (
                    sum(
                        1.0 / term_document_frequency[term]
                        for term in source_terms.intersection(chapter_terms[chapter_index])
                    )
                    + 3.0 * sum(
                        1.0 / term_document_frequency[term]
                        for term in root_terms.intersection(chapter_terms[chapter_index])
                    ),
                    chapter,
                )
                for chapter_index, chapter in enumerate(outline.chapters)
            ),
            key=lambda item: item[0],
            reverse=True,
        )
        if ranked and ranked[0][0] >= 2:
            targets[source_file_id] = ranked[0][1]
    return targets


def _title_term_coverage(left: str, right: str) -> float:
    left_terms = _meaningful_title_terms(left)
    right_terms = _meaningful_title_terms(right)
    if not left_terms or not right_terms:
        return 0.0
    return len(left_terms.intersection(right_terms)) / min(len(left_terms), len(right_terms))


def _meaningful_title_terms(value: str) -> set[str]:
    raw_stopwords = {
        "the", "and", "for", "from", "with", "into", "course", "chapter", "lesson",
        "de", "des", "du", "la", "le", "les", "et", "en", "dans", "sur", "aux", "une", "un",
        "avec", "sans", "nous", "vous", "avons", "pourquoi", "comment", "que", "qui", "quoi",
        "plus", "moins", "pas", "autres", "autre", "deux", "delà", "au", "par", "pour",
        "semaine", "week", "lecture", "module", "chapitre", "unit", "unite",
    }
    stopwords = {_fold_text(term) for term in raw_stopwords}
    return {
        term
        for term in _fold_text(value).split()
        if len(term) >= 3 and term not in stopwords and not term.isdigit()
    }


def _chunk_lesson_affinity(
    chunk: dict[str, Any],
    lesson: OutlineLesson,
    chunk_by_id: dict[str, dict[str, Any]],
    graph_neighbors: set[str],
) -> float:
    metadata = chunk.get("metadata", {})
    chunk_terms = set(_fold_text(
        f"{metadata.get('heading_path', '')} {' '.join(metadata.get('key_concepts') or [])} {chunk.get('text', '')[:1800]}"
    ).split())
    lesson_terms = set(_fold_text(
        f"{lesson.title} {lesson.summary} {' '.join(lesson.learning_objectives)} {' '.join(lesson.source_queries)}"
    ).split())
    score = float(len(chunk_terms & lesson_terms) * 2)
    explicit_heading = str(metadata.get("subchapter_title") or metadata.get("section_title") or "")
    heading_coverage = _title_term_coverage(explicit_heading, lesson.title)
    if heading_coverage >= 0.75:
        score += 24.0
    elif heading_coverage >= 0.45:
        score += 10.0
    binding_terms = _meaningful_title_terms(" ".join([lesson.title, *lesson.source_queries]))
    heading_terms = _meaningful_title_terms(explicit_heading)
    body_terms = _meaningful_title_terms(str(chunk.get("text") or "")[:1800])
    score += 12.0 * len(binding_terms.intersection(heading_terms)) / max(1, len(binding_terms))
    score += 6.0 * len(binding_terms.intersection(body_terms)) / max(1, len(binding_terms))
    lesson_key = _fold_text(lesson.title)
    heading_key = _fold_text(explicit_heading)
    if (
        (
            re.search(r"\bother\b.*\barchitectures?\b", lesson_key)
            or re.search(r"\bautres\b.*\barchitectures?\b", lesson_key)
        )
        and re.search(
            r"\b(?:auto[ -]?encodeurs?|auto[ -]?encoders?|cnn|transformers?|gan|graph neural)\b",
            heading_key,
        )
    ):
        # Named model families are the concrete evidence for a source-plan item such as
        # "Overview of other neural architectures", even when the heading omits that umbrella phrase.
        score += 30.0
    assigned_chunks = [chunk_by_id[item] for item in lesson.source_chunk_ids if item in chunk_by_id]
    if any(item.get("source_file_id") == chunk.get("source_file_id") for item in assigned_chunks):
        score += 0.25
    chunk_root = (metadata.get("heading_path_list") or [""])[0]
    if any((item.get("metadata", {}).get("heading_path_list") or [None])[0] == chunk_root for item in assigned_chunks):
        score += 0.5
    score += len(graph_neighbors.intersection(lesson.source_chunk_ids))
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


def _ensure_outline_source_queries(
    outline: CourseOutline,
    chunks: list[dict[str, Any]],
    graph: dict[str, Any] | None,
) -> CourseOutline:
    """Make the saved outline an executable retrieval contract."""
    chunk_by_id = {str(chunk["id"]): chunk for chunk in chunks}
    graph_terms: dict[str, list[str]] = {}
    for node in (graph or {}).get("nodes", []):
        label = _clean_title(str(node.get("label") or ""))
        if not label or node.get("node_type") in {"course", "file", "document", "chunk"}:
            continue
        for chunk_id in node.get("source_chunk_ids") or []:
            graph_terms.setdefault(str(chunk_id), []).append(label)

    for chapter in outline.chapters:
        chapter_hints = _source_query_hints(chapter.source_chunk_ids, chunk_by_id, graph_terms)
        chapter.source_queries = _clean_queries(
            [chapter.title, *chapter.source_queries, *chapter_hints, *chapter.learning_objectives],
            limit=10,
        )
        for lesson in chapter.lessons:
            lesson_hints = _source_query_hints(lesson.source_chunk_ids, chunk_by_id, graph_terms)
            lesson.source_queries = _clean_queries(
                [
                    lesson.title,
                    *lesson.source_queries,
                    *lesson_hints,
                    *lesson.learning_objectives,
                    f"{chapter.title} {lesson.pedagogical_role}",
                ],
                limit=8,
            )
    return outline


def _source_query_hints(
    source_chunk_ids: list[str],
    chunk_by_id: dict[str, dict[str, Any]],
    graph_terms: dict[str, list[str]],
) -> list[str]:
    hints: list[str] = []
    for chunk_id in source_chunk_ids:
        chunk = chunk_by_id.get(str(chunk_id))
        if chunk is None:
            continue
        metadata = chunk.get("metadata", {})
        path = metadata.get("heading_path_list") or []
        if isinstance(path, list):
            hints.extend(str(item) for item in path[-2:] if str(item).strip())
        elif metadata.get("heading_path"):
            hints.append(str(metadata["heading_path"]))
        hints.extend(str(item) for item in metadata.get("key_concepts") or [])
        hints.extend(graph_terms.get(str(chunk_id), []))
    return hints


def _clean_queries(values: list[str], *, limit: int) -> list[str]:
    queries: list[str] = []
    seen: set[str] = set()
    for value in values:
        query = re.sub(r"\s+", " ", str(value or "")).strip()[:500]
        key = _norm(query)
        if len(key) < 2 or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= limit:
            break
    return queries


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
    file_order = _dedupe(str(chunk.get("source_file_id") or "") for chunk in chunks)
    if len(file_order) > 1:
        grouped_by_file: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            file_id = str(chunk.get("source_file_id") or chunk.get("source_filename") or "source")
            grouped_by_file.setdefault(file_id, {"title": "", "chunks": []})["chunks"].append(chunk)
        groups = list(grouped_by_file.values())
        for group in groups:
            group["title"] = _course_unit_title(group["chunks"])
    else:
        groups = _chapter_groups_from_single_source(chunks)

    for group in groups:
        role = _infer_pedagogical_role(
            group["title"],
            " ".join(str(chunk.get("text", ""))[:800] for chunk in group["chunks"][:3]),
            architecture=architecture,
        )
        group["pedagogical_role"] = role
        group["sequencing_reason"] = _sequencing_reason(role, architecture)
        group["architecture_type"] = architecture
    return _sort_groups_with_graph(groups, graph, architecture)


def _chapter_groups_from_single_source(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    return list(grouped.values())


def _course_unit_title(chunks: list[dict[str, Any]]) -> str:
    candidates: list[str] = []
    source_filename = str(chunks[0].get("source_filename") or "Course unit") if chunks else "Course unit"
    filename_key = _norm(source_filename.rsplit(".", 1)[0].replace("_", " "))
    for chunk in chunks:
        metadata = chunk.get("metadata", {})
        path = metadata.get("heading_path_list") or []
        values = [*(path if isinstance(path, list) else []), metadata.get("section_title")]
        for value in values:
            title = _clean_title(str(value or ""))
            key = _fold_text(title)
            if not title or key == _fold_text(filename_key) or key == _fold_text(source_filename):
                continue
            if any(
                term in key
                for term in (
                    "plan de la seance", "agenda", "outline", "table des matieres",
                    "references", "bibliographie", "lectures recommandees", "conclusion",
                )
            ):
                continue
            if title not in candidates:
                candidates.append(title)
    for title in candidates:
        key = _fold_text(title)
        if any(term in key for term in ("semaine", "week", "chapter", "chapitre", "unit")):
            return title
    return candidates[0] if candidates else _clean_title(source_filename.rsplit(".", 1)[0].replace("_", " "))


def _fold_text(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", str(value or ""))
    ascii_text = "".join(char for char in normalized if not unicodedata.combining(char))
    return re.sub(r"\W+", " ", ascii_text.casefold()).strip()


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
                dependencies[source_group].update(
                    target_group
                    for target_group in target_groups - {source_group}
                    if _pedagogical_role_rank(groups[target_group]["pedagogical_role"], architecture)
                    <= _pedagogical_role_rank(groups[source_group]["pedagogical_role"], architecture)
                )
        elif relation in {"precedes", "causes", "leads_to", "foundation_for"}:
            for target_group in target_groups:
                dependencies[target_group].update(
                    source_group
                    for source_group in source_groups - {target_group}
                    if _pedagogical_role_rank(groups[source_group]["pedagogical_role"], architecture)
                    <= _pedagogical_role_rank(groups[target_group]["pedagogical_role"], architecture)
                )

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
    max_content_lessons = max(1, MAX_LESSONS_PER_CHAPTER - 2)
    if len(items) > max_content_lessons:
        base = items[:max_content_lessons]
        for index, (_, overflow_chunks) in enumerate(items[max_content_lessons:]):
            base[(max_content_lessons - 1) - (index % 2)][1].extend(overflow_chunks)
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
                "lesson_stage": "content",
                "prerequisite_lesson_ids": [lessons[-1]["id"]] if lessons else [],
                "source_chunk_ids": source_chunk_ids,
                "citations": _citations(section_chunks, source_chunk_ids),
                "source_queries": _clean_queries(
                    [title, *(_concept_for_chunk(chunk) for chunk in section_chunks)],
                    limit=8,
                ),
                "support_status": "supported" if _has_rich_teaching_material(section_chunks) else "insufficient_source_material",
                "blocks": _lesson_blocks(title, section_chunks),
                "content_fingerprint": _content_fingerprint(source_chunk_ids),
                "generation_status": "ready",
            }
        )
    content_lessons = lessons or [_single_fallback_lesson(conversation_id, chapter_index, group)]
    for lesson in content_lessons:
        lesson["lesson_stage"] = "content"
    introduction = _fallback_boundary_lesson(
        conversation_id,
        chapter_index,
        group,
        stage="introduction",
    )
    conclusion = _fallback_boundary_lesson(
        conversation_id,
        chapter_index,
        group,
        stage="conclusion",
    )
    chapter_lessons = [introduction, *content_lessons, conclusion]
    for lesson_index, lesson in enumerate(chapter_lessons):
        lesson["order_index"] = lesson_index
        lesson["prerequisite_lesson_ids"] = [chapter_lessons[lesson_index - 1]["id"]] if lesson_index else []
    return chapter_lessons


def _fallback_boundary_lesson(
    conversation_id: str,
    chapter_index: int,
    group: dict[str, Any],
    *,
    stage: Literal["introduction", "conclusion"],
) -> dict[str, Any]:
    chapter_title = str(group["title"])
    title = (
        f"Introduction to {chapter_title}"
        if stage == "introduction"
        else f"{chapter_title}: Summary and transition"
    )
    lesson = _single_fallback_lesson(
        conversation_id,
        chapter_index,
        {**group, "title": title},
    )
    lesson["blocks"] = _boundary_lesson_blocks(title, group["chunks"], stage)
    lesson.update(
        {
            "id": _stable_id(conversation_id, "lesson", chapter_index, stage, title),
            "title": title,
            "learning_objectives": [
                f"Identify the purpose and prerequisites of {chapter_title}"
                if stage == "introduction"
                else f"Synthesize {chapter_title} and connect it to what follows"
            ],
            "pedagogical_role": "definition" if stage == "introduction" else "synthesis",
            "sequencing_reason": (
                "Introduces this chapter before its detailed content."
                if stage == "introduction"
                else "Consolidates this chapter and prepares the next foundation."
            ),
            "lesson_stage": stage,
        }
    )
    return lesson


def _boundary_lesson_blocks(
    title: str,
    chunks: list[dict[str, Any]],
    stage: Literal["introduction", "conclusion"],
) -> list[dict[str, Any]]:
    content = _source_paragraphs(
        chunks,
        sentence_limit=4 if stage == "introduction" else 6,
        max_chars=1400,
    )
    if not content:
        content = "This subchapter is grounded in the chapter sources."
    return [
        _block(
            title,
            0,
            "markdown" if stage == "introduction" else "summary",
            "Chapter orientation" if stage == "introduction" else "Chapter synthesis",
            content,
            chunks,
        )
    ]


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
        "lesson_stage": str(group.get("lesson_stage") or "content"),
        "prerequisite_lesson_ids": [],
        "source_chunk_ids": ids,
        "citations": _citations(group["chunks"], ids),
        "source_queries": _clean_queries(
            [group["title"], *(_concept_for_chunk(chunk) for chunk in group["chunks"])],
            limit=8,
        ),
        "support_status": "supported" if _has_rich_teaching_material(group["chunks"]) else "insufficient_source_material",
        "blocks": _lesson_blocks(group["title"], group["chunks"]),
        "content_fingerprint": _content_fingerprint(ids),
        "generation_status": "ready",
    }


def _lesson_blocks(title: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    explanation = _source_paragraphs(chunks, sentence_limit=10)
    blocks = [_block(title, 0, "markdown", title, explanation, chunks)] if explanation else []
    blocks.extend(_special_blocks(title, chunks, start_index=len(blocks), existing_blocks=blocks))

    takeaway = _source_tail_paragraph(chunks, sentence_limit=3)
    if takeaway:
        blocks.append(_block(title, len(blocks), "summary", "Key takeaway", takeaway, chunks))
    return _dedupe_lesson_blocks(
        [block for block in blocks if str(block.get("content") or "").strip()]
    )[:MAX_BLOCKS_PER_LESSON]


def _special_blocks(
    title: str,
    chunks: list[dict[str, Any]],
    *,
    start_index: int,
    existing_blocks: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    seen: set[str] = set()
    for existing in existing_blocks or []:
        for expression in _equations(str(existing.get("content") or "")):
            if canonical := _canonical_math(expression):
                seen.add(f"equation:{canonical}")
        for table in _markdown_tables(str(existing.get("content") or "")):
            seen.add(f"table:{_norm(table.get('markdown', ''))}")
    # Structured rows are more readable as a real table than when flattened into prose,
    # so reserve room for them before adding formula groups.
    for chunk in chunks:
        for table in _markdown_tables(str(chunk.get("text", ""))):
            key = f"table:{_norm(table.get('markdown', ''))}"
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
                    "validation_status": "supported",
                }
            )
            if sum(block.get("block_type") == "table" for block in blocks) >= 3:
                break
        if sum(block.get("block_type") == "table" for block in blocks) >= 3:
            break

    equation_groups: dict[str, list[tuple[str, dict[str, Any]]]] = {
        "equation": [],
        "matrix": [],
        "chemical_equation": [],
    }
    for chunk in chunks:
        for expression in _source_equations(chunk):
            canonical = _canonical_math(expression)
            key = f"equation:{canonical}"
            if not canonical or key in seen:
                continue
            seen.add(key)
            block_type = (
                "matrix"
                if re.search(r"\\begin\{[pbvBV]?matrix\}", expression)
                else "chemical_equation"
                if _looks_chemical(expression)
                else "equation"
            )
            equation_groups[block_type].append((expression, chunk))

    for block_type in ("equation", "matrix", "chemical_equation"):
        expressions = equation_groups[block_type][:8]
        if not expressions:
            continue
        source_chunks = _dedupe_chunk_rows([chunk for _, chunk in expressions])
        source_ids = [str(chunk["id"]) for chunk in source_chunks]
        if block_type == "chemical_equation":
            content = "\n\n".join(f"$$\\ce{{{expression}}}$$" for expression, _ in expressions)
            label = "Chemical equations" if len(expressions) > 1 else "Chemical equation"
        else:
            content = "\n\n".join(f"$$\n{expression}\n$$" for expression, _ in expressions)
            label = (
                "Matrices" if block_type == "matrix" and len(expressions) > 1
                else "Matrix" if block_type == "matrix"
                else "Equations" if len(expressions) > 1
                else "Equation"
            )
        blocks.append(
            {
                "id": _stable_id(title, "special", start_index + len(blocks), block_type),
                "block_type": block_type,
                "title": label,
                "content": content,
                "data_json": {
                    "expression": expressions[0][0],
                    "expressions": [expression for expression, _ in expressions],
                },
                "source_chunk_ids": source_ids,
                "citations": _citations(source_chunks, source_ids),
                "validation_status": "supported",
            }
        )

    for chunk in chunks:
        events = _timeline_events(str(chunk.get("text", "")))
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
                    "validation_status": "supported",
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
    *,
    validation_status: str | None = None,
    source_query: str = "",
) -> dict[str, Any]:
    ids = _dedupe(chunk["id"] for chunk in chunks)[:8]
    citations = _citations(chunks, ids)
    status = validation_status or ("supported" if citations and str(content or "").strip() else "insufficient_source_material")
    if not citations or not str(content or "").strip():
        status = "insufficient_source_material"
    return {
        "id": _stable_id(title, "block", index, block_type),
        "block_type": block_type,
        "title": block_title,
        "content": content,
        "data_json": {},
        "source_chunk_ids": ids,
        "citations": citations,
        "source_query": source_query,
        "validation_status": status,
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
        "lesson_fingerprints": {},
        "quiz_fingerprints": {},
        "course_completed": False,
    }


def _resumable_chapter_prefix(
    conversation_id: str,
    outline: CourseOutline,
    chapters: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep only the completed leading chapters that still match the current source plan."""
    resumed: list[dict[str, Any]] = []
    for index, chapter_outline in enumerate(outline.chapters):
        if index >= len(chapters):
            break
        chapter = chapters[index]
        expected_id = _stable_id(conversation_id, "chapter", index, chapter_outline.title)
        expected_fingerprint = _content_fingerprint(chapter_outline.source_chunk_ids)
        if (
            chapter.get("id") != expected_id
            or chapter.get("content_fingerprint") != expected_fingerprint
            or chapter.get("generation_status", "ready") != "ready"
        ):
            break
        resumed.append(json.loads(json.dumps(chapter)))
    return resumed


def _quiz_fingerprint(quiz: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for question in quiz.get("questions", []):
        digest.update(_norm(question.get("prompt", "")).encode("utf-8"))
        for source_id in sorted(str(item) for item in question.get("source_chunk_ids", [])):
            digest.update(source_id.encode("utf-8"))
    return digest.hexdigest()[:24]


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
        lesson_chunks = [
            chunk_by_id[item]
            for item in lesson_outline.source_chunk_ids
            if item in chunk_by_id and not _looks_like_structure_only_chunk(chunk_by_id[item])
        ]
        lesson_chunks = _focused_fallback_lesson_chunks(
            lesson_chunks,
            outline,
            lesson_outline,
        )
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
                "lesson_stage": lesson_outline.lesson_stage,
                "source_queries": lesson_outline.source_queries,
                "support_status": "supported" if _has_rich_teaching_material(lesson_chunks) else "insufficient_source_material",
                "prerequisite_lesson_ids": [lessons[-1]["id"]] if lessons else [],
            }
        )
        if not lesson_chunks:
            # A lesson with no title-bound teaching evidence must be explicit about that gap.
            # Reusing the chapter's first chunks here is what made differently titled lessons
            # publish identical content.
            lesson["blocks"] = [
                _block(
                    lesson_outline.title,
                    0,
                    "warning",
                    "Insufficient source material",
                    _insufficient_source_message(lesson_outline.title),
                    [],
                    validation_status="insufficient_source_material",
                    source_query=" | ".join(lesson_outline.source_queries),
                )
            ]
        if lesson_outline.lesson_stage in {"introduction", "conclusion"}:
            lesson["blocks"] = _boundary_lesson_blocks(
                lesson_outline.title,
                lesson_chunks,
                lesson_outline.lesson_stage,
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
        "source_queries": outline.source_queries,
        "lessons": lessons,
        "quiz": _course_quiz(outline.title, selected, count=min(10, max(4, len(lessons) + 2)), scope="chapter"),
        "content_fingerprint": _content_fingerprint(chunk["id"] for chunk in selected),
        "generation_status": "ready",
    }


def _focused_fallback_lesson_chunks(
    chunks: list[dict[str, Any]],
    chapter: OutlineChapter,
    lesson: OutlineLesson,
) -> list[dict[str, Any]]:
    """Keep deterministic prose focused on the lesson instead of dumping its coverage tail."""
    if not chunks:
        return []
    binding_lesson = lesson.model_copy(
        update={
            "summary": "",
            "learning_objectives": [],
            "source_chunk_ids": [],
            # Validation appends broad coverage hints later. The title and optional agenda
            # segment are the stable, lesson-specific part of this retrieval contract.
            "source_queries": lesson.source_queries[:2],
        }
    )
    chunk_by_id = {str(chunk["id"]): chunk for chunk in chunks}
    scored = sorted(
        (
            (_chunk_lesson_affinity(chunk, binding_lesson, chunk_by_id, set()), index, chunk)
            for index, chunk in enumerate(chunks)
        ),
        key=lambda item: (-item[0], item[1]),
    )
    best_score = scored[0][0]
    minimum_score = max(4.0, best_score * 0.35)
    focused = [chunk for score, _index, chunk in scored if score >= minimum_score]
    return focused[:MAX_LESSON_EVIDENCE_CHUNKS]


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


def _focused_source_paragraphs(
    chunks: list[dict[str, Any]],
    *,
    focus: str,
    sentence_limit: int,
    primary_focus: str = "",
    max_chars: int = 5000,
) -> str:
    focus_terms = _meaningful_title_terms(focus)
    primary_terms = _meaningful_title_terms(primary_focus)
    if not focus_terms:
        return ""

    candidates: list[tuple[int, int, int, str]] = []
    candidate_keys: set[str] = set()
    ordinal = 0
    for chunk_index, chunk in enumerate(chunks[:10]):
        metadata = chunk.get("metadata") or {}
        heading = " ".join(
            str(metadata.get(key) or "")
            for key in ("heading_path", "section_title", "subchapter_title")
        )
        heading_overlap = len(focus_terms.intersection(_meaningful_title_terms(heading)))
        for sentence in _source_sentences(str(chunk.get("text") or "")):
            sentence_key = _norm(sentence)
            if not sentence_key or sentence_key in candidate_keys:
                continue
            candidate_keys.add(sentence_key)
            sentence_overlap = len(focus_terms.intersection(_meaningful_title_terms(sentence)))
            primary_overlap = len(primary_terms.intersection(_meaningful_title_terms(sentence)))
            score = primary_overlap * 8 + sentence_overlap * 3 + heading_overlap + max(0, 4 - chunk_index)
            candidates.append((score, ordinal, sentence_overlap, sentence))
            ordinal += 1

    direct_matches = [candidate for candidate in candidates if candidate[2] > 0]
    relevant = direct_matches or [candidate for candidate in candidates if candidate[0] > 0]
    if not relevant:
        return ""

    selected = sorted(relevant, key=lambda item: (-item[0], item[1]))[:sentence_limit]
    selected.sort(key=lambda item: item[1])
    sentences: list[str] = []
    seen: set[str] = set()
    for _, _, _, sentence in selected:
        key = _norm(sentence)
        if not key or key in seen:
            continue
        seen.add(key)
        sentences.append(sentence)
    return "\n\n".join(
        " ".join(sentences[index : index + 3]).strip()
        for index in range(0, len(sentences), 3)
        if sentences[index : index + 3]
    )[:max_chars].strip()


def _source_tail_paragraph(chunks: list[dict[str, Any]], *, sentence_limit: int) -> str:
    sentences: list[str] = []
    seen: set[str] = set()
    for chunk in chunks[:10]:
        for sentence in _source_sentences(str(chunk.get("text") or "")):
            key = _norm(sentence)
            if key and key not in seen:
                seen.add(key)
                sentences.append(sentence)
    return " ".join(sentences[-sentence_limit:]).strip()


def _dedupe_lesson_blocks(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    prose_types = {"markdown", "definition", "example", "procedure", "warning", "summary"}
    exact_seen: set[tuple[str, str]] = set()
    for block in blocks:
        content = str(block.get("content") or "").strip()
        if not content:
            continue
        block_type = str(block.get("block_type") or "markdown")
        normalized = _norm(re.sub(r"<[^>]+>", " ", content))
        exact_key = (block_type, normalized)
        if exact_key in exact_seen:
            continue
        exact_seen.add(exact_key)
        if block_type in prose_types and any(
            str(previous.get("block_type") or "markdown") in prose_types
            and _is_redundant_prose(content, str(previous.get("content") or ""))
            for previous in kept
        ):
            continue
        kept.append(block)
    return kept


def _is_redundant_prose(candidate: str, existing: str) -> bool:
    candidate_text = _norm(re.sub(r"<[^>]+>", " ", candidate))
    existing_text = _norm(re.sub(r"<[^>]+>", " ", existing))
    if not candidate_text or not existing_text:
        return False
    if candidate_text == existing_text:
        return True
    if len(candidate_text) > len(existing_text) * 1.15:
        return False
    if candidate_text in existing_text:
        return True
    candidate_tokens = candidate_text.split()
    existing_tokens = existing_text.split()
    if min(len(candidate_tokens), len(existing_tokens)) < 8:
        return False
    overlap = len(set(candidate_tokens).intersection(existing_tokens))
    return overlap / max(1, len(set(candidate_tokens))) >= 0.92


def _source_sentences(text: str) -> list[str]:
    prose = strip_tables_for_prose(str(text or ""))
    prose = re.sub(r"\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]", " ", prose)
    prose = re.sub(r"<[^>]+>", " ", prose)
    prose = "\n".join(
        line
        for line in prose.replace("\r", "\n").splitlines()
        if not (line.strip().startswith("|") and line.strip().endswith("|"))
    )
    clean = re.sub(r"\s+", " ", prose).strip()
    pieces = re.split(r"(?<=[.!?。！？])\s+|\s+[•*-]\s+", clean)
    out = []
    for piece in pieces:
        sentence = piece.strip(" -•\t")
        sentence = re.sub(r"^(?:[-+•—]\s+)", "", sentence)
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
    if re.search(r"\\ce\s*\{", text):
        return True
    sides = re.split(r"(?:<=>|->|<-|=>|→|⇌|\\rightarrow|\\leftrightarrow)", text, maxsplit=1)
    if len(sides) != 2:
        return False
    species = r"(?:\d+\s*)?(?:[A-Z][a-z]?(?:_?\{?\d+\}?)?)+(?:\^?\{?[+-]\}?)?(?:\((?:s|l|g|aq)\))?"
    side_pattern = re.compile(rf"^\s*{species}(?:\s*\+\s*{species})*\s*$")
    return all(side_pattern.fullmatch(side.strip()) for side in sides)


def _markdown_tables(text: str) -> list[dict[str, Any]]:
    return extract_markdown_tables(text)[:4]


def _timeline_events(text: str) -> list[dict[str, str]]:
    event_re = re.compile(r"\b(?:\d{3,4}(?:\s*(?:BCE|BC|CE|AD))?|Q[1-4]\s+\d{4})\b", re.IGNORECASE)
    events = []
    for line in strip_tables_for_prose(text).splitlines():
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


def _markdown_source_fingerprint(files: list[dict[str, Any]], markdown: str) -> str:
    digest = hashlib.sha256()
    digest.update(COURSEBUILDER_VERSION.encode())
    for file in sorted(files, key=lambda item: str(item.get("id") or "")):
        digest.update(
            f"{file.get('id', '')}:{file.get('filename', '')}:{file.get('size_bytes', 0)}".encode()
        )
    digest.update(str(markdown or "").encode())
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


def _clean_planned_title(value: Any) -> str:
    title = re.sub(r"\*+", "", str(value or ""))
    title = re.sub(r"^source plan item\s*:\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip(" -:;,.#")
    return title[:140] or "Untitled section"


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
