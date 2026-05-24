from __future__ import annotations

import logging
import re
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1
from typing import Any, Literal

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
    CourseSectionRecord,
    KnowledgeCheckRecord,
    SearchChunkRecord,
)
from services.concept_inventory_service import resolve_concept


logger = logging.getLogger(__name__)


BloomLevel = Literal["remember", "understand", "apply", "analyze"]
_BLOOM_LEVELS: set[str] = {"remember", "understand", "apply", "analyze"}
_LOCAL_FALLBACK_MODEL = "gemma4:e2b"
_MAX_PHASES = 12
_MAX_OBJECTIVES_PER_PHASE = 8
_FALLBACK_MAX_PHASES = 5
_WORD_RE = re.compile(r"[a-z0-9]+")


class LearningObjectiveCandidate(BaseModel):
    objective_text: str
    bloom_level: BloomLevel = "understand"
    concept_names: list[str] = Field(default_factory=list)
    source_chunk_ids: list[str] = Field(default_factory=list)


class LearningPhaseCandidate(BaseModel):
    title: str
    summary: str = ""
    order_index: int = 0
    source_chunk_ids: list[str] = Field(default_factory=list)
    objectives: list[LearningObjectiveCandidate] = Field(default_factory=list)


class LearningMapCandidateBatch(BaseModel):
    phases: list[LearningPhaseCandidate] = Field(default_factory=list)


@dataclass(slots=True)
class _PhaseAccumulator:
    title: str
    summary: str = ""
    order_index: int = 0
    source_file_ids: set[str] = field(default_factory=set)
    source_section_ids: set[str] = field(default_factory=set)
    source_chunk_ids: set[str] = field(default_factory=set)
    extraction_method: str = "llm"

    @property
    def key(self) -> str:
        return normalize_learning_key(self.title)


@dataclass(slots=True)
class _ObjectiveAccumulator:
    phase_key: str
    objective_text: str
    bloom_level: str = "understand"
    order_index: int = 0
    concept_ids: set[str] = field(default_factory=set)
    source_file_ids: set[str] = field(default_factory=set)
    source_section_ids: set[str] = field(default_factory=set)
    source_chunk_ids: set[str] = field(default_factory=set)
    extraction_method: str = "llm"

    @property
    def key(self) -> str:
        return normalize_learning_key(f"{self.phase_key}:{self.objective_text}")


class LearningMapService:
    """Build and load the student-facing course learning map."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def ensure_schema(self, session: AsyncSession) -> None:
        connection = await session.connection()

        def create_tables(sync_connection) -> None:  # noqa: ANN001
            CourseLearningPhaseRecord.__table__.create(sync_connection, checkfirst=True)
            CourseLearningObjectiveRecord.__table__.create(sync_connection, checkfirst=True)

        await connection.run_sync(create_tables)

    async def rebuild_map(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        *,
        llm_options: dict[str, Any] | None = None,
        use_llm: bool = True,
    ) -> tuple[list[CourseLearningPhaseRecord], list[CourseLearningObjectiveRecord]]:
        await self.ensure_schema(session)
        sections = await self._load_sections(session, conversation_id)
        chunks = await self._load_chunks(session, conversation_id)
        concepts = await self._load_active_concepts(session, conversation_id)
        if not sections and not chunks:
            return await self._persist_map(session, conversation_id, [], [])

        candidates: list[LearningPhaseCandidate] = []
        if use_llm:
            try:
                candidates = await self._llm_candidates(chunks, concepts, llm_options=llm_options)
            except Exception:  # noqa: BLE001
                logger.exception("LLM learning-map extraction failed; using deterministic fallback")
        if not candidates:
            candidates = self._fallback_candidates(sections, chunks, concepts)

        phases, objectives = self._merge_candidates(candidates, chunks, concepts)
        return await self._persist_map(session, conversation_id, phases, objectives)

    async def load_map(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> tuple[list[CourseLearningPhaseRecord], list[CourseLearningObjectiveRecord]]:
        await self.ensure_schema(session)
        phases_result = await session.execute(
            select(CourseLearningPhaseRecord)
            .where(CourseLearningPhaseRecord.conversation_id == conversation_id)
            .order_by(CourseLearningPhaseRecord.order_index, CourseLearningPhaseRecord.title)
        )
        objectives_result = await session.execute(
            select(CourseLearningObjectiveRecord)
            .where(CourseLearningObjectiveRecord.conversation_id == conversation_id)
            .order_by(CourseLearningObjectiveRecord.order_index, CourseLearningObjectiveRecord.objective_text)
        )
        phases = [phase for phase in phases_result.scalars().all() if _active_phase(phase)]
        active_phase_ids = {phase.id for phase in phases}
        objectives = [
            objective
            for objective in objectives_result.scalars().all()
            if _active_objective(objective) and objective.phase_id in active_phase_ids
        ]
        return phases, objectives

    async def _load_sections(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> list[CourseSectionRecord]:
        result = await session.execute(
            select(CourseSectionRecord)
            .where(CourseSectionRecord.conversation_id == conversation_id)
            .order_by(CourseSectionRecord.document_id, CourseSectionRecord.order_index)
        )
        return list(result.scalars().all())

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

    async def _load_active_concepts(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> list[CourseConceptRecord]:
        result = await session.execute(
            select(CourseConceptRecord)
            .where(CourseConceptRecord.conversation_id == conversation_id)
            .order_by(CourseConceptRecord.importance.desc(), CourseConceptRecord.canonical_name)
        )
        concepts = list(result.scalars().all())
        return [
            concept
            for concept in concepts
            if not (concept.concept_metadata or {}).get("inactive") and concept.source_chunk_ids
        ]

    async def _llm_candidates(
        self,
        chunks: list[SearchChunkRecord],
        concepts: list[CourseConceptRecord],
        *,
        llm_options: dict[str, Any] | None,
    ) -> list[LearningPhaseCandidate]:
        if not chunks:
            return []
        concept_lines = "\n".join(
            f"- {concept.canonical_name}: {concept.description[:180]}"
            for concept in concepts[:80]
        ) or "(no concept inventory available)"
        user_prompt = (
            "Known canonical concepts:\n"
            f"{concept_lines}\n\n"
            "Course chunks:\n"
            f"{_format_chunks_for_llm(chunks)}"
        )
        last_error: Exception | None = None
        for label, client in self._llm_clients(llm_options):
            try:
                response = await client.chat_structured(
                    messages=[
                        {"role": "system", "content": _SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    schema=LearningMapCandidateBatch,
                    options={"temperature": 0.1, "num_predict": 2000, "max_tokens": 2000},
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "learning-map extraction with %s model %s failed",
                    label,
                    client.model,
                    exc_info=True,
                )
                continue
            phases = [_clean_phase_candidate(item) for item in response.phases]
            phases = [item for item in phases if item is not None]
            if phases:
                return phases[:_MAX_PHASES]
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

    def _fallback_candidates(
        self,
        sections: list[CourseSectionRecord],
        chunks: list[SearchChunkRecord],
        concepts: list[CourseConceptRecord],
    ) -> list[LearningPhaseCandidate]:
        if concepts:
            return _concept_path_fallback(sections, chunks, concepts)

        chunks_by_section: dict[str, list[SearchChunkRecord]] = {}
        for chunk in chunks:
            chunks_by_section.setdefault(str(chunk.section_id), []).append(chunk)

        clean_sections = [
            section
            for section in sections
            if section.level <= 2 and _valid_phase_title(section.title)
        ]
        if not clean_sections:
            return []

        phase_count = min(_FALLBACK_MAX_PHASES, max(1, round(len(clean_sections) ** 0.5)))
        section_groups = _even_groups(clean_sections, phase_count)
        phases: list[LearningPhaseCandidate] = []
        for index, group in enumerate(section_groups):
            if not group:
                continue
            title = _generic_phase_title(index)
            source_chunks = [
                chunk.id
                for section in group
                for chunk in chunks_by_section.get(str(section.id), [])
            ][:8]
            objective_title = _clean_label(group[0].title)
            phases.append(
                LearningPhaseCandidate(
                    title=title,
                    summary=" ".join(section.summary for section in group if section.summary)[:800],
                    order_index=index,
                    source_chunk_ids=source_chunks,
                    objectives=[
                        LearningObjectiveCandidate(
                            objective_text=f"Explain the main idea of {objective_title}",
                            bloom_level="understand",
                            source_chunk_ids=source_chunks,
                        )
                    ],
                )
            )
        return phases

    def _merge_candidates(
        self,
        candidates: list[LearningPhaseCandidate],
        chunks: list[SearchChunkRecord],
        concepts: list[CourseConceptRecord],
    ) -> tuple[list[_PhaseAccumulator], list[_ObjectiveAccumulator]]:
        chunks_by_id = {chunk.id: chunk for chunk in chunks}
        phases: list[_PhaseAccumulator] = []
        objectives: list[_ObjectiveAccumulator] = []
        seen_phase_keys: set[str] = set()
        seen_objective_keys: set[str] = set()

        for phase_index, candidate in enumerate(candidates[:_MAX_PHASES]):
            title = _clean_label(candidate.title)[:512]
            if not _valid_phase_title(title):
                continue
            phase = _PhaseAccumulator(
                title=title,
                summary=" ".join(str(candidate.summary or "").split())[:800],
                order_index=int(candidate.order_index if candidate.order_index >= 0 else phase_index),
            )
            if phase.key in seen_phase_keys:
                continue
            seen_phase_keys.add(phase.key)
            _add_source_refs(phase, candidate.source_chunk_ids, chunks_by_id)
            phase_objective_count = 0
            for objective_index, objective_candidate in enumerate(candidate.objectives[:_MAX_OBJECTIVES_PER_PHASE]):
                objective_text = _clean_objective_text(objective_candidate.objective_text)
                if not objective_text:
                    continue
                objective = _ObjectiveAccumulator(
                    phase_key=phase.key,
                    objective_text=objective_text,
                    bloom_level=_coerce_bloom(objective_candidate.bloom_level),
                    order_index=objective_index,
                    extraction_method="llm",
                )
                for name in objective_candidate.concept_names:
                    resolved = resolve_concept(name, concepts)
                    if resolved is not None:
                        objective.concept_ids.add(str(resolved.id))
                _add_source_refs(objective, objective_candidate.source_chunk_ids, chunks_by_id)
                if not objective.source_chunk_ids:
                    objective.source_chunk_ids.update(phase.source_chunk_ids)
                    objective.source_section_ids.update(phase.source_section_ids)
                    objective.source_file_ids.update(phase.source_file_ids)
                if not objective.concept_ids:
                    objective.concept_ids.update(
                        str(concept.id)
                        for concept in concepts
                        if set(concept.source_chunk_ids or []) & objective.source_chunk_ids
                    )
                objective_key = objective.key
                if objective_key in seen_objective_keys:
                    continue
                seen_objective_keys.add(objective_key)
                objectives.append(objective)
                phase_objective_count += 1
            if phase_objective_count:
                phases.append(phase)
        return phases, objectives

    async def _persist_map(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        phases: list[_PhaseAccumulator],
        objectives: list[_ObjectiveAccumulator],
    ) -> tuple[list[CourseLearningPhaseRecord], list[CourseLearningObjectiveRecord]]:
        existing_phases, existing_objectives = await self._load_all_records(session, conversation_id)
        phase_refs, objective_refs = await self._assessment_reference_counts(session, conversation_id)
        phase_records = [self._to_phase_record(conversation_id, phase) for phase in phases]
        phase_id_by_key = {record.phase_key: record.id for record in phase_records}
        objective_records = [
            self._to_objective_record(conversation_id, objective, phase_id_by_key[objective.phase_key])
            for objective in objectives
            if objective.phase_key in phase_id_by_key
        ]

        existing_phases_by_id = {phase.id: phase for phase in existing_phases}
        existing_objectives_by_id = {objective.id: objective for objective in existing_objectives}
        desired_phase_ids = {phase.id for phase in phase_records}
        desired_objective_ids = {objective.id for objective in objective_records}
        persisted_phases: list[CourseLearningPhaseRecord] = []
        persisted_objectives: list[CourseLearningObjectiveRecord] = []

        for desired in phase_records:
            current = existing_phases_by_id.get(desired.id)
            if current is None:
                session.add(desired)
                persisted_phases.append(desired)
            else:
                _copy_phase_record(current, desired)
                persisted_phases.append(current)

        # Objectives have a strict FK to phases. Flush phase upserts first so
        # SQLAlchemy cannot batch objective inserts before new parent rows.
        await session.flush()

        for desired in objective_records:
            current = existing_objectives_by_id.get(desired.id)
            if current is None:
                session.add(desired)
                persisted_objectives.append(desired)
            else:
                _copy_objective_record(current, desired)
                persisted_objectives.append(current)

        now = datetime.now(timezone.utc)
        for objective in existing_objectives:
            if objective.id in desired_objective_ids:
                continue
            if objective_refs.get(str(objective.id), 0) > 0:
                objective.objective_metadata = {
                    **dict(objective.objective_metadata or {}),
                    "inactive": True,
                    "inactive_reason": "no_current_course_sources",
                }
                objective.source_file_ids = []
                objective.source_section_ids = []
                objective.source_chunk_ids = []
                objective.concept_ids = []
                objective.updated_at = now
            else:
                await session.delete(objective)

        referenced_objective_phase_ids = {
            objective.phase_id
            for objective in existing_objectives
            if objective_refs.get(str(objective.id), 0) > 0
        }
        for phase in existing_phases:
            if phase.id in desired_phase_ids:
                continue
            if phase_refs.get(str(phase.id), 0) > 0 or phase.id in referenced_objective_phase_ids:
                phase.phase_metadata = {
                    **dict(phase.phase_metadata or {}),
                    "inactive": True,
                    "inactive_reason": "no_current_course_sources",
                }
                phase.source_file_ids = []
                phase.source_section_ids = []
                phase.source_chunk_ids = []
                phase.updated_at = now
            else:
                await session.delete(phase)

        await session.flush()
        return persisted_phases, persisted_objectives

    async def _load_all_records(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> tuple[list[CourseLearningPhaseRecord], list[CourseLearningObjectiveRecord]]:
        phases = await session.execute(
            select(CourseLearningPhaseRecord).where(CourseLearningPhaseRecord.conversation_id == conversation_id)
        )
        objectives = await session.execute(
            select(CourseLearningObjectiveRecord).where(
                CourseLearningObjectiveRecord.conversation_id == conversation_id
            )
        )
        return list(phases.scalars().all()), list(objectives.scalars().all())

    async def _assessment_reference_counts(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
    ) -> tuple[dict[str, int], dict[str, int]]:
        phase_refs: dict[str, int] = {}
        objective_refs: dict[str, int] = {}
        result = await session.execute(
            select(KnowledgeCheckRecord).where(KnowledgeCheckRecord.conversation_id == conversation_id)
        )
        for check in result.scalars().all():
            metadata = check.check_metadata or {}
            phase_id = str(metadata.get("phase_id") or "")
            objective_id = str(metadata.get("objective_id") or "")
            if phase_id:
                phase_refs[phase_id] = phase_refs.get(phase_id, 0) + 1
            if objective_id:
                objective_refs[objective_id] = objective_refs.get(objective_id, 0) + 1
        return phase_refs, objective_refs

    @staticmethod
    def _to_phase_record(
        conversation_id: uuid.UUID,
        phase: _PhaseAccumulator,
    ) -> CourseLearningPhaseRecord:
        now = datetime.now(timezone.utc)
        return CourseLearningPhaseRecord(
            id=stable_phase_id(conversation_id, phase.title),
            conversation_id=conversation_id,
            phase_key=phase.key,
            title=phase.title,
            summary=phase.summary,
            order_index=phase.order_index,
            source_file_ids=sorted(phase.source_file_ids),
            source_section_ids=sorted(phase.source_section_ids),
            source_chunk_ids=sorted(phase.source_chunk_ids),
            phase_metadata={"extraction_method": phase.extraction_method},
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def _to_objective_record(
        conversation_id: uuid.UUID,
        objective: _ObjectiveAccumulator,
        phase_id: uuid.UUID,
    ) -> CourseLearningObjectiveRecord:
        now = datetime.now(timezone.utc)
        return CourseLearningObjectiveRecord(
            id=stable_objective_id(conversation_id, objective.phase_key, objective.objective_text),
            conversation_id=conversation_id,
            phase_id=phase_id,
            objective_key=objective.key,
            objective_text=objective.objective_text,
            bloom_level=objective.bloom_level,
            order_index=objective.order_index,
            concept_ids=sorted(objective.concept_ids),
            source_file_ids=sorted(objective.source_file_ids),
            source_section_ids=sorted(objective.source_section_ids),
            source_chunk_ids=sorted(objective.source_chunk_ids),
            objective_metadata={"extraction_method": objective.extraction_method},
            created_at=now,
            updated_at=now,
        )


def normalize_learning_key(value: str) -> str:
    return " ".join(_WORD_RE.findall(_ascii_fold(value).casefold()))[:256]


def stable_phase_id(conversation_id: uuid.UUID | str, title: str) -> uuid.UUID:
    key = normalize_learning_key(title)
    seed = f"phase:{conversation_id}:{key}"
    return uuid.uuid5(uuid.NAMESPACE_URL, f"teacherlm:{seed}:{sha1(seed.encode()).hexdigest()[:12]}")


def stable_objective_id(conversation_id: uuid.UUID | str, phase_key: str, objective_text: str) -> uuid.UUID:
    key = normalize_learning_key(f"{phase_key}:{objective_text}")
    seed = f"objective:{conversation_id}:{key}"
    return uuid.uuid5(uuid.NAMESPACE_URL, f"teacherlm:{seed}:{sha1(seed.encode()).hexdigest()[:12]}")


def needs_learning_map_compaction(
    phases: list[CourseLearningPhaseRecord],
    objectives: list[CourseLearningObjectiveRecord],
) -> bool:
    """Detect old fallback maps that exposed raw section headings as phases."""

    objectives_per_phase: dict[uuid.UUID, int] = {}
    for objective in objectives:
        objectives_per_phase[objective.phase_id] = objectives_per_phase.get(objective.phase_id, 0) + 1
    one_objective_phases = sum(1 for phase in phases if objectives_per_phase.get(phase.id, 0) <= 1)
    too_many_singletons = one_objective_phases / max(1, len(phases)) >= 0.75
    noisy_titles = sum(1 for phase in phases if _fallback_title_is_too_granular(phase.title))
    return noisy_titles > 0 or (len(phases) > _FALLBACK_MAX_PHASES and too_many_singletons)


def _ascii_fold(value: str) -> str:
    return unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode("ascii")


def _concept_path_fallback(
    sections: list[CourseSectionRecord],
    chunks: list[SearchChunkRecord],
    concepts: list[CourseConceptRecord],
) -> list[LearningPhaseCandidate]:
    chunk_order = {chunk.id: index for index, chunk in enumerate(chunks)}
    chunks_by_id = {chunk.id: chunk for chunk in chunks}
    sections_by_id = {str(section.id): section for section in sections}
    ordered = sorted(
        concepts,
        key=lambda concept: (
            min((chunk_order.get(chunk_id, 10**9) for chunk_id in concept.source_chunk_ids or []), default=10**9),
            -float(concept.importance),
            concept.canonical_name,
        ),
    )
    phase_count = min(_FALLBACK_MAX_PHASES, max(1, (len(ordered) + 7) // 8))
    concept_groups = _even_groups(ordered, phase_count)
    phases: list[LearningPhaseCandidate] = []
    for phase_index, group in enumerate(concept_groups):
        if not group:
            continue
        source_chunk_ids = _source_chunk_ids_for_concepts(group)
        title = _fallback_phase_title(phase_index, group, sections_by_id)
        objectives: list[LearningObjectiveCandidate] = []
        objective_groups = _even_groups(group, min(4, max(1, (len(group) + 3) // 4)))
        for objective_group in objective_groups:
            if not objective_group:
                continue
            objective_chunks = _source_chunk_ids_for_concepts(objective_group) or source_chunk_ids
            objectives.append(
                LearningObjectiveCandidate(
                    objective_text=_objective_from_concepts(title, objective_group),
                    bloom_level=_highest_bloom(objective_group),
                    concept_names=[concept.canonical_name for concept in objective_group],
                    source_chunk_ids=objective_chunks[:8],
                )
            )
        if not objectives:
            continue
        summary = _phase_summary_from_chunks(source_chunk_ids, chunks_by_id)
        phases.append(
            LearningPhaseCandidate(
                title=title,
                summary=summary,
                order_index=phase_index,
                source_chunk_ids=source_chunk_ids[:12],
                objectives=objectives,
            )
        )
    return phases


def _fallback_phase_title(
    phase_index: int,
    concepts: list[CourseConceptRecord],
    sections_by_id: dict[str, CourseSectionRecord],
) -> str:
    section_titles: dict[str, tuple[str, int]] = {}
    for concept in concepts:
        for section_id in concept.source_section_ids or []:
            section = sections_by_id.get(str(section_id))
            if section is None:
                continue
            title = _clean_label(_best_phase_heading(section))
            if not title or not _valid_phase_title(title) or _fallback_title_is_too_granular(title):
                continue
            key = normalize_learning_key(title)
            current_title, count = section_titles.get(key, (title, 0))
            section_titles[key] = (current_title, count + 1)
    if section_titles:
        title, _count = max(section_titles.values(), key=lambda item: item[1])
        return title[:120]
    return _generic_phase_title(phase_index)


def _best_phase_heading(section: CourseSectionRecord) -> str:
    for heading in section.heading_path or []:
        cleaned = _clean_label(heading)
        if _valid_phase_title(cleaned) and not _fallback_title_is_too_granular(cleaned):
            return cleaned
    return section.title


def _fallback_title_is_too_granular(title: str) -> bool:
    cleaned = _clean_label(title)
    key = normalize_learning_key(cleaned)
    if len(cleaned) > 90:
        return True
    if re.search(r"</?\w+|[$\\{}]|^\d+[\).:-]|^[a-z]$", cleaned, re.IGNORECASE):
        return True
    if re.search(r"\bdoc\s*\d|\bscore\s*:", cleaned, re.IGNORECASE):
        return True
    if key in {"question", "example", "exemple", "scenario", "stats", "python u2"}:
        return True
    return False


def _source_chunk_ids_for_concepts(concepts: list[CourseConceptRecord]) -> list[str]:
    seen: set[str] = set()
    chunk_ids: list[str] = []
    for concept in concepts:
        for chunk_id in concept.source_chunk_ids or []:
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            chunk_ids.append(chunk_id)
    return chunk_ids


def _phase_summary_from_chunks(source_chunk_ids: list[str], chunks_by_id: dict[str, SearchChunkRecord]) -> str:
    texts: list[str] = []
    for chunk_id in source_chunk_ids[:3]:
        chunk = chunks_by_id.get(chunk_id)
        if chunk is None:
            continue
        text = " ".join(chunk.text.split())
        if text:
            texts.append(text[:220])
    return " ".join(texts)[:800]


def _generic_phase_title(index: int) -> str:
    titles = [
        "Course Foundations",
        "Core Ideas",
        "Methods and Procedures",
        "Applications and Cases",
        "Evaluation and Synthesis",
    ]
    return titles[min(index, len(titles) - 1)]


def _even_groups(items: list[Any], group_count: int) -> list[list[Any]]:
    if not items:
        return []
    group_count = max(1, min(group_count, len(items)))
    return [
        items[(index * len(items)) // group_count : ((index + 1) * len(items)) // group_count]
        for index in range(group_count)
    ]


def _active_phase(phase: CourseLearningPhaseRecord) -> bool:
    return not bool((phase.phase_metadata or {}).get("inactive"))


def _active_objective(objective: CourseLearningObjectiveRecord) -> bool:
    return not bool((objective.objective_metadata or {}).get("inactive"))


def _copy_phase_record(target: CourseLearningPhaseRecord, source: CourseLearningPhaseRecord) -> None:
    metadata = dict(source.phase_metadata or {})
    metadata.pop("inactive", None)
    metadata.pop("inactive_reason", None)
    target.phase_key = source.phase_key
    target.title = source.title
    target.summary = source.summary
    target.order_index = source.order_index
    target.source_file_ids = list(source.source_file_ids or [])
    target.source_section_ids = list(source.source_section_ids or [])
    target.source_chunk_ids = list(source.source_chunk_ids or [])
    target.phase_metadata = metadata
    target.updated_at = source.updated_at


def _copy_objective_record(
    target: CourseLearningObjectiveRecord,
    source: CourseLearningObjectiveRecord,
) -> None:
    metadata = dict(source.objective_metadata or {})
    metadata.pop("inactive", None)
    metadata.pop("inactive_reason", None)
    target.phase_id = source.phase_id
    target.objective_key = source.objective_key
    target.objective_text = source.objective_text
    target.bloom_level = source.bloom_level
    target.order_index = source.order_index
    target.concept_ids = list(source.concept_ids or [])
    target.source_file_ids = list(source.source_file_ids or [])
    target.source_section_ids = list(source.source_section_ids or [])
    target.source_chunk_ids = list(source.source_chunk_ids or [])
    target.objective_metadata = metadata
    target.updated_at = source.updated_at


def _format_chunks_for_llm(chunks: list[SearchChunkRecord]) -> str:
    parts: list[str] = []
    for chunk in chunks[:64]:
        metadata = chunk.chunk_metadata or {}
        text = " ".join(chunk.text.split())
        if len(text) > 1100:
            text = text[:1100].rsplit(" ", 1)[0].strip()
        parts.append(
            "\n".join(
                [
                    f"chunk_id: {chunk.id}",
                    f"source: {chunk.source_filename}",
                    f"course_part: {metadata.get('heading_path') or ' > '.join(chunk.heading_path or [])}",
                    "text:",
                    text,
                ]
            )
        )
    return "\n\n---\n\n".join(parts)


def _clean_phase_candidate(candidate: LearningPhaseCandidate) -> LearningPhaseCandidate | None:
    title = _clean_label(candidate.title)
    if not _valid_phase_title(title):
        return None
    objectives = [
        objective.model_copy(update={"objective_text": _clean_objective_text(objective.objective_text)})
        for objective in candidate.objectives
        if _clean_objective_text(objective.objective_text)
    ]
    if not objectives:
        return None
    return candidate.model_copy(update={"title": title, "objectives": objectives})


def _clean_label(value: str) -> str:
    text = re.sub(r"</?\w+[^>]*>", " ", str(value or ""))
    text = text.replace("**", "").replace("__", "")
    text = re.sub(r"\s+", " ", text).strip(" -*#:;,.")
    if ">" in text:
        parts = [part.strip() for part in text.split(">") if part.strip()]
        text = parts[-1] if parts else text
    return text


def _clean_objective_text(value: str) -> str:
    text = _clean_label(value)
    if not text:
        return ""
    if len(text) > 220:
        text = text[:220].rsplit(" ", 1)[0].strip()
    if len(text.split()) < 3:
        return f"Explain {text}"
    return text


def _valid_phase_title(value: str) -> bool:
    text = _clean_label(value)
    key = normalize_learning_key(text)
    if not 2 <= len(text) <= 160 or not key:
        return False
    if re.search(r"[$\\{}<>]|</?\w+|^\d+(?:\.\d+)*$", text):
        return False
    if key in {"introduction", "conclusion", "summary", "resume", "overview", "agenda", "plan"}:
        return False
    return True


def _add_source_refs(
    item: _PhaseAccumulator | _ObjectiveAccumulator,
    source_chunk_ids: list[str],
    chunks_by_id: dict[str, SearchChunkRecord],
) -> None:
    for chunk_id in source_chunk_ids:
        chunk = chunks_by_id.get(str(chunk_id))
        if chunk is None:
            continue
        item.source_chunk_ids.add(chunk.id)
        item.source_section_ids.add(str(chunk.section_id))
        item.source_file_ids.add(chunk.source_file_id)


def _objective_from_concepts(section_title: str, concepts: list[CourseConceptRecord]) -> str:
    names = [concept.canonical_name for concept in concepts[:3]]
    if len(names) == 1:
        return f"Explain {names[0]} using the course material"
    return f"Explain how {', '.join(names[:-1])}, and {names[-1]} fit together"


def _highest_bloom(concepts: list[CourseConceptRecord]) -> str:
    order = {"remember": 0, "understand": 1, "apply": 2, "analyze": 3}
    return max((concept.bloom_level for concept in concepts), key=lambda item: order.get(item, 1), default="understand")


def _coerce_bloom(value: str) -> str:
    text = str(value or "understand").strip().lower()
    return text if text in _BLOOM_LEVELS else "understand"


_SYSTEM_PROMPT = """You build a student-facing learning map for TeacherLM.

The uploaded course can be about any domain: math, biology, law, language,
programming, medicine, business, history, or something else. Do not assume a
specific subject.

Return an ordered set of phases/modules. Under each phase, return measurable
learning objectives. Objectives should describe what the student should be able
to explain, apply, compare, calculate, analyze, or use after studying that part.

Rules:
- Use exact chunk_id values for source_chunk_ids.
- Prefer the course's natural order.
- Phase titles may come from meaningful course sections, but objectives must be
  learning outcomes, not raw titles.
- Link objectives to known canonical concepts by concept_names when possible.
- Do NOT return file metadata, page labels, table cells, examples, variables,
  formulas by themselves, percentages, or incomplete phrases.
- Keep the map compact: enough to show progress, not every slide heading.
- Return only JSON matching the schema."""


_service: LearningMapService | None = None


def get_learning_map_service() -> LearningMapService:
    global _service
    if _service is None:
        _service = LearningMapService()
    return _service
