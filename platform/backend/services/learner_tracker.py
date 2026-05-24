from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from teacherlm_core.schemas.generator_io import LearnerUpdates
from teacherlm_core.schemas.learner_state import LearnerState

from db.models import (
    CourseConceptRecord,
    CourseLearningObjectiveRecord,
    CourseLearningPhaseRecord,
    LearnerStateRecord,
)
from services.concept_inventory_service import (
    get_concept_inventory_service,
    resolve_concept,
)
from services.learning_map_service import get_learning_map_service, needs_learning_map_compaction


# Mastery math (per root CLAUDE.md):
#   demonstrated: mastery += 0.2 * (1 - mastery)
#   struggled:    mastery *= 0.7
# Classification thresholds for the understood / struggling lists.
_DEMONSTRATE_STEP = 0.2
_STRUGGLE_DECAY = 0.7
_UNDERSTOOD_THRESHOLD = 0.7
_STRUGGLING_THRESHOLD = 0.3


@dataclass(slots=True)
class AssessmentProgressResult:
    state: LearnerState
    mastery_delta: float
    evidence_strength: str


class LearnerTracker:
    """Loads and updates per-conversation learner state.

    Canonical progress is stored by course concept ID, while public
    `LearnerState` fields remain label-based for backwards compatibility.
    """

    async def load_state(self, session: AsyncSession, conversation_id: uuid.UUID) -> LearnerState:
        raw = await self._load_raw(session, conversation_id)
        concepts = await get_concept_inventory_service().load_concepts(session, conversation_id)
        learning_map = get_learning_map_service()
        phases, objectives = await learning_map.load_map(session, conversation_id)
        if not phases or needs_learning_map_compaction(phases, objectives):
            phases, objectives = await learning_map.rebuild_map(
                session,
                conversation_id,
                use_llm=False,
            )
        raw, migrated = self._migrate_legacy_progress(raw, concepts)
        if migrated:
            record = await self._get_or_create(session, conversation_id)
            record.state_json = raw
            await session.flush()
        return self._to_learner_state(conversation_id, raw, concepts, phases, objectives)

    async def apply_updates(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        updates: LearnerUpdates,
        *,
        bump_turn: bool = True,
        allow_mastery_updates: bool = True,
    ) -> LearnerState:
        record = await self._get_or_create(session, conversation_id)
        raw = dict(record.state_json or {})
        concepts = await get_concept_inventory_service().load_concepts(session, conversation_id)
        phases, objectives = await get_learning_map_service().load_map(session, conversation_id)
        raw, _migrated = self._migrate_legacy_progress(raw, concepts)

        if concepts:
            return await self._apply_canonical_updates(
                session,
                conversation_id,
                record,
                raw,
                concepts,
                phases,
                objectives,
                updates,
                bump_turn=bump_turn,
                allow_mastery_updates=allow_mastery_updates,
            )

        return await self._apply_label_updates(
            session,
            conversation_id,
            record,
            raw,
            phases,
            objectives,
            updates,
            bump_turn=bump_turn,
            allow_mastery_updates=allow_mastery_updates,
        )

    async def _apply_canonical_updates(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        record: LearnerStateRecord,
        raw: dict[str, Any],
        concepts: list[CourseConceptRecord],
        phases: list[CourseLearningPhaseRecord],
        objectives: list[CourseLearningObjectiveRecord],
        updates: LearnerUpdates,
        *,
        bump_turn: bool,
        allow_mastery_updates: bool = True,
    ) -> LearnerState:
        mastery: dict[str, float] = _float_map(raw.get("mastery_by_concept_id", {}))
        encounters: dict[str, int] = _int_map(raw.get("_encounters_by_concept_id", {}))
        struggle_evidence: dict[str, int] = _int_map(raw.get("_struggle_evidence_by_concept_id", {}))
        unmapped: dict[str, list[str]] = {
            "covered": list((raw.get("_unmapped_updates") or {}).get("covered", [])),
            "demonstrated": list((raw.get("_unmapped_updates") or {}).get("demonstrated", [])),
            "struggled": list((raw.get("_unmapped_updates") or {}).get("struggled", [])),
        }
        prev_mastery = dict(mastery)

        covered_labels = list(updates.concepts_covered)
        if not allow_mastery_updates:
            covered_labels.extend(updates.concepts_demonstrated)
            covered_labels.extend(updates.concepts_struggled)

        for concept in covered_labels:
            resolved = resolve_concept(concept, concepts)
            if resolved is None:
                _append_unique(unmapped["covered"], concept)
                continue
            concept_id = str(resolved.id)
            encounters[concept_id] = encounters.get(concept_id, 0) + 1
            mastery.setdefault(concept_id, 0.0)

        if allow_mastery_updates:
            for concept in updates.concepts_demonstrated:
                resolved = resolve_concept(concept, concepts)
                if resolved is None:
                    _append_unique(unmapped["demonstrated"], concept)
                    continue
                concept_id = str(resolved.id)
                current = mastery.get(concept_id, 0.0)
                mastery[concept_id] = min(1.0, current + _DEMONSTRATE_STEP * (1.0 - current))
                encounters.setdefault(concept_id, 0)

            for concept in updates.concepts_struggled:
                resolved = resolve_concept(concept, concepts)
                if resolved is None:
                    _append_unique(unmapped["struggled"], concept)
                    continue
                concept_id = str(resolved.id)
                current = mastery.get(concept_id, 0.0)
                mastery[concept_id] = max(0.0, current * _STRUGGLE_DECAY)
                encounters.setdefault(concept_id, 0)
                struggle_evidence[concept_id] = struggle_evidence.get(concept_id, 0) + 1

        session_turns = int(raw.get("session_turns", 0)) + (1 if bump_turn else 0)
        prev_tsp = int(raw.get("turns_since_progress", 0))
        progressed = any(mastery.get(c, 0.0) > prev_mastery.get(c, 0.0) for c in mastery)
        turns_since_progress = 0 if progressed else prev_tsp + (1 if bump_turn else 0)

        raw.update(
            {
                "mastery_by_concept_id": mastery,
                "session_turns": session_turns,
                "turns_since_progress": turns_since_progress,
                "_concept_progress_version": 1,
                "_encounters_by_concept_id": encounters,
                "_struggle_evidence_by_concept_id": struggle_evidence,
                "_unmapped_updates": unmapped,
            }
        )
        self._sync_derived_fields(raw, concepts)
        record.state_json = raw
        await session.flush()
        return self._to_learner_state(conversation_id, raw, concepts, phases, objectives)

    async def _apply_label_updates(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        record: LearnerStateRecord,
        raw: dict[str, Any],
        phases: list[CourseLearningPhaseRecord],
        objectives: list[CourseLearningObjectiveRecord],
        updates: LearnerUpdates,
        *,
        bump_turn: bool,
        allow_mastery_updates: bool = True,
    ) -> LearnerState:
        mastery: dict[str, float] = _float_map(raw.get("mastery_scores", {}))
        encounters: dict[str, int] = _int_map(raw.get("_encounters", {}))
        struggle_evidence: dict[str, int] = _int_map(raw.get("_struggle_evidence", {}))
        prev_mastery = dict(mastery)

        covered_labels = list(updates.concepts_covered)
        if not allow_mastery_updates:
            covered_labels.extend(updates.concepts_demonstrated)
            covered_labels.extend(updates.concepts_struggled)

        for concept in covered_labels:
            if not concept:
                continue
            encounters[concept] = encounters.get(concept, 0) + 1
            mastery.setdefault(concept, 0.0)

        if allow_mastery_updates:
            for concept in updates.concepts_demonstrated:
                if not concept:
                    continue
                current = mastery.get(concept, 0.0)
                mastery[concept] = min(1.0, current + _DEMONSTRATE_STEP * (1.0 - current))
                encounters.setdefault(concept, 0)

            for concept in updates.concepts_struggled:
                if not concept:
                    continue
                current = mastery.get(concept, 0.0)
                mastery[concept] = max(0.0, current * _STRUGGLE_DECAY)
                encounters.setdefault(concept, 0)
                struggle_evidence[concept] = struggle_evidence.get(concept, 0) + 1

        understood = sorted(c for c, m in mastery.items() if m >= _UNDERSTOOD_THRESHOLD)
        struggling = sorted(
            c
            for c, m in mastery.items()
            if m <= _STRUGGLING_THRESHOLD and struggle_evidence.get(c, 0) > 0
        )

        session_turns = int(raw.get("session_turns", 0)) + (1 if bump_turn else 0)
        prev_tsp = int(raw.get("turns_since_progress", 0))
        progressed = any(mastery.get(c, 0.0) > prev_mastery.get(c, 0.0) for c in mastery)
        turns_since_progress = 0 if progressed else prev_tsp + (1 if bump_turn else 0)

        raw.update(
            {
                "understood_concepts": understood,
                "struggling_concepts": struggling,
                "mastery_scores": mastery,
                "session_turns": session_turns,
                "turns_since_progress": turns_since_progress,
                "_encounters": encounters,
                "_struggle_evidence": struggle_evidence,
            }
        )
        record.state_json = raw
        await session.flush()
        return self._to_learner_state(conversation_id, raw, [], phases, objectives)

    async def reset(self, session: AsyncSession, conversation_id: uuid.UUID) -> None:
        record = await self._get_or_create(session, conversation_id)
        record.state_json = {}
        await session.flush()

    async def apply_assessment_result(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        concept_id: uuid.UUID,
        *,
        score: float,
        bloom_level: str = "understand",
        bump_turn: bool = True,
    ) -> AssessmentProgressResult:
        """Apply graded answer evidence to one canonical concept.

        This path is intentionally separate from generator `LearnerUpdates`:
        generated content can expose concepts, but only student answers should
        create assessment evidence.
        """

        record = await self._get_or_create(session, conversation_id)
        raw = dict(record.state_json or {})
        concepts = await get_concept_inventory_service().load_concepts(session, conversation_id)
        phases, objectives = await get_learning_map_service().load_map(session, conversation_id)
        raw, _migrated = self._migrate_legacy_progress(raw, concepts)
        concept = next((item for item in concepts if item.id == concept_id), None)
        if concept is None:
            raise ValueError(f"unknown concept_id for conversation: {concept_id}")

        concept_key = str(concept_id)
        score = max(0.0, min(1.0, float(score)))
        mastery: dict[str, float] = _float_map(raw.get("mastery_by_concept_id", {}))
        encounters: dict[str, int] = _int_map(raw.get("_encounters_by_concept_id", {}))
        struggle_evidence: dict[str, int] = _int_map(raw.get("_struggle_evidence_by_concept_id", {}))
        current = mastery.get(concept_key, 0.0)
        next_mastery = current
        evidence_strength = "weak"

        if score >= 0.85:
            evidence_strength = "strong" if bloom_level in {"apply", "analyze"} else "medium"
            step = 0.25 if evidence_strength == "strong" else 0.16
            next_mastery = current + step * (1.0 - current)
        elif score >= 0.6:
            evidence_strength = "medium"
            next_mastery = current + 0.10 * (1.0 - current)
        else:
            decay = 0.75 if score >= 0.25 else 0.60
            next_mastery = current * decay
            struggle_evidence[concept_key] = struggle_evidence.get(concept_key, 0) + 1

        next_mastery = max(0.0, min(1.0, next_mastery))
        mastery[concept_key] = next_mastery
        encounters[concept_key] = encounters.get(concept_key, 0) + 1

        session_turns = int(raw.get("session_turns", 0)) + (1 if bump_turn else 0)
        prev_tsp = int(raw.get("turns_since_progress", 0))
        progressed = next_mastery > current
        turns_since_progress = 0 if progressed else prev_tsp + (1 if bump_turn else 0)

        raw.update(
            {
                "mastery_by_concept_id": mastery,
                "session_turns": session_turns,
                "turns_since_progress": turns_since_progress,
                "_concept_progress_version": 1,
                "_encounters_by_concept_id": encounters,
                "_struggle_evidence_by_concept_id": struggle_evidence,
            }
        )
        self._sync_derived_fields(raw, concepts)
        record.state_json = raw
        await session.flush()
        return AssessmentProgressResult(
            state=self._to_learner_state(conversation_id, raw, concepts, phases, objectives),
            mastery_delta=next_mastery - current,
            evidence_strength=evidence_strength,
        )

    # --- internals ---

    async def _load_raw(self, session: AsyncSession, conversation_id: uuid.UUID) -> dict[str, Any]:
        result = await session.execute(
            select(LearnerStateRecord).where(LearnerStateRecord.conversation_id == conversation_id)
        )
        record = result.scalar_one_or_none()
        return dict(record.state_json or {}) if record else {}

    async def _get_or_create(
        self, session: AsyncSession, conversation_id: uuid.UUID
    ) -> LearnerStateRecord:
        result = await session.execute(
            select(LearnerStateRecord).where(LearnerStateRecord.conversation_id == conversation_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            record = LearnerStateRecord(conversation_id=conversation_id, state_json={})
            session.add(record)
            await session.flush()
        return record

    def _migrate_legacy_progress(
        self,
        raw: dict[str, Any],
        concepts: list[CourseConceptRecord],
    ) -> tuple[dict[str, Any], bool]:
        if not concepts or raw.get("_concept_progress_version") == 1:
            return raw, False

        legacy_mastery = _float_map(raw.get("mastery_scores", {}))
        if not legacy_mastery:
            raw["_concept_progress_version"] = 1
            self._sync_derived_fields(raw, concepts)
            return raw, True

        mastery: dict[str, float] = _float_map(raw.get("mastery_by_concept_id", {}))
        encounters: dict[str, int] = _int_map(raw.get("_encounters_by_concept_id", {}))
        struggle_evidence: dict[str, int] = _int_map(raw.get("_struggle_evidence_by_concept_id", {}))
        legacy_encounters = _int_map(raw.get("_encounters", {}))
        legacy_struggling = {str(item) for item in raw.get("struggling_concepts", [])}
        unmapped: dict[str, float] = dict(raw.get("_unmapped_mastery_scores", {}))

        for label, score in legacy_mastery.items():
            resolved = resolve_concept(label, concepts)
            if resolved is None:
                unmapped[label] = score
                continue
            concept_id = str(resolved.id)
            mastery[concept_id] = max(score, mastery.get(concept_id, 0.0))
            if label in legacy_encounters:
                encounters[concept_id] = max(encounters.get(concept_id, 0), legacy_encounters[label])
            if label in legacy_struggling:
                struggle_evidence[concept_id] = max(struggle_evidence.get(concept_id, 0), 1)

        raw.update(
            {
                "mastery_by_concept_id": mastery,
                "_encounters_by_concept_id": encounters,
                "_struggle_evidence_by_concept_id": struggle_evidence,
                "_unmapped_mastery_scores": unmapped,
                "_concept_progress_version": 1,
            }
        )
        self._sync_derived_fields(raw, concepts)
        return raw, True

    @staticmethod
    def _sync_derived_fields(raw: dict[str, Any], concepts: list[CourseConceptRecord]) -> None:
        by_id = {str(concept.id): concept for concept in concepts}
        mastery = _float_map(raw.get("mastery_by_concept_id", {}))
        struggle_evidence = _int_map(raw.get("_struggle_evidence_by_concept_id", {}))
        mastery_scores: dict[str, float] = {}
        understood: list[str] = []
        struggling: list[str] = []
        for concept_id, score in mastery.items():
            concept = by_id.get(concept_id)
            if concept is None:
                continue
            mastery_scores[concept.canonical_name] = score
            if score >= _UNDERSTOOD_THRESHOLD:
                understood.append(concept.canonical_name)
            if score <= _STRUGGLING_THRESHOLD and struggle_evidence.get(concept_id, 0) > 0:
                struggling.append(concept.canonical_name)
        raw["mastery_scores"] = dict(sorted(mastery_scores.items()))
        raw["understood_concepts"] = sorted(understood)
        raw["struggling_concepts"] = sorted(struggling)

    @staticmethod
    def _to_learner_state(
        conversation_id: uuid.UUID,
        raw: dict[str, Any],
        concepts: list[CourseConceptRecord],
        phases: list[CourseLearningPhaseRecord] | None = None,
        objectives: list[CourseLearningObjectiveRecord] | None = None,
    ) -> LearnerState:
        phases = phases or []
        objectives = objectives or []
        mastery = _float_map(raw.get("mastery_by_concept_id", {}))
        encounters = _int_map(raw.get("_encounters_by_concept_id", {}))
        struggle_evidence = _int_map(raw.get("_struggle_evidence_by_concept_id", {}))
        objective_progress = _objective_progress(objectives, mastery, encounters, struggle_evidence)
        phase_progress = _phase_progress(phases, objective_progress)
        objectives_by_phase: dict[str, list[str]] = {}
        for objective in objectives:
            objectives_by_phase.setdefault(str(objective.phase_id), []).append(str(objective.id))
        return LearnerState(
            conversation_id=str(conversation_id),
            understood_concepts=list(raw.get("understood_concepts", [])),
            struggling_concepts=list(raw.get("struggling_concepts", [])),
            mastery_scores=dict(raw.get("mastery_scores", {})),
            session_turns=int(raw.get("session_turns", 0)),
            turns_since_progress=int(raw.get("turns_since_progress", 0)),
            known_concepts=[
                {
                    "id": str(concept.id),
                    "name": concept.canonical_name,
                    "aliases": list(concept.aliases or []),
                    "description": concept.description,
                    "bloom_level": concept.bloom_level,
                    "importance": float(concept.importance),
                    "course_parts": list((concept.concept_metadata or {}).get("course_parts") or []),
                }
                for concept in concepts
            ],
            concept_progress=[
                {
                    "concept_id": str(concept.id),
                    "name": concept.canonical_name,
                    "mastery": mastery.get(str(concept.id), 0.0),
                    "encounters": encounters.get(str(concept.id), 0),
                    "struggle_evidence": struggle_evidence.get(str(concept.id), 0),
                }
                for concept in concepts
            ],
            learning_phases=[
                {
                    "id": str(phase.id),
                    "title": phase.title,
                    "summary": phase.summary,
                    "order_index": int(phase.order_index),
                    "objective_ids": objectives_by_phase.get(str(phase.id), []),
                }
                for phase in phases
            ],
            objective_progress=objective_progress,
            phase_progress=phase_progress,
        )


def _objective_progress(
    objectives: list[CourseLearningObjectiveRecord],
    mastery: dict[str, float],
    encounters: dict[str, int],
    struggle_evidence: dict[str, int],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for objective in objectives:
        concept_ids = [str(item) for item in (objective.concept_ids or [])]
        scores = [mastery.get(concept_id, 0.0) for concept_id in concept_ids]
        score = sum(scores) / len(scores) if scores else 0.0
        out.append(
            {
                "objective_id": str(objective.id),
                "phase_id": str(objective.phase_id),
                "objective_text": objective.objective_text,
                "bloom_level": objective.bloom_level,
                "mastery": score,
                "encounters": sum(encounters.get(concept_id, 0) for concept_id in concept_ids),
                "struggle_evidence": sum(struggle_evidence.get(concept_id, 0) for concept_id in concept_ids),
                "concept_ids": concept_ids,
                "order_index": int(objective.order_index),
            }
        )
    return out


def _phase_progress(
    phases: list[CourseLearningPhaseRecord],
    objective_progress: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    objectives_by_phase: dict[str, list[dict[str, Any]]] = {}
    for objective in objective_progress:
        objectives_by_phase.setdefault(str(objective["phase_id"]), []).append(objective)

    out: list[dict[str, Any]] = []
    for phase in phases:
        phase_objectives = objectives_by_phase.get(str(phase.id), [])
        scores = [float(objective["mastery"]) for objective in phase_objectives]
        score = sum(scores) / len(scores) if scores else 0.0
        out.append(
            {
                "phase_id": str(phase.id),
                "title": phase.title,
                "mastery": score,
                "objectives_total": len(phase_objectives),
                "objectives_mastered": sum(1 for item in scores if item >= _UNDERSTOOD_THRESHOLD),
                "struggle_evidence": sum(int(objective["struggle_evidence"]) for objective in phase_objectives),
                "order_index": int(phase.order_index),
            }
        )
    return out


def _float_map(value: object) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, float] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = max(0.0, min(1.0, float(raw)))
        except (TypeError, ValueError):
            continue
    return out


def _int_map(value: object) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    out: dict[str, int] = {}
    for key, raw in value.items():
        try:
            out[str(key)] = max(0, int(raw))
        except (TypeError, ValueError):
            continue
    return out


def _append_unique(values: list[str], value: str) -> None:
    text = str(value or "").strip()
    if text and text not in values:
        values.append(text)


_tracker: LearnerTracker | None = None


def get_learner_tracker() -> LearnerTracker:
    global _tracker
    if _tracker is None:
        _tracker = LearnerTracker()
    return _tracker
