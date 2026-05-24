from __future__ import annotations

import sys
import uuid
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from db.models import (  # noqa: E402
    CourseConceptRecord,
    CourseLearningObjectiveRecord,
    CourseLearningPhaseRecord,
    LearnerStateRecord,
)
from services.concept_inventory_service import stable_concept_id  # noqa: E402
from services.learner_tracker import LearnerTracker  # noqa: E402
from services.learning_map_service import stable_objective_id, stable_phase_id  # noqa: E402
from teacherlm_core.schemas.generator_io import LearnerUpdates  # noqa: E402


CONV_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


class _FakeSession:
    async def flush(self) -> None:
        return None


class LearnerTrackerCanonicalTests(unittest.IsolatedAsyncioTestCase):
    async def test_covered_only_does_not_create_struggling_concept(self) -> None:
        tracker = LearnerTracker()
        record = LearnerStateRecord(conversation_id=CONV_ID, state_json={})
        concept = _concept("Matrix Factorization")

        state = await tracker._apply_canonical_updates(
            _FakeSession(),
            CONV_ID,
            record,
            {},
            [concept],
            [],
            [],
            LearnerUpdates(concepts_covered=["Matrix Factorization"]),
            bump_turn=True,
        )

        self.assertEqual(state.mastery_scores, {"Matrix Factorization": 0.0})
        self.assertEqual(state.struggling_concepts, [])

    async def test_alias_update_resolves_to_same_concept_id(self) -> None:
        tracker = LearnerTracker()
        record = LearnerStateRecord(conversation_id=CONV_ID, state_json={})
        concept = _concept("Singular Value Decomposition", aliases=["SVD"])

        state = await tracker._apply_canonical_updates(
            _FakeSession(),
            CONV_ID,
            record,
            {},
            [concept],
            [],
            [],
            LearnerUpdates(concepts_demonstrated=["SVD"]),
            bump_turn=True,
        )

        self.assertIn("Singular Value Decomposition", state.mastery_scores)
        self.assertGreater(state.mastery_scores["Singular Value Decomposition"], 0)
        self.assertNotIn("SVD", state.mastery_scores)

    async def test_chat_updates_record_exposure_without_mastery(self) -> None:
        tracker = LearnerTracker()
        record = LearnerStateRecord(conversation_id=CONV_ID, state_json={})
        concept = _concept("Singular Value Decomposition", aliases=["SVD"])

        state = await tracker._apply_canonical_updates(
            _FakeSession(),
            CONV_ID,
            record,
            {},
            [concept],
            [],
            [],
            LearnerUpdates(
                concepts_covered=[],
                concepts_demonstrated=["SVD"],
                concepts_struggled=["SVD"],
            ),
            bump_turn=True,
            allow_mastery_updates=False,
        )

        self.assertEqual(state.mastery_scores, {"Singular Value Decomposition": 0.0})
        self.assertEqual(state.struggling_concepts, [])
        self.assertEqual(state.concept_progress[0].encounters, 2)

    def test_legacy_mastery_scores_migrate_to_canonical_concept(self) -> None:
        tracker = LearnerTracker()
        concept = _concept("Singular Value Decomposition", aliases=["SVD"])
        raw = {
            "mastery_scores": {"SVD": 0.8, "Unknown concept": 0.4},
            "_encounters": {"SVD": 2},
        }

        migrated, changed = tracker._migrate_legacy_progress(raw, [concept])
        state = tracker._to_learner_state(CONV_ID, migrated, [concept])

        self.assertTrue(changed)
        self.assertEqual(state.mastery_scores, {"Singular Value Decomposition": 0.8})
        self.assertEqual(migrated["_unmapped_mastery_scores"], {"Unknown concept": 0.4})

    def test_learning_map_progress_aggregates_canonical_concepts(self) -> None:
        tracker = LearnerTracker()
        concept = _concept("Photosynthesis")
        raw = {
            "mastery_by_concept_id": {str(concept.id): 0.8},
            "_encounters_by_concept_id": {str(concept.id): 1},
        }
        phase = _phase("Plant Energy")
        objective = _objective(phase, "Explain photosynthesis", [concept])

        state = tracker._to_learner_state(CONV_ID, raw, [concept], [phase], [objective])

        self.assertEqual(state.learning_phases[0].title, "Plant Energy")
        self.assertEqual(state.objective_progress[0].mastery, 0.8)
        self.assertEqual(state.phase_progress[0].mastery, 0.8)


def _concept(name: str, aliases: list[str] | None = None) -> CourseConceptRecord:
    return CourseConceptRecord(
        id=stable_concept_id(CONV_ID, name),
        conversation_id=CONV_ID,
        canonical_key=name.casefold(),
        canonical_name=name,
        aliases=aliases or [],
        description="",
        bloom_level="understand",
        importance=0.8,
        source_file_ids=["uploads/lecture.pdf"],
        source_section_ids=[],
        source_chunk_ids=[],
        concept_metadata={},
    )


def _phase(title: str) -> CourseLearningPhaseRecord:
    return CourseLearningPhaseRecord(
        id=stable_phase_id(CONV_ID, title),
        conversation_id=CONV_ID,
        phase_key=title.casefold(),
        title=title,
        summary="",
        order_index=0,
        source_file_ids=[],
        source_section_ids=[],
        source_chunk_ids=[],
        phase_metadata={},
    )


def _objective(
    phase: CourseLearningPhaseRecord,
    text: str,
    concepts: list[CourseConceptRecord],
) -> CourseLearningObjectiveRecord:
    return CourseLearningObjectiveRecord(
        id=stable_objective_id(CONV_ID, phase.phase_key, text),
        conversation_id=CONV_ID,
        phase_id=phase.id,
        objective_key=text.casefold(),
        objective_text=text,
        bloom_level="understand",
        order_index=0,
        concept_ids=[str(concept.id) for concept in concepts],
        source_file_ids=[],
        source_section_ids=[],
        source_chunk_ids=[],
        objective_metadata={},
    )


if __name__ == "__main__":
    unittest.main()
