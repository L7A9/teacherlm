from __future__ import annotations

import sys
import uuid
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from db.models import CourseConceptRecord, CourseSectionRecord, SearchChunkRecord  # noqa: E402
from services.concept_inventory_service import stable_concept_id  # noqa: E402
from services.learning_map_service import LearningMapService  # noqa: E402


CONV_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")
DOC_ID = uuid.UUID("11111111-1111-1111-1111-111111111111")
SECTION_ID = uuid.UUID("22222222-2222-2222-2222-222222222222")


class LearningMapServiceTests(unittest.TestCase):
    def test_fallback_builds_domain_general_phase_and_objective(self) -> None:
        service = LearningMapService()
        section = _section(
            "Plant Energy Systems",
            "Photosynthesis converts light energy into chemical energy.",
        )
        chunk = _chunk("Photosynthesis: plants convert light energy into glucose.")
        concept = _concept("Photosynthesis")

        candidates = service._fallback_candidates([section], [chunk], [concept])
        phases, objectives = service._merge_candidates(candidates, [chunk], [concept])

        self.assertEqual(phases[0].title, "Plant Energy Systems")
        self.assertIn("Photosynthesis", objectives[0].objective_text)
        self.assertEqual(objectives[0].concept_ids, {str(concept.id)})

    def test_fallback_is_not_recommender_specific(self) -> None:
        service = LearningMapService()
        section = _section("Due Process", "Due process protects legal rights before state action.")
        chunk = _chunk("Due Process: legal protection before the state removes life, liberty, or property.")
        concept = _concept("Due Process")

        candidates = service._fallback_candidates([section], [chunk], [concept])
        phases, objectives = service._merge_candidates(candidates, [chunk], [concept])

        self.assertEqual(phases[0].title, "Due Process")
        self.assertEqual(objectives[0].concept_ids, {str(concept.id)})


def _section(title: str, text: str) -> CourseSectionRecord:
    return CourseSectionRecord(
        id=SECTION_ID,
        conversation_id=CONV_ID,
        document_id=DOC_ID,
        parent_section_id=None,
        level=1,
        title=title,
        heading_path=[title],
        order_index=0,
        text=text,
        summary=text,
        key_concepts=[],
        equations=[],
        tables=[],
        timeline_events=[],
        section_metadata={},
    )


def _chunk(text: str) -> SearchChunkRecord:
    return SearchChunkRecord(
        id="chunk-1",
        conversation_id=CONV_ID,
        document_id=DOC_ID,
        section_id=SECTION_ID,
        source_filename="course.pdf",
        source_file_id="uploads/course.pdf",
        text=text,
        chunk_index=0,
        token_count=20,
        heading_path=["Course"],
        chunk_metadata={"section_title": "Course"},
    )


def _concept(name: str) -> CourseConceptRecord:
    return CourseConceptRecord(
        id=stable_concept_id(CONV_ID, name),
        conversation_id=CONV_ID,
        canonical_key=name.casefold(),
        canonical_name=name,
        aliases=[],
        description="",
        bloom_level="understand",
        importance=0.8,
        source_file_ids=["uploads/course.pdf"],
        source_section_ids=[str(SECTION_ID)],
        source_chunk_ids=["chunk-1"],
        concept_metadata={},
    )


if __name__ == "__main__":
    unittest.main()
