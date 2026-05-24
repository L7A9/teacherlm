from __future__ import annotations

import sys
import uuid
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from db.models import AnsweredCourseQuestionRecord, CourseConceptRecord  # noqa: E402
from services.concept_inventory_service import stable_concept_id  # noqa: E402
from services.review_test_service import (  # noqa: E402
    WINDOW_SIZE,
    _review_question_count,
    _window_due,
    _window_from_questions,
)


CONV_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


class ReviewTestServiceTests(unittest.TestCase):
    def test_ten_answered_questions_create_pending_window_payload(self) -> None:
        questions = [_answered_question(index) for index in range(WINDOW_SIZE)]

        window = _window_from_questions(CONV_ID, questions, total_answered=10)

        self.assertEqual(window.status, "pending")
        self.assertEqual(window.answer_count, 10)
        self.assertEqual(len(window.answered_question_ids), 10)
        self.assertEqual(window.concept_ids, [str(_concept_id(0)), str(_concept_id(1))])

    def test_snoozed_window_reappears_after_two_more_answers(self) -> None:
        window = _window_from_questions(CONV_ID, [_answered_question(index) for index in range(10)], 10)
        window.status = "snoozed"
        window.snooze_until_count = 12

        self.assertFalse(_window_due(window, 11))
        self.assertTrue(_window_due(window, 12))

    def test_review_question_count_scales_with_richness(self) -> None:
        window = _window_from_questions(CONV_ID, [_answered_question(index) for index in range(10)], 10)
        concepts = [_concept(index) for index in range(8)]
        window.objective_ids = [str(uuid.uuid4()) for _ in range(6)]

        self.assertEqual(_review_question_count(window, concepts), 7)

    def test_sparse_review_uses_available_concepts(self) -> None:
        window = _window_from_questions(CONV_ID, [_answered_question(0)], 1)
        concepts = [_concept(0)]

        self.assertEqual(_review_question_count(window, concepts), 1)


def _answered_question(index: int) -> AnsweredCourseQuestionRecord:
    return AnsweredCourseQuestionRecord(
        id=uuid.uuid4(),
        conversation_id=CONV_ID,
        user_message_id=uuid.uuid4(),
        assistant_message_id=uuid.uuid4(),
        concept_ids=[str(_concept_id(index % 2))],
        objective_ids=[str(uuid.uuid4())],
        phase_ids=[str(uuid.uuid4())],
        source_chunk_ids=[f"chunk-{index}"],
        question_metadata={},
    )


def _concept(index: int) -> CourseConceptRecord:
    name = f"Concept {index}"
    return CourseConceptRecord(
        id=_concept_id(index),
        conversation_id=CONV_ID,
        canonical_key=name.casefold(),
        canonical_name=name,
        aliases=[],
        description="",
        bloom_level="understand",
        importance=0.8,
        source_file_ids=[],
        source_section_ids=[],
        source_chunk_ids=[],
        concept_metadata={},
    )


def _concept_id(index: int) -> uuid.UUID:
    return stable_concept_id(CONV_ID, f"Concept {index}")


if __name__ == "__main__":
    unittest.main()
