from __future__ import annotations

import sys
import uuid
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from db.models import (  # noqa: E402
    ChapterAttemptRecord,
    ChapterQuizRecord,
    CourseChapterRecord,
    CourseConceptRecord,
    CourseLearningObjectiveRecord,
    CourseLearningPhaseRecord,
    SearchChunkRecord,
)
from services.concept_inventory_service import stable_concept_id  # noqa: E402
from services.course_player_service import (  # noqa: E402
    PASS_SCORE,
    _CourseBlockCandidate,
    _CourseChapterCandidate,
    _CourseLessonCandidate,
    _assemble_chapters,
    _blocks_from_candidate,
    _chapter_from_phase,
    _chapter_from_candidate,
    _mark_stale_records,
    _lesson_from_candidate,
    _quiz_for_chapter,
)
from services.learning_map_service import stable_objective_id, stable_phase_id  # noqa: E402
from teacherlm_core.schemas.learner_state import LearnerState  # noqa: E402


CONV_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


class CoursePlayerServiceTests(unittest.TestCase):
    def test_chapter_id_is_stable_for_phase(self) -> None:
        phase = _phase("Course Foundations", 0)
        objective = _objective(phase, "Explain foundations")

        first = _chapter_from_phase(CONV_ID, phase, [objective], _now())
        second = _chapter_from_phase(CONV_ID, phase, [objective], _now())

        self.assertEqual(first.id, second.id)
        self.assertEqual(first.phase_id, phase.id)
        self.assertEqual(first.objective_ids, [str(objective.id)])

    def test_pass_threshold_is_seventy_percent(self) -> None:
        self.assertEqual(PASS_SCORE, 0.7)

    def test_best_score_wins_and_marks_completed(self) -> None:
        concept = _concept("Photosynthesis")
        phase = _phase("Plant Energy", 0)
        objective = _objective(phase, "Explain photosynthesis")
        chapter = _chapter_from_phase(CONV_ID, phase, [objective], _now())
        quiz = _quiz_for_chapter(CONV_ID, chapter, _now())
        attempts = [
            _attempt(chapter, quiz, 0.4),
            _attempt(chapter, quiz, 0.8),
            _attempt(chapter, quiz, 0.6),
        ]

        chapters = _assemble_chapters(
            [chapter],
            [],
            [],
            [quiz],
            attempts,
            [],
            [concept],
            LearnerState(conversation_id=str(CONV_ID)),
        )

        self.assertEqual(chapters[0].best_score, 0.8)
        self.assertEqual(chapters[0].state, "completed")

    def test_soft_unlock_makes_locked_chapter_available(self) -> None:
        first = _chapter_from_phase(CONV_ID, _phase("First", 0), [], _now())
        second = _chapter_from_phase(CONV_ID, _phase("Second", 1), [], _now())
        second.state_metadata = {"soft_lock_overridden": True}

        chapters = _assemble_chapters(
            [first, second],
            [],
            [],
            [],
            [],
            [],
            [],
            LearnerState(conversation_id=str(CONV_ID)),
        )

        self.assertEqual(chapters[0].state, "available")
        self.assertEqual(chapters[1].state, "available")
        self.assertTrue(chapters[1].soft_lock_overridden)

    def test_stale_chapters_are_hidden_after_rebuild(self) -> None:
        active = _chapter_from_phase(CONV_ID, _phase("Active", 0), [], _now())
        stale = _chapter_from_phase(CONV_ID, _phase("Stale", 1), [], _now())

        _mark_stale_records({active.id: active, stale.id: stale}, {active.id}, "state_metadata", _now())
        chapters = _assemble_chapters(
            [chapter for chapter in [active, stale] if not chapter.state_metadata.get("inactive")],
            [],
            [],
            [],
            [],
            [],
            [],
            LearnerState(conversation_id=str(CONV_ID)),
        )

        self.assertFalse(active.state_metadata.get("inactive", False))
        self.assertTrue(stale.state_metadata.get("inactive", False))
        self.assertEqual([chapter.id for chapter in chapters], [active.id])

    def test_llm_course_candidate_becomes_structured_chapter_lesson_and_blocks(self) -> None:
        phase = _phase("Foundations", 0)
        objective = _objective(phase, "Explain photosynthesis")
        concept = _concept("Photosynthesis")
        chunk = _chunk("chunk-1")
        candidate = _CourseChapterCandidate(
            title="Foundations of Plant Energy",
            summary="Start with the basic energy idea before applications.",
            phase_id=str(phase.id),
            objective_ids=[str(objective.id)],
            concept_names=["Photosynthesis"],
            source_chunk_ids=[chunk.id],
            lessons=[
                _CourseLessonCandidate(
                    title="What photosynthesis means",
                    summary="A concise lesson on the meaning of photosynthesis.",
                    objective_id=str(objective.id),
                    concept_names=["Photosynthesis"],
                    source_chunk_ids=[chunk.id],
                    blocks=[
                        _CourseBlockCandidate(
                            block_type="definition",
                            title="Definition",
                            content="Photosynthesis is explained here as the plant energy process.",
                            source_chunk_ids=[chunk.id],
                        )
                    ],
                )
            ],
        )

        chapter = _chapter_from_candidate(CONV_ID, candidate, [phase], [objective], [concept], [chunk], _now())
        lesson = _lesson_from_candidate(CONV_ID, chapter.id, candidate.lessons[0], [objective], [concept], [chunk], _now())
        blocks = _blocks_from_candidate(CONV_ID, lesson, candidate.lessons[0], [concept], [chunk], _now())

        self.assertEqual(chapter.phase_id, phase.id)
        self.assertEqual(chapter.objective_ids, [str(objective.id)])
        self.assertEqual(chapter.concept_ids, [str(concept.id)])
        self.assertEqual(lesson.objective_id, objective.id)
        self.assertEqual(blocks[0].block_type, "definition")
        self.assertIn("plant energy process", blocks[0].content)
        self.assertEqual(blocks[0].block_metadata["generation"], "llm_course_plan")


def _phase(title: str, order: int) -> CourseLearningPhaseRecord:
    return CourseLearningPhaseRecord(
        id=stable_phase_id(CONV_ID, title),
        conversation_id=CONV_ID,
        phase_key=title.casefold(),
        title=title,
        summary="",
        order_index=order,
        source_file_ids=[],
        source_section_ids=[],
        source_chunk_ids=[],
        phase_metadata={},
    )


def _objective(phase: CourseLearningPhaseRecord, text: str) -> CourseLearningObjectiveRecord:
    concept = _concept("Photosynthesis")
    return CourseLearningObjectiveRecord(
        id=stable_objective_id(CONV_ID, phase.phase_key, text),
        conversation_id=CONV_ID,
        phase_id=phase.id,
        objective_key=text.casefold(),
        objective_text=text,
        bloom_level="understand",
        order_index=0,
        concept_ids=[str(concept.id)],
        source_file_ids=[],
        source_section_ids=[],
        source_chunk_ids=[],
        objective_metadata={},
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
        source_file_ids=[],
        source_section_ids=[],
        source_chunk_ids=[],
        concept_metadata={},
    )


def _attempt(chapter: CourseChapterRecord, quiz: ChapterQuizRecord, score: float) -> ChapterAttemptRecord:
    return ChapterAttemptRecord(
        conversation_id=CONV_ID,
        chapter_id=chapter.id,
        quiz_id=quiz.id,
        score=score,
        passed=score >= PASS_SCORE,
        answers=[],
        results=[],
        attempt_metadata={},
    )


def _chunk(chunk_id: str) -> SearchChunkRecord:
    return SearchChunkRecord(
        id=chunk_id,
        conversation_id=CONV_ID,
        document_id=uuid.uuid4(),
        section_id=uuid.uuid4(),
        source_filename="lecture.pdf",
        source_file_id="lecture.pdf",
        text="Photosynthesis lets plants convert light into usable energy.",
        chunk_index=0,
        token_count=9,
        heading_path=["Foundations"],
        chunk_metadata={},
    )


def _now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc)


if __name__ == "__main__":
    unittest.main()
