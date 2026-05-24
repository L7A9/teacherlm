from __future__ import annotations

import uuid
import unittest

from db.models import CourseBuilderChapterRecord, SearchChunkRecord
from services.coursebuilder_service import (
    PASS_SCORE,
    _fallback_outline,
    _fallback_questions,
    _selected_index,
    _stable_id,
)


CONV_ID = uuid.UUID("00000000-0000-0000-0000-000000000000")


class CourseBuilderServiceTests(unittest.TestCase):
    def test_stable_ids_are_deterministic(self) -> None:
        first = _stable_id(CONV_ID, "chapter:0:Foundations")
        second = _stable_id(CONV_ID, " chapter:0:foundations ")

        self.assertEqual(first, second)

    def test_pass_threshold_is_seventy_percent(self) -> None:
        self.assertEqual(PASS_SCORE, 0.7)

    def test_fallback_outline_uses_source_hierarchy_as_ordered_chapters(self) -> None:
        chunks = [
            _chunk("chunk-1", ["Foundations", "Definition"], "Foundational explanation."),
            _chunk("chunk-2", ["Applications", "Example"], "Applied example."),
        ]

        outline = _fallback_outline(chunks)

        self.assertEqual([chapter.title for chapter in outline.chapters], ["Foundations", "Applications"])
        self.assertEqual(outline.chapters[0].lessons[0].title, "Definition")

    def test_fallback_quiz_is_grounded_in_chunk_ids(self) -> None:
        chunk = _chunk("chunk-1", ["Foundations"], "A supported statement from the uploaded file.")
        chapter = CourseBuilderChapterRecord(
            id=_stable_id(CONV_ID, "chapter"),
            course_id=_stable_id(CONV_ID, "course"),
            conversation_id=CONV_ID,
            title="Foundations",
            order_index=0,
        )

        questions = _fallback_questions(chapter, [chunk])

        self.assertEqual(questions[0].correct_index, 0)
        self.assertEqual(questions[0].source_chunk_ids, ["chunk-1"])

    def test_selected_index_accepts_index_label_or_option_text(self) -> None:
        options = ["Alpha", "Beta", "Gamma"]

        self.assertEqual(_selected_index(1, options), 1)
        self.assertEqual(_selected_index("B", options), 1)
        self.assertEqual(_selected_index("Gamma", options), 2)
        self.assertIsNone(_selected_index("", options))


def _chunk(chunk_id: str, heading_path: list[str], text: str) -> SearchChunkRecord:
    return SearchChunkRecord(
        id=chunk_id,
        conversation_id=CONV_ID,
        document_id=uuid.uuid4(),
        section_id=uuid.uuid4(),
        source_filename="course.pdf",
        source_file_id="course.pdf",
        text=text,
        chunk_index=0,
        token_count=16,
        heading_path=heading_path,
        chunk_metadata={},
    )
