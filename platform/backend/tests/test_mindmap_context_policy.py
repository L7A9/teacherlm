from __future__ import annotations

import sys
import unittest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


BACKEND_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = Path(__file__).resolve().parents[3] / "packages" / "teacherlm_core"
for path in (BACKEND_DIR, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from services.course_context_service import (  # noqa: E402
        _document_sort_key,
        _is_noisy_section,
        _is_supplemental_document,
        _select_mindmap_sections,
    )
except ModuleNotFoundError:
    _document_sort_key = None  # type: ignore[assignment]
    _is_noisy_section = None  # type: ignore[assignment]
    _is_supplemental_document = None  # type: ignore[assignment]
    _select_mindmap_sections = None  # type: ignore[assignment]


@dataclass
class FakeSection:
    title: str
    level: int = 1
    text: str = "This section has enough source content to be useful for a study map."
    heading_path: list[str] = field(default_factory=list)
    key_concepts: list[str] = field(default_factory=list)
    id: object = field(default_factory=uuid4)


class MindmapContextPolicyTests(unittest.TestCase):
    def test_documents_sort_by_natural_course_sequence_before_supplements(self) -> None:
        if _document_sort_key is None:
            self.skipTest("backend runtime dependencies are not installed")
        created = datetime(2026, 1, 1, tzinfo=timezone.utc)
        filenames = [
            "Guide_for_Students.pdf",
            "Lecture_03.pdf",
            "Lecture_01.pdf",
            "Chapter II.pdf",
            "Week 10.pdf",
        ]

        ordered = sorted(
            filenames,
            key=lambda filename: _document_sort_key(
                filename=filename,
                title=filename,
                created_at=created,
            ),
        )

        self.assertEqual(
            ordered,
            ["Lecture_01.pdf", "Chapter II.pdf", "Lecture_03.pdf", "Week 10.pdf", "Guide_for_Students.pdf"],
        )

    def test_supplemental_documents_are_detected_generically(self) -> None:
        if _is_supplemental_document is None:
            self.skipTest("backend runtime dependencies are not installed")

        self.assertTrue(_is_supplemental_document("appendix_reference.pdf", "Reference notes"))
        self.assertTrue(_is_supplemental_document("Guide_for_Students.pdf", "Student Guide"))
        self.assertFalse(_is_supplemental_document("Lecture_02.pdf", "Core Lesson"))

    def test_noisy_sections_are_filtered_from_mindmap_selection(self) -> None:
        if _is_noisy_section is None or _select_mindmap_sections is None:
            self.skipTest("backend runtime dependencies are not installed")

        noisy = FakeSection("Questions ?", text="Questions ?")
        noisy_path = FakeSection(
            "Table des matières",
            heading_path=["Course Title", "Table des matières"],
            text="Table des matières",
        )
        useful = FakeSection("Causes and Consequences", key_concepts=["causes", "effects"])

        self.assertTrue(_is_noisy_section(noisy))
        self.assertTrue(_is_noisy_section(noisy_path))
        self.assertFalse(_is_noisy_section(useful))
        self.assertEqual(_select_mindmap_sections([noisy, noisy_path, useful], limit=4), [useful])


if __name__ == "__main__":
    unittest.main()
