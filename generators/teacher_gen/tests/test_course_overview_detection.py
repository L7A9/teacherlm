from __future__ import annotations

import sys
import unittest
from pathlib import Path


GENERATOR_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = GENERATOR_DIR.parents[1]
CORE_DIR = REPO_ROOT / "packages" / "teacherlm_core"
for path in (GENERATOR_DIR.parent, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from teacher_gen.pipeline import _is_course_overview_question  # noqa: E402


class CourseOverviewDetectionTests(unittest.TestCase):
    def test_student_style_missing_verb_question_is_overview(self) -> None:
        self.assertTrue(_is_course_overview_question("what this course about?"))

    def test_common_overview_variants_are_overview(self) -> None:
        examples = [
            "what is this course about?",
            "what's this course about?",
            "explain the course",
            "summarize the course",
            "what are these files about?",
        ]
        for example in examples:
            with self.subTest(example=example):
                self.assertTrue(_is_course_overview_question(example))

    def test_specific_question_is_not_overview(self) -> None:
        self.assertFalse(
            _is_course_overview_question(
                "How is Pearson correlation used in collaborative filtering?"
            )
        )


if __name__ == "__main__":
    unittest.main()
