from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.course_structure_service import CourseStructureExtractor  # noqa: E402


class CourseStructureExtractorTests(unittest.TestCase):
    def test_extracts_sections_equations_tables_and_concepts(self) -> None:
        extractor = CourseStructureExtractor()
        document = extractor.extract(
            """
# Recommender Systems

## Matrix Factorization
**Latent factors**: hidden dimensions that explain preferences.
The prediction is $$ \\hat r_{ui} = p_u^T q_i $$.

| Metric | Meaning |
| --- | --- |
| Precision@k | Relevant recommendations in top k |

## Timeline
In 2006, the Netflix Prize popularized collaborative filtering.
""",
            conversation_id="00000000-0000-0000-0000-000000000000",
            source_file_id="uploads/course.pdf",
            source_filename="course.pdf",
        )

        self.assertEqual(document.title, "Recommender Systems")
        self.assertEqual(len(document.sections), 2)
        first = document.sections[0]
        self.assertEqual(first.heading_path, ["Recommender Systems", "Matrix Factorization"])
        self.assertIn("Latent factors", first.key_concepts)
        self.assertTrue(first.equations)
        self.assertTrue(first.tables)
        self.assertIn("Netflix Prize", document.sections[1].timeline_events[0])


if __name__ == "__main__":
    unittest.main()
