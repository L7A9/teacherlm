from __future__ import annotations

import sys
import unittest
from pathlib import Path


GENERATORS_DIR = Path(__file__).resolve().parents[2]
CORE_DIR = Path(__file__).resolve().parents[3] / "packages" / "teacherlm_core"
for path in (GENERATORS_DIR, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from teacherlm_core.schemas.chunk import Chunk  # noqa: E402
from quiz_gen.services.concept_extractor import _fallback_concepts  # noqa: E402


class QuizConceptFallbackTests(unittest.TestCase):
    def test_fallback_extracts_domain_agnostic_course_concepts(self) -> None:
        chunks = [
            Chunk(
                text=(
                    "Photosynthesis: process by which plants convert light energy.\n"
                    "Cellular respiration: process that releases energy from glucose."
                ),
                source="biology.pdf",
                score=1,
                chunk_id="bio-1",
                metadata={
                    "section_title": "Plant Energy Systems",
                    "key_concepts": ["Photosynthesis", "Cellular respiration"],
                },
            )
        ]

        concepts = _fallback_concepts(chunks)
        names = {
            card.name
            for level in ("remember", "understand", "apply", "analyze")
            for card in getattr(concepts, level)
        }

        self.assertIn("Photosynthesis", names)
        self.assertIn("Cellular respiration", names)


if __name__ == "__main__":
    unittest.main()
