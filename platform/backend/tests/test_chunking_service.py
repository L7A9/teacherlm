from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import Settings  # noqa: E402
from services.chunking_service import ChunkingService  # noqa: E402


class ChunkingServiceTests(unittest.TestCase):
    def test_hierarchical_metadata_and_neighbors_are_added(self) -> None:
        chunker = ChunkingService(
            Settings(chunk_max_tokens=30, chunk_overlap_tokens=0)
        )
        chunks = chunker.chunk_text(
            """
# Introduction
Les systemes de recommandation aident les apprenants a trouver des ressources utiles.
Ils comparent les profils, les contenus et les traces d'activite.

# Architecture
Un pipeline typique contient la collecte, le filtrage, le classement et l'evaluation.
La precision mesure la proportion de recommandations utiles.
Le rappel mesure la proportion d'elements pertinents retrouves.
""",
            source="course.md",
        )

        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0].metadata["chunker"], "structured-section-v1")
        self.assertIn("Introduction", chunks[0].metadata["heading_path"])
        self.assertIn("section_id", chunks[0].metadata)
        self.assertIn("section_summary", chunks[0].metadata)
        self.assertEqual(chunks[0].metadata["next_chunk_id"], chunks[1].chunk_id)
        self.assertEqual(chunks[1].metadata["prev_chunk_id"], chunks[0].chunk_id)

    def test_plain_headings_are_preserved_without_becoming_content(self) -> None:
        chunker = ChunkingService(
            Settings(chunk_max_tokens=80, chunk_overlap_tokens=0)
        )
        chunks = chunker.chunk_text(
            """
Systemes intelligents pour l education

Les tuteurs intelligents utilisent un modele de l'apprenant.
Ce modele represente les connaissances, les difficultes et les progres.
""",
            source="course.md",
        )

        self.assertEqual(len(chunks), 1)
        self.assertIn("Systemes intelligents", chunks[0].metadata["heading_path"])
        self.assertNotIn("Systemes intelligents pour l education", chunks[0].text)

    def test_dates_and_page_counters_are_not_headings(self) -> None:
        chunker = ChunkingService(
            Settings(chunk_max_tokens=80, chunk_overlap_tokens=0)
        )
        chunks = chunker.chunk_text(
            """
November 13, 2025

5 / 35

# Métriques d'Évaluation
La précision mesure la proportion de recommandations pertinentes.
""",
            source="course.md",
        )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].metadata["heading_path"], "Métriques d'Évaluation")
        self.assertNotIn("November", chunks[0].metadata["heading_path"])


    def test_chunk_ids_are_stable_for_same_source_and_text(self) -> None:
        chunker = ChunkingService(
            Settings(chunk_max_tokens=80, chunk_overlap_tokens=0)
        )
        text = """
# Evaluation
La precision et le rappel mesurent la qualite des recommandations.
"""

        first = chunker.chunk_text(text, source="course.md")
        second = chunker.chunk_text(text, source="course.md")

        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])


if __name__ == "__main__":
    unittest.main()
