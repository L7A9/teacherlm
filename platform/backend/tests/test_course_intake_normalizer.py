from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from config import Settings  # noqa: E402
from services.chunking_service import ChunkingService  # noqa: E402
from services.course_intake_normalizer import CourseIntakeNormalizer  # noqa: E402
from services.course_structure_service import CourseStructureExtractor  # noqa: E402


class CourseIntakeNormalizerTests(unittest.TestCase):
    def test_merged_lecture_pdf_is_split_into_primary_units_and_supplemental_guide(self) -> None:
        intake = CourseIntakeNormalizer().normalize(
            raw_markdown=_merged_course_markdown(),
            cleaned_markdown=_merged_course_markdown(),
            source_filename="rs_course.pdf",
        )

        self.assertTrue(intake.normalized)
        self.assertEqual(intake.metadata["primary_unit_count"], 5)
        self.assertEqual(intake.metadata["supplemental_unit_count"], 1)
        self.assertEqual([unit.unit_number for unit in intake.units[:5]], [1, 2, 3, 4, 5])
        self.assertEqual(intake.units[-1].role, "supplemental")
        self.assertIn("Semaine 4", intake.units[3].title)

    def test_plan_de_la_seance_items_become_subchapters(self) -> None:
        intake = CourseIntakeNormalizer().normalize(
            raw_markdown=_merged_course_markdown(),
            cleaned_markdown=_merged_course_markdown(),
            source_filename="rs_course.pdf",
        )

        first = intake.units[0]
        self.assertEqual(
            [item.title for item in first.subchapters],
            [
                "Le probleme de la surcharge informationnelle",
                "Definition et objectifs d'un systeme de recommandation",
            ],
        )

    def test_plan_de_la_seance_stops_before_slide_body_numbered_noise(self) -> None:
        markdown = """
Semaine 1 : Fondements des Systemes de Recommandation
Plan de la seance
1. Le probleme de la surcharge informationnelle
2. Definition et objectifs d'un systeme de recommandation
3. Les donnees au coeur des SR
Le Probleme
1 Judson Meinhart | Behavioral Finance, Millennial
2 PEOPLE WHO BOUGHT
3 jars and
4 % sales, right table has
"""

        intake = CourseIntakeNormalizer().normalize(
            raw_markdown=markdown,
            cleaned_markdown=markdown,
            source_filename="course.pdf",
        )

        self.assertEqual(
            [item.title for item in intake.units[0].subchapters],
            [
                "Le probleme de la surcharge informationnelle",
                "Definition et objectifs d'un systeme de recommandation",
                "Les donnees au coeur des SR",
            ],
        )

    def test_normalized_metadata_flows_to_sections_and_chunks(self) -> None:
        intake = CourseIntakeNormalizer().normalize(
            raw_markdown=_merged_course_markdown(),
            cleaned_markdown=_merged_course_markdown(),
            source_filename="rs_course.pdf",
        )
        document = CourseStructureExtractor().extract(
            intake.markdown,
            conversation_id="00000000-0000-0000-0000-000000000000",
            source_file_id="uploads/rs_course.pdf",
            source_filename="rs_course.pdf",
            intake_metadata=intake.metadata,
            infer_plain_headings=not intake.normalized,
        )
        chunks = ChunkingService(Settings(chunk_max_tokens=120, chunk_overlap_tokens=0)).chunk_course_document(
            document,
            source_file_id="uploads/rs_course.pdf",
        )

        self.assertTrue(document.metadata["intake_normalized"])
        self.assertTrue(any(section.metadata.get("course_unit_title", "").startswith("Semaine 3") for section in document.sections))
        self.assertTrue(any(chunk.metadata.get("course_unit_role") == "supplemental" for chunk in chunks))
        self.assertTrue(any("subchapter_titles" in chunk.metadata for chunk in chunks))

    def test_duplicate_title_slide_does_not_create_duplicate_unit(self) -> None:
        markdown = """
Semaine 1 : Foundations
Plan de la seance
1. Intro
Semaine 1 : Foundations
More repeated footer text.
Semaine 2 : Retrieval
Plan de la seance
1. Search
"""

        intake = CourseIntakeNormalizer().normalize(
            raw_markdown=markdown,
            cleaned_markdown=markdown,
            source_filename="course.pdf",
        )

        self.assertEqual([unit.unit_number for unit in intake.units], [1, 2])


def _merged_course_markdown() -> str:
    return """
Systemes de Recommandation et Blockchain
Semaine 1 : Fondements des Systemes de Recommandation Pr. Teacher
Plan de la seance
1.Le probleme de la surcharge informationnelle :Pourquoi avons-nous besoin de recommandations?
2.Definition et objectifs d'un systeme de recommandation

Body for week one.

Semaine 2 : Le Filtrage Collaboratif (CF) : Des Voisins aux Modeles Latents Pr. Teacher
Plan de la seance
1.Introduction au Filtrage Collaboratif (CF) :L'intelligence collective.
2.CF Base sur les Voisins (Memory-Based)

Body for week two.

Semaine 3 : Approches Basees sur le Contenu et Systemes Hybrides Pr. Teacher
Plan de la seance
1.Introduction au Filtrage Base sur le Contenu
2.Architecture d'un systeme CBF

Body for week three.

Semaine 4 : L'Ere du Deep Learning dans la Recommandation Pr. Teacher
Plan de la seance
1.Les limites des modeles traditionnels
2.Le Deep Learning au service de la recommandation

Body for week four.

Semaine 5 : Evaluation Avancee, Ethique et Deploiement Pr. Teacher
Plan de la seance
1.Au-dela de la RMSE
2.Metriques de Classement

Body for week five.

Guide Complet d'Evaluation des Systemes de Recommandation
Table des matieres
1 Introduction 4
2 Pourquoi la RMSE ne suffit pas 5
"""


if __name__ == "__main__":
    unittest.main()
