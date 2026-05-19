from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.document_cleaning_service import DocumentCleaningService  # noqa: E402


class DocumentCleaningServiceTests(unittest.TestCase):
    def test_removes_slide_parser_noise_and_keeps_course_content(self) -> None:
        cleaner = DocumentCleaningService()
        cleaned = cleaner.clean_markdown(
            """
![Logo](page_1_image_2_v2.jpg)
<!-- layout: odqx, ytjx -->
Pr. Abdelaaziz Hessane (École Normale Supérieure de Meknès) | Systèmes de Recommandation et Blockchain | November 13, 2025 | 5 / 35

# Neural Collaborative Filtering
**Idée principale :** Remplacer le produit scalaire par un réseau de neurones.
$$ \\hat{r}_{ui} = f(emb_u, emb_i) $$

![Navigation icons](page_7_image_1_v2.jpg)
9.1 Choix de Métrique par Cas d'Usage . . . . . . . . . . . . . . . . . 13
- **Précision@k** : proportion de recommandations pertinentes.
"""
        )

        self.assertNotIn("![", cleaned)
        self.assertNotIn("layout:", cleaned)
        self.assertNotIn("Pr. Abdelaaziz", cleaned)
        self.assertNotIn("Navigation icons", cleaned)
        self.assertNotIn(". . .", cleaned)
        self.assertIn("# Neural Collaborative Filtering", cleaned)
        self.assertIn("Remplacer le produit scalaire", cleaned)
        self.assertIn("$$ \\hat{r}_{ui} = f(emb_u, emb_i) $$", cleaned)
        self.assertIn("Précision@k", cleaned)

    def test_removes_footer_fragments_inside_content_line(self) -> None:
        cleaner = DocumentCleaningService()
        cleaned = cleaner.clean_markdown(
            "Pr. Abdelaaziz Hessane November 13, 2025 15 / 35 **Motivation** : Le comportement utilisateur est séquentiel."
        )

        self.assertNotIn("Abdelaaziz", cleaned)
        self.assertNotIn("15 / 35", cleaned)
        self.assertIn("Motivation", cleaned)
        self.assertIn("séquentiel", cleaned)

    def test_removes_footer_remnants_after_partial_cleanup(self) -> None:
        cleaner = DocumentCleaningService()
        cleaned = cleaner.clean_markdown(
            "Professor Ada Lovelace University of Example May 1, 2026 15 / 35 **Motivation**: Models need evidence.\n"
            "Department of Biology | Intro course | 16 / 35"
        )

        self.assertNotIn("Professor Ada", cleaned)
        self.assertNotIn("Department of Biology", cleaned)
        self.assertIn("Motivation", cleaned)
        self.assertIn("Models need evidence", cleaned)


if __name__ == "__main__":
    unittest.main()
