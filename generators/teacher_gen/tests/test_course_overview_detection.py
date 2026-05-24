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

from teacherlm_core.schemas.chunk import Chunk  # noqa: E402
from teacher_gen.pipeline import (  # noqa: E402
    _course_overview_response,
    _has_context_evidence,
    _is_course_overview_question,
    _is_formula_only_question,
)


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
            "what should I study first?",
            "give me a beginner roadmap",
            "prepare me for the exam",
            "teach me this course",
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

    def test_context_evidence_detects_acronym_even_with_negative_rerank_score(self) -> None:
        chunks = [
            Chunk(
                text=(
                    "Section summary: $$ \\hat{r}_{ui} = \\vec{p}_u \\cdot \\vec{q}_i $$"
                ),
                source="Lecture_04_V2.pdf",
                score=-6.2,
                chunk_id="svd-formula",
                metadata={
                    "heading_path": "Rappel : Factorisation de Matrices (SVD)"
                },
            )
        ]

        self.assertTrue(
            _has_context_evidence("explain to me svd and its equations", chunks)
        )

    def test_context_evidence_ignores_generic_formula_terms_for_off_topic_query(self) -> None:
        chunks = [
            Chunk(
                text="Recommendation metrics: $$ RMSE = \\sqrt{...} $$",
                source="Lecture_04_V2.pdf",
                score=-5.8,
                chunk_id="formula",
                metadata={"heading_path": "Metrics"},
            )
        ]

        self.assertFalse(
            _has_context_evidence("explain photosynthesis equations", chunks)
        )

    def test_explain_with_equations_is_not_formula_only(self) -> None:
        self.assertFalse(_is_formula_only_question("explain SVD and its equations"))
        self.assertTrue(_is_formula_only_question("what is the formula for SVD?"))

    def test_overview_response_is_structured_and_uses_synthetic_course_terms(self) -> None:
        chunks = [
            Chunk(
                text=(
                    "Module 1: Cellular Biology\n"
                    "Source file: chapter1.pdf\n"
                    "Major headings:\n- Cell structure\n- Membranes\n"
                    "Study outline details:\n- Cell structure: organelles and their roles\n"
                ),
                source="chapter1.pdf",
                score=1,
                chunk_id="m1",
                metadata={"context_type": "mindmap_module_pack", "document_title": "Cellular Biology"},
            ),
            Chunk(
                text=(
                    "Module 2: Genetics\n"
                    "Source file: chapter2.pdf\n"
                    "Major headings:\n- DNA replication\n- Protein synthesis\n"
                    "Study outline details:\n- DNA replication: copying genetic information\n"
                ),
                source="chapter2.pdf",
                score=1,
                chunk_id="m2",
                metadata={"context_type": "mindmap_module_pack", "document_title": "Genetics"},
            ),
        ]

        response = _course_overview_response(chunks)

        assert response is not None
        self.assertIn("Cellular Biology", response)
        self.assertIn("Genetics", response)
        self.assertIn("## Main path through the course", response)
        self.assertIn("## What to study first", response)
        self.assertNotIn("recommendation systems", response.casefold())

    def test_overview_response_filters_table_and_formula_artifacts(self) -> None:
        chunks = [
            Chunk(
                text=(
                    "Module 1: Qu’est-ce qu’un Système de Recommandation?\n"
                    "Source file: Lecture_01_organized.pdf\n"
                    "Major headings:\n"
                    "- Input from User</th\n"
                    "- Output to User</th\n"
                    "- Filtrage Collaboratif\n"
                    "- $\\vec{h}_t$\n"
                    "Study outline details:\n"
                    "- Ratings</td: Analysis of ratings</td\n"
                    "- Section path: sim(Cours C, Cours A)**\n"
                    "- Filtrage Collaboratif: method based on user-item interactions\n"
                    "$$ R = \\begin{pmatrix} 1 & ? \\\\ 2 & 5 \\end{pmatrix} $$\n"
                ),
                source="Lecture_01_organized.pdf",
                score=1,
                chunk_id="bad-overview",
                metadata={
                    "context_type": "mindmap_module_pack",
                    "document_title": "Systèmes de Recommandation",
                    "key_concepts": [
                        "Input from User</th",
                        "Ratings</td",
                        "Filtrage Collaboratif",
                    ],
                },
            )
        ]

        response = _course_overview_response(chunks)

        assert response is not None
        self.assertIn("Filtrage Collaboratif", response)
        self.assertNotIn("</th", response)
        self.assertNotIn("</td", response)
        self.assertNotIn("Formal pieces", response)
        self.assertNotIn("\\begin{pmatrix}", response)
        self.assertNotIn("Section path", response)


if __name__ == "__main__":
    unittest.main()
