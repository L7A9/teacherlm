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

from mindmap_gen.pipeline import (  # noqa: E402
    _apply_fresh_layout_variation,
    _build_from_module_packs,
    _ensure_rich_mindmap,
    _fresh_generation_hint,
    _make_outline_batches,
    _mindmap_response_text,
    _use_module_pack_fast_path,
)
from mindmap_gen.schemas import MindMap, MindMapNode  # noqa: E402


class MindmapModuleBatchTests(unittest.TestCase):
    def test_module_packs_are_batched_per_document_in_document_order(self) -> None:
        chunks = [
            Chunk(
                text="Course document sequence:\n1. Lecture 01\n2. Lecture 02",
                source="course_outline",
                score=1,
                chunk_id="outline",
                metadata={"context_type": "mindmap_course_outline"},
            ),
            _module("Lecture 02 content", "Lecture_02.pdf", 1),
            _module("Lecture 01 content", "Lecture_01.pdf", 0),
        ]

        batches = _make_outline_batches(chunks, max_batches=1)

        self.assertEqual(len(batches), 2)
        self.assertIn("GLOBAL COURSE SEQUENCE", batches[0])
        self.assertIn("Lecture 01 content", batches[0])
        self.assertIn("Lecture 02 content", batches[1])

    def test_regular_chunks_still_use_char_batched_fallback(self) -> None:
        chunks = [
            Chunk(text="A" * 40, source="a.pdf", score=1, chunk_id="a", metadata={}),
            Chunk(text="B" * 40, source="b.pdf", score=1, chunk_id="b", metadata={}),
        ]

        batches = _make_outline_batches(chunks, max_chars=60, max_batches=4)

        self.assertEqual(len(batches), 2)

    def test_module_pack_fallback_builds_course_wide_branches(self) -> None:
        chunks = [
            _module(
                "Module 1: Week 1: Foundations\nMajor headings:\n- Definitions\n- Objectives\nStudy outline details:\n- Definitions: core terms",
                "week1.pdf",
                0,
            ),
            _module(
                "Module 2: Week 2: Methods\nMajor headings:\n- Method A\n- Method B\nStudy outline details:\n- Method A: procedure",
                "week2.pdf",
                1,
            ),
            _module(
                "Module 3: Week 3: Evaluation\nMajor headings:\n- Metrics\n- Limitations\nStudy outline details:\n- Metrics: scoring",
                "week3.pdf",
                2,
            ),
        ]

        mindmap = _build_from_module_packs(chunks, max_nodes=30)

        self.assertIsNotNone(mindmap)
        assert mindmap is not None
        self.assertEqual(
            [branch.text for branch in mindmap.branches],
            ["Foundations", "Methods", "Evaluation"],
        )
        self.assertIn("Definitions", [child.text for child in mindmap.branches[0].children])

    def test_module_pack_fallback_uses_week_titles_and_filters_slide_noise(self) -> None:
        chunks = [
            _module(
                "\n".join(
                    [
                        "Module 1: Lecture 01 organized",
                        "Major headings:",
                        "- Lecture 01 organized > Semaine 1 : Fondements des Systemes de Recommandation (SR) et Applications Educatives > Source material",
                        "- Lecture 01 organized > Le Probleme : La Surcharge Informationnelle",
                        "- Le Probleme : La Surcharge Informationnelle > PEOPLE WHO BOUGHT",
                        "- Les Donnees : Le Carburant des SR",
                        "Study outline details:",
                        "- Le Probleme : La Surcharge Informationnelle: overload makes recommendation systems useful.",
                    ]
                ),
                "Lecture_01_organized.pdf",
                0,
                title="Lecture 01 organized",
            ),
            _module(
                "\n".join(
                    [
                        "Module 2: Lecture 02",
                        "Major headings:",
                        "- Lecture 02 > Semaine 2 : Le Filtrage Collaboratif (CF) : Des Voisins aux Modeles Latents > Introduction",
                        "- Les Deux Grandes Familles du CF > Memory-Based",
                        "- Les Deux Grandes Familles du CF > Model-Based",
                        "Study outline details:",
                        "- Memory-Based: user and item neighborhoods.",
                    ]
                ),
                "Lecture_02.pdf",
                1,
                title="Lecture 02",
            ),
            _module(
                "\n".join(
                    [
                        "Module 3: Lecture 03 V2 organized",
                        "Major headings:",
                        "- Lecture 03 V2 organized > Semaine 3 : Approches Basees sur le Contenu et Systemes Hybrides > Source material",
                        "- Architecture d'un Systeme CBF",
                        "- Architecture d'un Systeme CBF > Layout Attribution (Critical)",
                        "Study outline details:",
                        "- Architecture d'un Systeme CBF: content analyzer and profile learner.",
                    ]
                ),
                "Lecture_03_V2_organized.pdf",
                2,
                title="Lecture 03 V2 organized",
            ),
        ]

        mindmap = _build_from_module_packs(chunks, max_nodes=45)

        self.assertIsNotNone(mindmap)
        assert mindmap is not None
        self.assertEqual(mindmap.central_topic, "Systemes de Recommandation")
        self.assertEqual(
            [branch.text for branch in mindmap.branches],
            [
                "Fondements des Systemes de Recommandation (SR) et Applications Educatives",
                "Le Filtrage Collaboratif (CF) : Des Voisins aux Modeles Latents",
                "Approches Basees sur le Contenu et Systemes Hybrides",
            ],
        )
        labels = _flatten_labels(mindmap)
        self.assertNotIn("Lecture 01 organized", labels)
        self.assertNotIn("PEOPLE WHO BOUGHT", labels)
        self.assertNotIn("Source material", labels)
        self.assertNotIn("Layout Attribution (Critical)", labels)

    def test_force_regenerate_keeps_module_pack_fast_path(self) -> None:
        self.assertTrue(
            _use_module_pack_fast_path(
                has_module_packs=True,
                llm_refine=False,
                force_regenerate=True,
            )
        )
        self.assertTrue(
            _use_module_pack_fast_path(
                has_module_packs=True,
                llm_refine=False,
                force_regenerate=False,
            )
        )

    def test_generation_hint_changes_with_generation_id(self) -> None:
        first = _fresh_generation_hint({"generation_id": "run-a"})
        second = _fresh_generation_hint({"generation_id": "run-b"})

        self.assertIn("Fresh regeneration request", first)
        self.assertNotEqual(first, second)

    def test_fallback_variation_rotates_layout_for_fresh_run(self) -> None:
        mindmap = _build_from_module_packs(
            [
                _module("Module 1: Foundations\nMajor headings:\n- A", "a.pdf", 0),
                _module("Module 2: Methods\nMajor headings:\n- B", "b.pdf", 1),
                _module("Module 3: Evaluation\nMajor headings:\n- C", "c.pdf", 2),
            ],
            max_nodes=30,
        )

        assert mindmap is not None
        original = [branch.text for branch in mindmap.branches]
        varied = _apply_fresh_layout_variation(mindmap, "fresh-run")

        self.assertCountEqual([branch.text for branch in varied.branches], original)
        self.assertNotEqual([branch.text for branch in varied.branches], original)

    def test_generation_sequence_changes_top_level_positions(self) -> None:
        chunks = [
            _module("Module 1: Foundations\nMajor headings:\n- A", "a.pdf", 0),
            _module("Module 2: Methods\nMajor headings:\n- B", "b.pdf", 1),
            _module("Module 3: Evaluation\nMajor headings:\n- C", "c.pdf", 2),
        ]
        first = _build_from_module_packs(chunks, max_nodes=30)
        second = _build_from_module_packs(chunks, max_nodes=30)

        assert first is not None
        assert second is not None
        first = _apply_fresh_layout_variation(first, "mindmap:conversation:1:first")
        second = _apply_fresh_layout_variation(second, "mindmap:conversation:2:second")

        self.assertNotEqual(
            [branch.text for branch in first.branches],
            [branch.text for branch in second.branches],
        )

    def test_flat_outline_is_enriched_from_source_structure(self) -> None:
        chunks = [
            _module(
                "Module 1: Understanding Recommender Systems\n"
                "Major headings:\n"
                "- Definitions\n"
                "- Multiple Objectives\n"
                "- User Feedback\n",
                "week1.pdf",
                0,
            ),
            _module(
                "Module 2: Fuel for RS Data\n"
                "Major headings:\n"
                "- Explicit Data\n"
                "- Implicit Signals\n"
                "- Ratings Matrix\n",
                "week2.pdf",
                1,
            ),
            _module(
                "Module 3: Recommendation Approaches\n"
                "Major headings:\n"
                "- Collaborative Filtering\n"
                "- Content-Based Filtering\n"
                "- Hybrid Systems\n",
                "week3.pdf",
                2,
            ),
        ]
        flat = MindMap(
            central_topic="Recommender Systems Course",
            branches=[
                MindMapNode(text="Understanding Recommender Systems", children=[]),
                MindMapNode(text="Fuel for RS Data", children=[]),
                MindMapNode(text="Recommendation Approaches", children=[]),
            ],
        )

        enriched = _ensure_rich_mindmap(flat, chunks, max_nodes=40)

        self.assertGreaterEqual(sum(len(branch.children) for branch in enriched.branches), 6)
        self.assertIn("Definitions", [child.text for child in enriched.branches[0].children])
        self.assertIn("Explicit Data", [child.text for child in enriched.branches[1].children])

    def test_fallback_variation_changes_wording_without_losing_nodes(self) -> None:
        mindmap = MindMap(
            central_topic="Systemes de Recommandation",
            branches=[
                MindMapNode(
                    text="Fondements des Systemes de Recommandation",
                    children=[
                        MindMapNode(text="Qu'est-ce qu'un Systeme de Recommandation?", children=[]),
                        MindMapNode(text="Pourquoi la RMSE ne suffit pas?", children=[]),
                    ],
                ),
                MindMapNode(
                    text="Approches Basees sur le Contenu",
                    children=[
                        MindMapNode(text="Introduction au Filtrage Collaboratif", children=[]),
                    ],
                ),
                MindMapNode(
                    text="Evaluation Avancee, Ethique et Deploiement",
                    children=[],
                ),
            ],
        )
        original_count = len(_flatten_labels(mindmap))

        varied = _apply_fresh_layout_variation(mindmap, "fresh-wording")
        varied_labels = _flatten_labels(varied)

        self.assertEqual(len(varied_labels), original_count)
        self.assertIn("Bases des Systemes de Recommandation", varied_labels)
        self.assertIn("Comprendre Systeme de Recommandation", varied_labels)
        self.assertIn("Methodes basees sur le Contenu", varied_labels)
        self.assertNotIn("Fondements des Systemes de Recommandation", varied_labels)

    def test_mindmap_response_uses_forced_language_template(self) -> None:
        mindmap = MindMap(
            central_topic="Reseaux",
            branches=[
                MindMapNode(text="TCP", children=[]),
                MindMapNode(text="IP", children=[]),
                MindMapNode(text="DNS", children=[]),
            ],
        )

        response = _mindmap_response_text(mindmap, 4, "fr-fr")

        self.assertIn("carte mentale", response)
        self.assertIn("Reseaux", response)
        self.assertNotIn("I've built", response)


def _module(
    text: str,
    source: str,
    order: int,
    *,
    title: str | None = None,
) -> Chunk:
    metadata = {
        "context_type": "mindmap_module_pack",
        "document_order": order,
        "source_filename": source,
    }
    if title is not None:
        metadata["document_title"] = title
    return Chunk(
        text=text,
        source=source,
        score=1,
        chunk_id=source,
        metadata=metadata,
    )


def _flatten_labels(mindmap: MindMap) -> set[str]:
    labels = {mindmap.central_topic}

    def walk(node: MindMapNode) -> None:
        labels.add(node.text)
        for child in node.children:
            walk(child)

    for branch in mindmap.branches:
        walk(branch)
    return labels


if __name__ == "__main__":
    unittest.main()
