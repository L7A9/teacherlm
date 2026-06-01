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

    def test_force_regenerate_skips_module_pack_fast_path(self) -> None:
        self.assertFalse(
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


def _module(text: str, source: str, order: int) -> Chunk:
    return Chunk(
        text=text,
        source=source,
        score=1,
        chunk_id=source,
        metadata={
            "context_type": "mindmap_module_pack",
            "document_order": order,
            "source_filename": source,
        },
    )


if __name__ == "__main__":
    unittest.main()
