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

from mindmap_gen.pipeline import _build_from_module_packs, _make_outline_batches  # noqa: E402


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
