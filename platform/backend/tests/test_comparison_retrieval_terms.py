from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = Path(__file__).resolve().parents[3] / "packages" / "teacherlm_core"
for path in (BACKEND_DIR, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from teacherlm_core.schemas.chunk import Chunk  # noqa: E402

try:
    from services.course_context_service import (  # noqa: E402
        _balanced_term_merge,
        _comparison_terms,
        _merge_formula_hits,
    )
except ModuleNotFoundError:
    _balanced_term_merge = None  # type: ignore[assignment]
    _comparison_terms = None  # type: ignore[assignment]
    _merge_formula_hits = None  # type: ignore[assignment]


class ComparisonRetrievalTermTests(unittest.TestCase):
    def test_extracts_generic_comparison_terms(self) -> None:
        if _comparison_terms is None:
            self.skipTest("backend runtime dependencies are not installed")

        terms = _comparison_terms("what is the difference between mitosis, meiosis, and binary fission?")

        self.assertEqual([term.label for term in terms], ["mitosis", "meiosis", "binary fission"])

    def test_extracts_acronym_comparison_terms_without_known_topic_list(self) -> None:
        if _comparison_terms is None:
            self.skipTest("backend runtime dependencies are not installed")

        terms = _comparison_terms("compare TCP and UDP for transport protocols")

        self.assertEqual([term.label for term in terms], ["TCP", "UDP"])

    def test_balanced_merge_keeps_each_concept_in_front(self) -> None:
        if _balanced_term_merge is None:
            self.skipTest("backend runtime dependencies are not installed")

        term_hits = {
            "mitosis": [_chunk("mitosis-1"), _chunk("mitosis-2")],
            "meiosis": [_chunk("meiosis-1")],
            "binary fission": [_chunk("binary-fission-1")],
        }
        merged = _balanced_term_merge(term_hits, [_chunk("mitosis-1"), _chunk("other")], target=5)

        self.assertEqual([chunk.chunk_id for chunk in merged[:3]], ["mitosis-1", "meiosis-1", "binary-fission-1"])
        self.assertEqual(len({chunk.chunk_id for chunk in merged}), len(merged))
        self.assertEqual(merged[1].metadata["matched_query_term"], "meiosis")

    def test_formula_query_boosts_math_chunks(self) -> None:
        if _merge_formula_hits is None:
            self.skipTest("backend runtime dependencies are not installed")

        text_hit = _chunk("definition")
        formula = Chunk(
            text="Bayes theorem is written as $$ P(A|B)=P(B|A)P(A)/P(B) $$.",
            source="stats.pdf",
            score=0.0,
            chunk_id="formula",
            metadata={"heading_path": "Probability > Bayes theorem"},
        )

        merged = _merge_formula_hits(
            "what is the formula for Bayes theorem?",
            [text_hit],
            [text_hit, formula],
            target=3,
        )

        self.assertEqual(merged[0].chunk_id, "formula")


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(text=chunk_id, source="course", score=1.0, chunk_id=chunk_id, metadata={})


if __name__ == "__main__":
    unittest.main()
