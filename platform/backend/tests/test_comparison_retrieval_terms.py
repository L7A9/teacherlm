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
    )
except ModuleNotFoundError:
    _balanced_term_merge = None  # type: ignore[assignment]
    _comparison_terms = None  # type: ignore[assignment]


class ComparisonRetrievalTermTests(unittest.TestCase):
    def test_extracts_known_comparison_terms(self) -> None:
        if _comparison_terms is None:
            self.skipTest("backend runtime dependencies are not installed")

        terms = _comparison_terms("what is the difference between svd, rnn, and ncf")

        self.assertEqual([term.label for term in terms], ["svd", "ncf", "rnn"])

    def test_balanced_merge_keeps_each_concept_in_front(self) -> None:
        if _balanced_term_merge is None:
            self.skipTest("backend runtime dependencies are not installed")

        term_hits = {
            "svd": [_chunk("svd-1"), _chunk("svd-2")],
            "ncf": [_chunk("ncf-1")],
            "rnn": [_chunk("rnn-1")],
        }
        merged = _balanced_term_merge(term_hits, [_chunk("svd-1"), _chunk("other")], target=5)

        self.assertEqual([chunk.chunk_id for chunk in merged[:3]], ["svd-1", "ncf-1", "rnn-1"])
        self.assertEqual(len({chunk.chunk_id for chunk in merged}), len(merged))
        self.assertEqual(merged[1].metadata["matched_query_term"], "ncf")


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(text=chunk_id, source="course", score=1.0, chunk_id=chunk_id, metadata={})


if __name__ == "__main__":
    unittest.main()
