from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from teacherlm_core.retrieval.hybrid_retriever import HybridRetriever  # noqa: E402
from teacherlm_core.schemas.chunk import Chunk  # noqa: E402


class _FakeEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class _FakeQdrant:
    def __init__(self) -> None:
        self.limits: list[int] = []

    async def query_points(self, **kwargs: object) -> SimpleNamespace:
        limit = int(kwargs["limit"])
        self.limits.append(limit)
        points = [
            SimpleNamespace(
                id=f"dense-{idx}",
                score=1.0 / (idx + 1),
                payload={"text": f"Dense chunk {idx}", "source": "course.md"},
            )
            for idx in range(limit)
        ]
        return SimpleNamespace(points=points)


class HybridRetrieverTests(unittest.TestCase):
    def test_candidate_pool_settings_are_used_before_rrf_trim(self) -> None:
        async def run() -> None:
            qdrant = _FakeQdrant()
            retriever = HybridRetriever(
                qdrant_client=qdrant,
                collection_name="conv_test",
                embedder=_FakeEmbedder(),
                dense_top_k=12,
                sparse_top_k=9,
            )
            retriever.index_bm25(
                [
                    Chunk(
                        text=f"collaborative filtering example {idx}",
                        source="course.md",
                        score=0.0,
                        chunk_id=f"sparse-{idx}",
                    )
                    for idx in range(20)
                ]
            )

            hits = await retriever.retrieve("collaborative filtering", top_k=3)
            self.assertEqual(qdrant.limits, [12])
            self.assertEqual(len(hits), 3)

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
