from __future__ import annotations

import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = Path(__file__).resolve().parents[3] / "packages" / "teacherlm_core"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from config import Settings  # noqa: E402

try:
    from routers.generate import _effective_topic, _retrieval_query  # noqa: E402
    from services.retrieval_orchestrator import (  # noqa: E402
        _BROAD_NO_TOPIC_OUTPUTS,
        RetrievalOrchestrator,
    )
except ModuleNotFoundError:
    RetrievalOrchestrator = None  # type: ignore[assignment]
    _BROAD_NO_TOPIC_OUTPUTS = set()
    _effective_topic = None  # type: ignore[assignment]
    _retrieval_query = None  # type: ignore[assignment]


class RetrievalOrchestratorConfigTests(unittest.TestCase):
    def test_backend_reranker_is_enabled_for_all_generators_by_default(self) -> None:
        settings = Settings()

        self.assertTrue(settings.retrieval_rerank_enabled)
        self.assertTrue(settings.retrieval_rerank_warmup_enabled)
        self.assertEqual(settings.retrieval_reranker_model, "BAAI/bge-reranker-base")
        self.assertEqual(
            set(settings.retrieval_rerank_modes),
            {
                "semantic_topk",
                "coverage_broad",
                "narrative_arc",
                "topic_clusters",
                "relationship_dense",
            },
        )

    def test_all_generator_outputs_route_through_backend_modes(self) -> None:
        if RetrievalOrchestrator is None:
            self.skipTest("backend runtime dependencies are not installed")
        orchestrator = RetrievalOrchestrator(settings=Settings())

        self.assertEqual(orchestrator.mode_for("text"), "semantic_topk")
        self.assertEqual(orchestrator.mode_for("quiz"), "coverage_broad")
        self.assertEqual(orchestrator.mode_for("mindmap"), "topic_clusters")
        self.assertEqual(orchestrator.mode_for("podcast"), "narrative_arc")

    def test_broad_outputs_use_full_course_context_only_without_topic(self) -> None:
        if _effective_topic is None or _retrieval_query is None:
            self.skipTest("backend runtime dependencies are not installed")

        self.assertEqual(_BROAD_NO_TOPIC_OUTPUTS, {"quiz", "mindmap", "presentation", "podcast"})
        self.assertEqual(_effective_topic("quiz", "SVD"), "SVD")
        self.assertEqual(_effective_topic("mindmap", "SVD"), "SVD")
        self.assertEqual(_retrieval_query("quiz", "SVD"), "SVD")
        self.assertEqual(_retrieval_query("mindmap", "SVD"), "SVD")

    def test_teacher_and_podcast_keep_topic_queries(self) -> None:
        if _effective_topic is None or _retrieval_query is None:
            self.skipTest("backend runtime dependencies are not installed")

        self.assertEqual(_effective_topic("text", "SVD"), "SVD")
        self.assertEqual(_effective_topic("podcast", "SVD"), "SVD")
        self.assertEqual(_retrieval_query("text", "SVD"), "SVD")
        self.assertEqual(_retrieval_query("podcast", "SVD"), "SVD")


if __name__ == "__main__":
    unittest.main()
