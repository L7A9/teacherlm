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
    from config import Settings  # noqa: E402
    from services.course_context_service import _searchable_chunks  # noqa: E402
    from services.retrieval_orchestrator import RetrievalOrchestrator  # noqa: E402
except ModuleNotFoundError:
    Settings = None  # type: ignore[assignment]
    _searchable_chunks = None  # type: ignore[assignment]
    RetrievalOrchestrator = None  # type: ignore[assignment]


class FakeContext:
    def __init__(
        self,
        *,
        overview_empty: bool = False,
        graph_chunks: list[Chunk] | None = None,
        relevant_empty: bool = False,
    ) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.overview_empty = overview_empty
        self.graph_chunks = graph_chunks or []
        self.relevant_empty = relevant_empty

    async def get_generator_context(self, **kwargs):
        self.calls.append(("generator", kwargs["output_type"]))
        return [Chunk(text="outline", source="course", score=1, chunk_id="outline", metadata={})]

    async def get_full_course_outline(self, conversation_id):
        self.calls.append(("outline", None))
        if self.overview_empty:
            return []
        return [
            Chunk(
                text="Course outline",
                source="course",
                score=1,
                chunk_id="outline",
                metadata={"context_type": "course_outline"},
            )
        ]

    async def get_mindmap_course_context(self, conversation_id):
        self.calls.append(("mindmap", None))
        if self.overview_empty:
            return []
        return [
            Chunk(
                text="Module 1: Foundations\nMajor headings:\n- Basics",
                source="course",
                score=1,
                chunk_id="mindmap",
                metadata={"context_type": "mindmap_module_pack"},
            )
        ]

    async def get_representative_course_context(self, conversation_id):
        self.calls.append(("representative", None))
        if self.overview_empty:
            return []
        return [
            Chunk(
                text="Representative section",
                source="course.pdf",
                score=1,
                chunk_id="representative",
                metadata={"context_type": "representative_section"},
            )
        ]

    async def get_relevant_chunks(self, conversation_id, query, mode):
        self.calls.append(("relevant", mode))
        if self.relevant_empty:
            return []
        return [
            Chunk(
                text="hit",
                source="course.pdf",
                score=1,
                chunk_id="c1",
                metadata={"section_id": "00000000-0000-0000-0000-000000000001"},
            )
        ]

    async def get_graph_relevant_chunks(self, conversation_id, query, *, limit):
        return self.graph_chunks[:limit]

    async def get_course_sections(self, conversation_id):
        return [
            Chunk(
                text="section",
                source="course.pdf",
                score=1,
                chunk_id="s1",
                metadata={"section_id": "00000000-0000-0000-0000-000000000001"},
            )
        ]

    async def get_equations(self, conversation_id):
        self.calls.append(("equations", None))
        return []

    async def get_tables(self, conversation_id):
        self.calls.append(("tables", None))
        return []


class CourseContextPolicyTests(unittest.IsolatedAsyncioTestCase):
    def test_searchable_chunks_filters_dates_page_numbers_and_tiny_fragments(self) -> None:
        if _searchable_chunks is None:
            self.skipTest("backend runtime dependencies are not installed")
        chunks = [
            Chunk(text="2022/2023", source="course.pdf", score=0, chunk_id="date", metadata={"token_count": 1}),
            Chunk(text="22", source="course.pdf", score=0, chunk_id="page", metadata={"token_count": 1}),
            Chunk(
                text="Android Intents let activities request actions from other Android components.",
                source="course.pdf",
                score=0,
                chunk_id="intent",
                metadata={"token_count": 10},
            ),
        ]

        self.assertEqual([chunk.chunk_id for chunk in _searchable_chunks(chunks)], ["intent"])

    async def test_quiz_without_topic_uses_broad_generator_context(self) -> None:
        if Settings is None or RetrievalOrchestrator is None:
            self.skipTest("backend runtime dependencies are not installed")
        context = FakeContext()
        orchestrator = RetrievalOrchestrator(
            settings=Settings(
                retrieval_rerank_enabled=False,
                retrieval_context_expansion_enabled=False,
            ),
            context_service=context,  # type: ignore[arg-type]
        )

        chunks = await orchestrator.retrieve_for(
            output_type="quiz",
            query="",
            conversation_id="00000000-0000-0000-0000-000000000000",
            topic=None,
        )

        self.assertEqual(chunks[0].chunk_id, "outline")
        self.assertEqual(
            context.calls,
            [("generator", "quiz")],
        )

    async def test_podcast_without_topic_uses_broad_generator_context(self) -> None:
        if Settings is None or RetrievalOrchestrator is None:
            self.skipTest("backend runtime dependencies are not installed")
        context = FakeContext()
        orchestrator = RetrievalOrchestrator(
            settings=Settings(
                retrieval_rerank_enabled=False,
                retrieval_context_expansion_enabled=False,
            ),
            context_service=context,  # type: ignore[arg-type]
        )

        chunks = await orchestrator.retrieve_for(
            output_type="podcast",
            query="",
            conversation_id="00000000-0000-0000-0000-000000000000",
            topic=None,
        )

        self.assertEqual(chunks[0].chunk_id, "outline")
        self.assertEqual(context.calls, [("generator", "podcast")])

    async def test_topic_quiz_uses_retrieval_plus_section_context(self) -> None:
        if Settings is None or RetrievalOrchestrator is None:
            self.skipTest("backend runtime dependencies are not installed")
        context = FakeContext()
        orchestrator = RetrievalOrchestrator(
            settings=Settings(
                retrieval_rerank_enabled=False,
                retrieval_context_expansion_enabled=False,
            ),
            context_service=context,  # type: ignore[arg-type]
        )

        chunks = await orchestrator.retrieve_for(
            output_type="quiz",
            query="SVD",
            conversation_id="00000000-0000-0000-0000-000000000000",
            topic="SVD",
        )

        self.assertEqual([call[0] for call in context.calls], ["relevant"])
        self.assertIn("topic-section", chunks[0].chunk_id)
        self.assertEqual(chunks[-1].chunk_id, "c1")

    async def test_topic_podcast_falls_back_to_generator_context_when_retrieval_is_empty(self) -> None:
        if Settings is None or RetrievalOrchestrator is None:
            self.skipTest("backend runtime dependencies are not installed")
        context = FakeContext(relevant_empty=True)
        orchestrator = RetrievalOrchestrator(
            settings=Settings(
                retrieval_rerank_enabled=False,
                retrieval_context_expansion_enabled=False,
            ),
            context_service=context,  # type: ignore[arg-type]
        )

        chunks = await orchestrator.retrieve_for(
            output_type="podcast",
            query="missing topic",
            conversation_id="00000000-0000-0000-0000-000000000000",
            topic="missing topic",
        )

        self.assertEqual(chunks[0].chunk_id, "outline")
        self.assertEqual([call[0] for call in context.calls], ["relevant", "generator"])

    async def test_vague_text_course_request_uses_course_overview_context(self) -> None:
        if Settings is None or RetrievalOrchestrator is None:
            self.skipTest("backend runtime dependencies are not installed")
        context = FakeContext()
        orchestrator = RetrievalOrchestrator(
            settings=Settings(
                retrieval_rerank_enabled=False,
                retrieval_context_expansion_enabled=False,
            ),
            context_service=context,  # type: ignore[arg-type]
        )

        chunks = await orchestrator.retrieve_for(
            output_type="text",
            query="explain this course to me",
            conversation_id="00000000-0000-0000-0000-000000000000",
        )

        self.assertEqual([chunk.chunk_id for chunk in chunks], ["mindmap", "outline", "representative"])
        self.assertEqual([call[0] for call in context.calls], ["mindmap", "outline", "representative"])

    async def test_student_style_course_about_question_uses_course_overview_context(self) -> None:
        if Settings is None or RetrievalOrchestrator is None:
            self.skipTest("backend runtime dependencies are not installed")
        context = FakeContext()
        orchestrator = RetrievalOrchestrator(
            settings=Settings(
                retrieval_rerank_enabled=False,
                retrieval_context_expansion_enabled=False,
            ),
            context_service=context,  # type: ignore[arg-type]
        )

        chunks = await orchestrator.retrieve_for(
            output_type="text",
            query="what this course about?",
            conversation_id="00000000-0000-0000-0000-000000000000",
        )

        self.assertEqual([chunk.chunk_id for chunk in chunks], ["mindmap", "outline", "representative"])
        self.assertEqual([call[0] for call in context.calls], ["mindmap", "outline", "representative"])

    async def test_specific_text_question_still_uses_semantic_retrieval(self) -> None:
        if Settings is None or RetrievalOrchestrator is None:
            self.skipTest("backend runtime dependencies are not installed")
        context = FakeContext()
        orchestrator = RetrievalOrchestrator(
            settings=Settings(
                retrieval_rerank_enabled=False,
                retrieval_context_expansion_enabled=False,
            ),
            context_service=context,  # type: ignore[arg-type]
        )

        chunks = await orchestrator.retrieve_for(
            output_type="text",
            query="what is the difference between SVD and NCF?",
            conversation_id="00000000-0000-0000-0000-000000000000",
        )

        self.assertEqual(chunks[0].chunk_id, "c1")
        self.assertEqual(context.calls, [("relevant", "semantic_topk")])

    async def test_graph_candidates_are_merged_before_reranking(self) -> None:
        if Settings is None or RetrievalOrchestrator is None:
            self.skipTest("backend runtime dependencies are not installed")

        class RecordingReranker:
            def __init__(self) -> None:
                self.seen_ids: list[str] = []

            async def rerank(self, query, chunks, top_k):  # noqa: ANN001, ANN202
                self.seen_ids = [chunk.chunk_id for chunk in chunks]
                return chunks[:top_k]

        context = FakeContext(
            graph_chunks=[
                Chunk(
                    text="Graph-connected intent evidence",
                    source="course.pdf",
                    score=0.9,
                    chunk_id="graph-c1",
                    metadata={"retrieval_via": "knowledge_graph"},
                )
            ]
        )
        orchestrator = RetrievalOrchestrator(
            settings=Settings(retrieval_context_expansion_enabled=False),
            context_service=context,  # type: ignore[arg-type]
        )
        reranker = RecordingReranker()
        orchestrator._reranker = reranker  # type: ignore[assignment]

        chunks = await orchestrator.retrieve_for(
            output_type="text",
            query="what is an Intent Android?",
            conversation_id="00000000-0000-0000-0000-000000000000",
        )

        self.assertEqual(reranker.seen_ids[:2], ["graph-c1", "c1"])
        self.assertEqual(chunks[0].chunk_id, "graph-c1")

    async def test_vague_text_course_request_falls_back_to_broad_chunks_without_sections(self) -> None:
        if Settings is None or RetrievalOrchestrator is None:
            self.skipTest("backend runtime dependencies are not installed")
        context = FakeContext(overview_empty=True)
        orchestrator = RetrievalOrchestrator(
            settings=Settings(
                retrieval_rerank_enabled=False,
                retrieval_context_expansion_enabled=False,
            ),
            context_service=context,  # type: ignore[arg-type]
        )

        chunks = await orchestrator.retrieve_for(
            output_type="text",
            query="summarize this course",
            conversation_id="00000000-0000-0000-0000-000000000000",
        )

        self.assertEqual(chunks[0].chunk_id, "c1")
        self.assertEqual(
            context.calls,
            [("mindmap", None), ("outline", None), ("representative", None), ("relevant", "coverage_broad")],
        )


if __name__ == "__main__":
    unittest.main()
