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
    from services.retrieval_orchestrator import RetrievalOrchestrator  # noqa: E402
except ModuleNotFoundError:
    Settings = None  # type: ignore[assignment]
    RetrievalOrchestrator = None  # type: ignore[assignment]


class FakeContext:
    def __init__(self, *, overview_empty: bool = False) -> None:
        self.calls: list[tuple[str, str | None]] = []
        self.overview_empty = overview_empty

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
        return [
            Chunk(
                text="hit",
                source="course.pdf",
                score=1,
                chunk_id="c1",
                metadata={"section_id": "00000000-0000-0000-0000-000000000001"},
            )
        ]

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
        return []

    async def get_tables(self, conversation_id):
        return []


class CourseContextPolicyTests(unittest.IsolatedAsyncioTestCase):
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
        self.assertEqual(context.calls, [("generator", "quiz")])

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

        self.assertEqual([chunk.chunk_id for chunk in chunks], ["outline", "representative"])
        self.assertEqual([call[0] for call in context.calls], ["outline", "representative"])

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

        self.assertEqual([chunk.chunk_id for chunk in chunks], ["outline", "representative"])
        self.assertEqual([call[0] for call in context.calls], ["outline", "representative"])

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
            [("outline", None), ("representative", None), ("relevant", "coverage_broad")],
        )


if __name__ == "__main__":
    unittest.main()
