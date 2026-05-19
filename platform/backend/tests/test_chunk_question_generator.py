from __future__ import annotations

import asyncio
import uuid
import unittest

from config import Settings
from services.chunk_question_generator import (
    ChunkQuestionBatch,
    ChunkQuestionGenerator,
    ChunkQuestionItem,
    _sanitize_questions,
    searchable_chunk_text,
)
from services.chunking_service import Chunk


class ChunkQuestionGeneratorTests(unittest.TestCase):
    def test_annotate_chunks_stores_sanitized_generated_questions(self) -> None:
        chunk = _chunk("SVD decomposes ratings into latent user and item factors.")
        generator = ChunkQuestionGenerator(
            Settings(chunk_question_generation_enabled=True, chunk_question_count=2)
        )

        async def fake_generate_batch(_chunks):  # noqa: ANN001, ANN202
            return ChunkQuestionBatch(
                chunks=[
                    ChunkQuestionItem(
                        chunk_id=chunk.chunk_id,
                        questions=[
                            "How does SVD represent user preferences",
                            "How does SVD represent user preferences?",
                            "x",
                            "What are latent factors in SVD?",
                        ],
                    )
                ]
            )

        generator._generate_batch = fake_generate_batch  # type: ignore[method-assign]

        [annotated] = asyncio.run(generator.annotate_chunks([chunk]))

        self.assertEqual(
            annotated.metadata["generated_questions"],
            [
                "How does SVD represent user preferences?",
                "What are latent factors in SVD?",
            ],
        )
        self.assertEqual(annotated.metadata["question_generator"], "llm-v1")

    def test_searchable_chunk_text_includes_questions_without_replacing_source(self) -> None:
        chunk = _chunk(
            "The rating matrix is factorized.",
            metadata={
                "heading_path": "Collaborative Filtering > SVD",
                "key_concepts": ["matrix factorization"],
                "generated_questions": ["What equations define SVD?"],
            },
        )

        text = searchable_chunk_text(chunk)

        self.assertIn("The rating matrix is factorized.", text)
        self.assertIn("Collaborative Filtering > SVD", text)
        self.assertIn("What equations define SVD?", text)

    def test_sanitize_questions_deduplicates_and_limits(self) -> None:
        self.assertEqual(
            _sanitize_questions(
                [" Explain NCF ", "Explain NCF?", "tiny", "How does NCF compare to SVD?"],
                limit=2,
            ),
            ["Explain NCF?", "How does NCF compare to SVD?"],
        )


def _chunk(text: str, metadata: dict[str, object] | None = None) -> Chunk:
    return Chunk(
        chunk_id=str(uuid.uuid4()),
        text=text,
        source="lecture.pdf",
        index=0,
        token_count=20,
        metadata=metadata or {},
    )


if __name__ == "__main__":
    unittest.main()

