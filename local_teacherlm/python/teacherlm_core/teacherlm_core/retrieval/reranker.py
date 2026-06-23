from __future__ import annotations

import asyncio

from teacherlm_core.schemas.chunk import Chunk


class CrossEncoderReranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-base",
        *,
        cache_dir: str | None = None,
    ) -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        try:
            self._encoder = TextCrossEncoder(model_name=model_name, cache_dir=cache_dir)
        except TypeError as exc:
            if "cache_dir" not in str(exc):
                raise
            # Keep compatibility with small test doubles and older fastembed builds.
            self._encoder = TextCrossEncoder(model_name=model_name)

    async def rerank(self, query: str, chunks: list[Chunk], top_k: int = 16) -> list[Chunk]:
        if not query.strip() or not chunks:
            return chunks[:top_k]
        scores = await asyncio.to_thread(self._score, query, [chunk.text for chunk in chunks])
        ranked = sorted(zip(scores, chunks, strict=True), key=lambda pair: pair[0], reverse=True)[
            :top_k
        ]
        return [
            Chunk(
                text=chunk.text,
                source=chunk.source,
                score=float(score),
                chunk_id=chunk.chunk_id,
                metadata={**chunk.metadata, "retrieval_score_type": "reranker"},
            )
            for score, chunk in ranked
        ]

    def _score(self, query: str, documents: list[str]) -> list[float]:
        return [float(score) for score in self._encoder.rerank(query, documents)]

