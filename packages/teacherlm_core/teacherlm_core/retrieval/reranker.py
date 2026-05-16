import asyncio

from teacherlm_core.schemas.chunk import Chunk


class CrossEncoderReranker:
    """Fastembed cross-encoder reranker for refining a retrieved candidate set."""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base") -> None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.model_name = model_name
        self._encoder = TextCrossEncoder(model_name=model_name)

    async def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int = 5,
    ) -> list[Chunk]:
        if not chunks:
            return []
        scores = await asyncio.to_thread(self._score, query, [c.text for c in chunks])
        ranked = sorted(
            zip(scores, chunks, strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )[:top_k]
        return [
            Chunk(
                text=c.text,
                source=c.source,
                score=float(score),
                chunk_id=c.chunk_id,
                metadata=c.metadata,
            )
            for score, c in ranked
        ]

    def _score(self, query: str, documents: list[str]) -> list[float]:
        return [float(s) for s in self._encoder.rerank(query, documents)]
