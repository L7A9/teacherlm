from __future__ import annotations

import asyncio
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from fastembed import TextEmbedding
from qdrant_client import AsyncQdrantClient
from qdrant_client.http import models as qm

from config import Settings, get_settings
from services.chunking_service import Chunk


@dataclass(slots=True)
class ScoredChunk:
    chunk_id: str
    text: str
    source: str
    score: float
    metadata: dict[str, object]


def _collection_name(conversation_id: uuid.UUID | str) -> str:
    return f"conv_{conversation_id}"


class VectorService:
    """Async vector store over Qdrant + fastembed BGE embeddings."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._client = AsyncQdrantClient(
            url=self._settings.qdrant_url,
            api_key=self._settings.qdrant_api_key,
        )
        self._embed_model_name = self._settings.embedding_model
        self._dim = self._settings.embedding_dim
        self._embedder: TextEmbedding | None = None
        self._embedder_lock = asyncio.Lock()

    # --- embeddings ---

    async def _get_embedder(self) -> TextEmbedding:
        if self._embedder is not None:
            return self._embedder
        async with self._embedder_lock:
            if self._embedder is None:
                self._embedder = await asyncio.to_thread(TextEmbedding, self._embed_model_name)
        return self._embedder

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embedder = await self._get_embedder()

        def _run() -> list[list[float]]:
            return [list(vec) for vec in embedder.embed(list(texts))]

        return await asyncio.to_thread(_run)

    # --- collection lifecycle ---

    async def ensure_collection(self, conversation_id: uuid.UUID | str) -> None:
        name = _collection_name(conversation_id)
        existing = await self._client.collection_exists(name)
        if existing:
            return
        await self._client.create_collection(
            collection_name=name,
            vectors_config=qm.VectorParams(size=self._dim, distance=qm.Distance.COSINE),
        )
        # Index payload fields used for filtering.
        for field_name, schema in (
            ("source", qm.PayloadSchemaType.KEYWORD),
            ("file_id", qm.PayloadSchemaType.KEYWORD),
        ):
            await self._client.create_payload_index(
                collection_name=name,
                field_name=field_name,
                field_schema=schema,
            )

    async def delete_collection(self, conversation_id: uuid.UUID | str) -> None:
        name = _collection_name(conversation_id)
        if await self._client.collection_exists(name):
            await self._client.delete_collection(name)

    async def delete_by_file(self, conversation_id: uuid.UUID | str, file_id: str) -> None:
        name = _collection_name(conversation_id)
        if not await self._client.collection_exists(name):
            return
        await self._client.delete(
            collection_name=name,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="file_id", match=qm.MatchValue(value=file_id))],
                )
            ),
        )

    # --- writes ---

    async def upsert_chunks(
        self,
        conversation_id: uuid.UUID | str,
        chunks: Sequence[Chunk],
        *,
        file_id: str,
    ) -> int:
        if not chunks:
            return 0
        await self.ensure_collection(conversation_id)
        vectors = await self.embed([c.text for c in chunks])
        points = [
            qm.PointStruct(
                id=c.chunk_id,
                vector=vectors[i],
                payload={
                    "text": c.text,
                    "source": c.source,
                    "file_id": file_id,
                    "index": c.index,
                    "token_count": c.token_count,
                    **c.metadata,
                },
            )
            for i, c in enumerate(chunks)
        ]
        await self._client.upsert(
            collection_name=_collection_name(conversation_id),
            points=points,
            wait=True,
        )
        return len(points)

    # --- reads ---

    async def search(
        self,
        conversation_id: uuid.UUID | str,
        query: str,
        *,
        top_k: int | None = None,
        file_ids: Iterable[str] | None = None,
    ) -> list[ScoredChunk]:
        name = _collection_name(conversation_id)
        if not await self._client.collection_exists(name):
            return []

        limit = top_k or self._settings.retrieval_top_k
        [vector] = await self.embed([query])

        flt: qm.Filter | None = None
        if file_ids:
            flt = qm.Filter(
                must=[qm.FieldCondition(key="file_id", match=qm.MatchAny(any=list(file_ids)))]
            )

        result = await self._client.query_points(
            collection_name=name,
            query=vector,
            limit=limit,
            query_filter=flt,
            with_payload=True,
        )
        return [self._to_scored(p) for p in result.points]

    async def scroll_all(
        self,
        conversation_id: uuid.UUID | str,
        *,
        limit: int = 500,
    ) -> list[ScoredChunk]:
        """Return all chunks (unranked) — feeds coverage_broad / narrative_arc modes."""
        name = _collection_name(conversation_id)
        if not await self._client.collection_exists(name):
            return []

        collected: list[ScoredChunk] = []
        offset: qm.PointId | None = None
        while len(collected) < limit:
            batch, offset = await self._client.scroll(
                collection_name=name,
                limit=min(256, limit - len(collected)),
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            collected.extend(self._to_scored(p, default_score=0.0) for p in batch)
            if offset is None:
                break
        return collected

    # --- helpers ---

    @staticmethod
    def _to_scored(point: object, *, default_score: float = 0.0) -> ScoredChunk:
        payload = getattr(point, "payload", None) or {}
        return ScoredChunk(
            chunk_id=str(getattr(point, "id", "")),
            text=str(payload.get("text", "")),
            source=str(payload.get("source", "")),
            score=float(getattr(point, "score", default_score) or default_score),
            metadata={k: v for k, v in payload.items() if k not in {"text", "source"}},
        )

    async def aclose(self) -> None:
        await self._client.close()


_vector_service: VectorService | None = None


def get_vector_service() -> VectorService:
    global _vector_service
    if _vector_service is None:
        _vector_service = VectorService()
    return _vector_service
