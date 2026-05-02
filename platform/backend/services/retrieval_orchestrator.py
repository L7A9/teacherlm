from __future__ import annotations

import uuid
from typing import Literal

from teacherlm_core.retrieval.hybrid_retriever import HybridRetriever
from teacherlm_core.retrieval.retrieval_modes import (
    coverage_broad,
    narrative_arc,
    relationship_dense,
    semantic_topk,
    topic_clusters,
)
from teacherlm_core.schemas.chunk import Chunk

from config import Settings, get_settings
from services.vector_service import VectorService, _collection_name, get_vector_service


OutputType = Literal[
    "text",
    "quiz",
    "report",
    "presentation",
    "chart",
    "podcast",
    "mindmap",
]
RetrievalMode = Literal[
    "semantic_topk",
    "coverage_broad",
    "narrative_arc",
    "topic_clusters",
    "relationship_dense",
]


# Mapping from backend CLAUDE.md.
_OUTPUT_TO_MODE: dict[str, RetrievalMode] = {
    "text": "semantic_topk",
    "chat": "semantic_topk",
    "quiz": "coverage_broad",
    "report": "topic_clusters",
    "presentation": "topic_clusters",
    "podcast": "narrative_arc",
    "chart": "relationship_dense",
    "mindmap": "topic_clusters",
}


class RetrievalOrchestrator:
    """Picks a retrieval mode from the requested output type and runs it.

    Builds a HybridRetriever on top of the VectorService's AsyncQdrantClient
    and cached fastembed TextEmbedding. BM25 is primed from the per-conversation
    chunk corpus scrolled out of Qdrant.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        vector_service: VectorService | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._vectors = vector_service or get_vector_service()

    def mode_for(self, output_type: str) -> RetrievalMode:
        return _OUTPUT_TO_MODE.get(output_type, "semantic_topk")

    async def retrieve_for(
        self,
        *,
        output_type: str,
        query: str,
        conversation_id: uuid.UUID | str,
    ) -> list[Chunk]:
        mode = self.mode_for(output_type)
        return await self.retrieve(mode=mode, query=query, conversation_id=conversation_id)

    async def retrieve(
        self,
        *,
        mode: RetrievalMode,
        query: str,
        conversation_id: uuid.UUID | str,
    ) -> list[Chunk]:
        collection = _collection_name(conversation_id)
        if not await self._vectors._client.collection_exists(collection):
            return []

        embedder = await self._vectors._get_embedder()
        retriever = HybridRetriever(
            qdrant_client=self._vectors._client,
            collection_name=collection,
            embedder=embedder,
        )

        all_chunks = await self._load_corpus(conversation_id)
        if all_chunks:
            retriever.index_bm25(all_chunks)

        k = self._settings.retrieval_top_k

        # No-topic path: callers (quiz, mind map, etc.) pass query=""
        # when the user didn't narrow to a topic. Query-based retrieval
        # would return an empty pool, so feed the generator a uniform
        # stride sample of the whole corpus instead.
        if not query.strip():
            if mode == "topic_clusters":
                target = max(k * 12, 96)
            elif mode == "coverage_broad":
                target = max(k * 8, 64)
            else:
                target = max(k * 4, 24)
            return self._broad_sample(all_chunks, target=target)

        match mode:
            case "semantic_topk":
                return await semantic_topk(query, retriever, k=k)
            case "coverage_broad":
                return await coverage_broad(query, retriever, k=max(k * 2, 12))
            case "narrative_arc":
                return await narrative_arc(query, retriever, all_chunks)
            case "topic_clusters":
                return await topic_clusters(query, retriever, n_clusters=6)
            case "relationship_dense":
                return await relationship_dense(query, retriever)

    @staticmethod
    def _broad_sample(chunks: list[Chunk], *, target: int) -> list[Chunk]:
        if len(chunks) <= target:
            return list(chunks)
        stride = max(1, len(chunks) // target)
        return chunks[::stride][:target]

    async def _load_corpus(self, conversation_id: uuid.UUID | str) -> list[Chunk]:
        scored = await self._vectors.scroll_all(conversation_id, limit=2000)
        chunks = [
            Chunk(
                text=s.text,
                source=s.source,
                score=s.score,
                chunk_id=s.chunk_id,
                metadata=dict(s.metadata),
            )
            for s in scored
        ]
        return sorted(
            chunks,
            key=lambda c: (
                str(c.metadata.get("file_id", "")),
                c.source,
                int(c.metadata.get("index", 0) or 0),
                c.chunk_id,
            ),
        )


_orchestrator: RetrievalOrchestrator | None = None


def get_retrieval_orchestrator() -> RetrievalOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RetrievalOrchestrator()
    return _orchestrator
