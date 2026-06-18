from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any

from teacherlm_core.schemas.chunk import Chunk

from local_api.db import get_store


class LocalVectorService:
    """Device-local fastembed vector index backed by SQLite chunk rows."""

    def __init__(self) -> None:
        self._embedder: Any | None = None
        self._embedder_model = ""
        self._lock = asyncio.Lock()

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embedder = await self._get_embedder()
        batch_size = max(1, _runtime_settings().embedding_batch_size)
        vectors: list[list[float]] = []
        for batch in _batched(list(texts), batch_size):
            def _run(batch_texts: list[str] = batch) -> list[list[float]]:
                embed = getattr(embedder, "passage_embed", embedder.embed)
                return [list(vec) for vec in embed(batch_texts)]

            vectors.extend(await asyncio.to_thread(_run))
        return vectors

    async def embed_queries(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        embedder = await self._get_embedder()

        def _run() -> list[list[float]]:
            embed = getattr(embedder, "query_embed", embedder.embed)
            return [list(vec) for vec in embed(list(texts))]

        return await asyncio.to_thread(_run)

    async def embed_chunks(self, chunks: list[dict[str, Any]]) -> dict[str, list[float]]:
        texts = [_searchable_chunk_text(_dict_chunk(chunk)) for chunk in chunks]
        vectors = await self.embed_passages(texts)
        return {chunk["id"]: vectors[index] for index, chunk in enumerate(chunks)}

    async def rebuild_conversation(self, conversation_id: str) -> dict[str, Any]:
        rows = get_store().list_chunks(conversation_id)
        embeddings = await self.embed_chunks(rows)
        settings = _runtime_settings()
        get_store().update_chunk_embeddings(
            embeddings,
            model=settings.embedding_model,
            dim=settings.embedding_dim,
        )
        return self.index_status(conversation_id)

    async def search(
        self,
        conversation_id: str,
        query: str,
        *,
        top_k: int,
        source_file_ids: list[str] | None = None,
    ) -> list[Chunk]:
        if not query.strip():
            return []
        rows = get_store().list_chunks(conversation_id, source_file_ids=source_file_ids)
        searchable = [row for row in rows if _valid_embedding(row)]
        if not searchable:
            return []
        [query_vector] = await self.embed_queries([query])
        scored: list[tuple[float, dict[str, Any]]] = []
        for row in searchable:
            score = _cosine(query_vector, row.get("embedding") or [])
            if score <= 0:
                continue
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            Chunk(
                text=row["text"],
                source=row["source_filename"],
                score=float(score),
                chunk_id=row["id"],
                metadata={
                    **row.get("metadata", {}),
                    "retrieval_via": "dense_vector",
                    "retrieval_score_type": "cosine",
                },
            )
            for score, row in scored[:top_k]
        ]

    def index_status(self, conversation_id: str | None = None) -> dict[str, Any]:
        settings = _runtime_settings()
        params: list[Any] = []
        where = ""
        if conversation_id:
            where = "WHERE conversation_id = ?"
            params.append(conversation_id)
        rows = get_store().query(
            f"SELECT embedding_json, metadata_json FROM search_chunks {where}",
            params,
        )
        chunk_count = len(rows)
        embedded = 0
        stale = 0
        for row in rows:
            metadata = _json(row.get("metadata_json"), {})
            vector = _json(row.get("embedding_json"), None)
            if isinstance(vector, list) and vector:
                embedded += 1
            if not _metadata_matches(metadata, vector):
                stale += 1
        graph_where = "WHERE conversation_id = ?" if conversation_id else ""
        graph_params = [conversation_id] if conversation_id else []
        node_count = get_store().one(
            f"SELECT COUNT(*) AS count FROM knowledge_graph_nodes {graph_where}",
            graph_params,
        ) or {"count": 0}
        edge_count = get_store().one(
            f"SELECT COUNT(*) AS count FROM knowledge_graph_edges {graph_where}",
            graph_params,
        ) or {"count": 0}
        return {
            "embedding_model": settings.embedding_model,
            "embedding_dim": settings.embedding_dim,
            "embedding_batch_size": settings.embedding_batch_size,
            "chunk_count": chunk_count,
            "embedded_chunk_count": embedded,
            "stale_chunk_count": stale,
            "graph_node_count": int(node_count["count"]),
            "graph_edge_count": int(edge_count["count"]),
            "ready": chunk_count > 0 and stale == 0,
        }

    async def _get_embedder(self) -> Any:
        settings = _runtime_settings()
        if self._embedder is not None and self._embedder_model == settings.embedding_model:
            return self._embedder
        async with self._lock:
            if self._embedder is None or self._embedder_model != settings.embedding_model:
                from fastembed import TextEmbedding

                self._embedder = await asyncio.to_thread(TextEmbedding, model_name=settings.embedding_model)
                self._embedder_model = settings.embedding_model
        return self._embedder


def _dict_chunk(row: dict[str, Any]) -> Chunk:
    return Chunk(
        text=row["text"],
        source=row["source_filename"],
        score=0.0,
        chunk_id=row["id"],
        metadata=row.get("metadata", {}),
    )


def _searchable_chunk_text(chunk: Chunk) -> str:
    metadata = chunk.metadata or {}
    parts = [chunk.text, str(metadata.get("section_title", ""))]
    heading = metadata.get("heading_path")
    if isinstance(heading, list):
        parts.extend(str(item) for item in heading)
    elif heading:
        parts.append(str(heading))
    for key in ("key_concepts", "generated_questions", "formula_labels", "table_captions"):
        value = metadata.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
        elif value:
            parts.append(str(value))
    return "\n".join(part for part in parts if str(part).strip())


def _valid_embedding(row: dict[str, Any]) -> bool:
    return _metadata_matches(row.get("metadata", {}), row.get("embedding"))


def _metadata_matches(metadata: dict[str, Any], vector: object) -> bool:
    settings = _runtime_settings()
    if not isinstance(vector, list) or not vector:
        return False
    return (
        str(metadata.get("embedding_model") or "") == settings.embedding_model
        and int(metadata.get("embedding_dim") or 0) == settings.embedding_dim
        and len(vector) == settings.embedding_dim
    )


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(float(x) * float(x) for x in a))
    norm_b = math.sqrt(sum(float(y) * float(y) for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def _batched[T](items: Sequence[T], batch_size: int) -> list[list[T]]:
    return [list(items[index : index + batch_size]) for index in range(0, len(items), batch_size)]


def _runtime_settings() -> Any:
    from local_api.services.settings import get_settings_service

    return get_settings_service().effective_retrieval_settings()


def _json(value: object, default: Any) -> Any:
    if isinstance(value, str):
        try:
            return __import__("json").loads(value)
        except ValueError:
            return default
    return value if value is not None else default


_vector_service: LocalVectorService | None = None


def get_vector_service() -> LocalVectorService:
    global _vector_service
    if _vector_service is None:
        _vector_service = LocalVectorService()
    return _vector_service
