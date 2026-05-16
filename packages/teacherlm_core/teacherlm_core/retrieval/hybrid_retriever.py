from typing import Any

from teacherlm_core.retrieval.bm25 import BM25Index
from teacherlm_core.schemas.chunk import Chunk

RRF_K = 60


class HybridRetriever:
    """BM25 + dense semantic retrieval fused via Reciprocal Rank Fusion.

    `qdrant_client` must expose an async `query_points` method (AsyncQdrantClient).
    `embedder` must expose `.embed(texts) -> Iterable[Sequence[float]]`
    (fastembed TextEmbedding).
    """

    def __init__(
        self,
        qdrant_client: Any,
        collection_name: str,
        embedder: Any,
        *,
        dense_top_k: int = 20,
        sparse_top_k: int = 20,
    ) -> None:
        self.qdrant = qdrant_client
        self.collection_name = collection_name
        self.embedder = embedder
        self.dense_top_k = dense_top_k
        self.sparse_top_k = sparse_top_k
        self._bm25: BM25Index | None = None

    def index_bm25(self, chunks: list[Chunk]) -> None:
        """Prime the BM25 index from a known corpus (preferred over lazy build)."""
        self._bm25 = BM25Index(chunks)

    async def retrieve(self, query: str, top_k: int = 20) -> list[Chunk]:
        dense_hits = await self._dense_search(
            query,
            limit=max(top_k, self.dense_top_k),
        )
        sparse_hits = self._sparse_search(query, limit=max(top_k, self.sparse_top_k))
        return _rrf_fuse([dense_hits, sparse_hits], top_k=top_k)

    async def _dense_search(self, query: str, limit: int) -> list[Chunk]:
        embed = getattr(self.embedder, "query_embed", self.embedder.embed)
        query_vec = list(next(iter(embed([query]))))
        result = await self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=limit,
            with_payload=True,
        )
        points = getattr(result, "points", result)
        return [_point_to_chunk(p) for p in points]

    def _sparse_search(self, query: str, limit: int) -> list[Chunk]:
        if self._bm25 is None:
            return []
        return self._bm25.query(query, top_k=limit)


def _point_to_chunk(point: Any) -> Chunk:
    payload = getattr(point, "payload", None) or {}
    return Chunk(
        text=payload.get("text", ""),
        source=payload.get("source", ""),
        score=float(getattr(point, "score", 0.0)),
        chunk_id=str(getattr(point, "id", payload.get("chunk_id", ""))),
        metadata={k: v for k, v in payload.items() if k not in {"text", "source"}},
    )


def _rrf_fuse(rankings: list[list[Chunk]], top_k: int) -> list[Chunk]:
    """Reciprocal Rank Fusion: score(d) = Σ 1 / (RRF_K + rank_i(d))."""
    fused_scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking):
            fused_scores[chunk.chunk_id] = (
                fused_scores.get(chunk.chunk_id, 0.0) + 1.0 / (RRF_K + rank + 1)
            )
            chunk_by_id.setdefault(chunk.chunk_id, chunk)
    ordered = sorted(fused_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        Chunk(
            text=chunk_by_id[cid].text,
            source=chunk_by_id[cid].source,
            score=score,
            chunk_id=cid,
            metadata=chunk_by_id[cid].metadata,
        )
        for cid, score in ordered
    ]
