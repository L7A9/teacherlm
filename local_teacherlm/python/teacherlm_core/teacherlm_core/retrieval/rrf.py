from __future__ import annotations

from teacherlm_core.schemas.chunk import Chunk

RRF_K = 60


def rrf_fuse(rankings: list[list[Chunk]], top_k: int = 16) -> list[Chunk]:
    """Reciprocal Rank Fusion: score(d) = sum(1 / (RRF_K + rank_i))."""

    fused_scores: dict[str, float] = {}
    chunk_by_id: dict[str, Chunk] = {}
    via_by_id: dict[str, set[str]] = {}

    for ranking in rankings:
        for index, chunk in enumerate(ranking):
            if not chunk.chunk_id:
                continue
            fused_scores[chunk.chunk_id] = fused_scores.get(chunk.chunk_id, 0.0) + (
                1.0 / (RRF_K + index + 1)
            )
            chunk_by_id.setdefault(chunk.chunk_id, chunk)
            via = str(chunk.metadata.get("retrieval_via", "") or "").strip()
            if via:
                via_by_id.setdefault(chunk.chunk_id, set()).add(via)

    ordered = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    fused: list[Chunk] = []
    for chunk_id, score in ordered:
        original = chunk_by_id[chunk_id]
        metadata = dict(original.metadata)
        metadata["retrieval_via"] = sorted(via_by_id.get(chunk_id, set())) or ["rrf"]
        metadata["retrieval_score_type"] = "rrf"
        fused.append(
            Chunk(
                text=original.text,
                source=original.source,
                score=score,
                chunk_id=chunk_id,
                metadata=metadata,
            )
        )
    return fused

