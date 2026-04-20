import asyncio
import re

import numpy as np

from teacherlm_core.retrieval.bm25 import tokenize
from teacherlm_core.retrieval.hybrid_retriever import HybridRetriever
from teacherlm_core.schemas.chunk import Chunk


async def semantic_topk(
    query: str,
    retriever: HybridRetriever,
    k: int = 10,
) -> list[Chunk]:
    """Top-K closest to query (chat mode)."""
    return await retriever.retrieve(query, top_k=k)


async def coverage_broad(
    query: str,
    retriever: HybridRetriever,
    k: int = 20,
    diversity_lambda: float = 0.7,
) -> list[Chunk]:
    """MMR-like: balance query relevance (λ) against diversity (1-λ)."""
    pool = await retriever.retrieve(query, top_k=max(k * 3, 30))
    if not pool:
        return []
    q_tokens = set(tokenize(query))
    candidates: list[tuple[Chunk, set[str]]] = [
        (c, set(tokenize(c.text))) for c in pool
    ]
    selected: list[tuple[Chunk, set[str]]] = []
    while candidates and len(selected) < k:
        best_idx = 0
        best_score = -1e9
        for i, (_c, toks) in enumerate(candidates):
            rel = _jaccard(q_tokens, toks)
            div = max(
                (_jaccard(toks, s_toks) for _, s_toks in selected), default=0.0
            )
            score = diversity_lambda * rel - (1 - diversity_lambda) * div
            if score > best_score:
                best_score = score
                best_idx = i
        selected.append(candidates.pop(best_idx))
    return [c for c, _ in selected]


async def narrative_arc(
    query: str,
    retriever: HybridRetriever,
    all_chunks: list[Chunk],
) -> list[Chunk]:
    """Intro-like + key middle points (query-relevant) + conclusion-like."""
    if not all_chunks:
        return []
    intro = _pick_intro(all_chunks)
    conclusion = _pick_conclusion(all_chunks)
    middle = await retriever.retrieve(query, top_k=6)
    seen: set[str] = set()
    out: list[Chunk] = []
    for c in [*intro, *middle, *conclusion]:
        if c.chunk_id in seen:
            continue
        seen.add(c.chunk_id)
        out.append(c)
    return out


async def topic_clusters(
    query: str,
    retriever: HybridRetriever,
    n_clusters: int = 6,
) -> list[Chunk]:
    """K-means on chunk embeddings; return one representative per cluster."""
    pool = await retriever.retrieve(query, top_k=max(n_clusters * 5, 30))
    if len(pool) <= n_clusters:
        return pool
    embeddings = await asyncio.to_thread(
        lambda: np.array(list(retriever.embedder.embed([c.text for c in pool])))
    )
    labels, centers = _kmeans(embeddings, n_clusters)
    reps: list[Chunk] = []
    for k in range(n_clusters):
        mask = labels == k
        if not mask.any():
            continue
        idx = np.where(mask)[0]
        dists = np.linalg.norm(embeddings[idx] - centers[k], axis=1)
        reps.append(pool[int(idx[int(np.argmin(dists))])])
    return reps


_VERB_RE = re.compile(r"\b\w+(?:ed|ing|es)\b", re.IGNORECASE)
_ENTITY_RE = re.compile(r"\b(?:[A-Z][a-z]+)(?:\s+[A-Z][a-z]+)+\b")


async def relationship_dense(
    query: str,
    retriever: HybridRetriever,
) -> list[Chunk]:
    """Rank chunks by entity + verb density (regex heuristic)."""
    pool = await retriever.retrieve(query, top_k=30)
    scored: list[tuple[float, Chunk]] = []
    for c in pool:
        tokens = tokenize(c.text)
        if not tokens:
            continue
        entity_hits = len(_ENTITY_RE.findall(c.text))
        verb_hits = len(_VERB_RE.findall(c.text))
        density = (entity_hits + verb_hits) / len(tokens)
        scored.append((density, c))
    scored.sort(key=lambda kv: kv[0], reverse=True)
    return [
        Chunk(
            text=c.text,
            source=c.source,
            score=float(density),
            chunk_id=c.chunk_id,
            metadata=c.metadata,
        )
        for density, c in scored[:10]
    ]


# ---------- helpers ----------

def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


_INTRO_KEYWORDS = ("introduction", "overview", "preface", "abstract")
_CONCLUSION_KEYWORDS = ("conclusion", "summary", "in summary", "final thoughts")


def _pick_intro(chunks: list[Chunk], n: int = 2) -> list[Chunk]:
    hits = [c for c in chunks if any(k in c.text.lower() for k in _INTRO_KEYWORDS)]
    if hits:
        return hits[:n]
    cutoff = max(1, len(chunks) // 10)
    return chunks[:cutoff][:n]


def _pick_conclusion(chunks: list[Chunk], n: int = 2) -> list[Chunk]:
    hits = [
        c for c in chunks if any(k in c.text.lower() for k in _CONCLUSION_KEYWORDS)
    ]
    if hits:
        return hits[-n:]
    cutoff = max(1, len(chunks) // 10)
    return chunks[-cutoff:][-n:]


def _kmeans(
    data: np.ndarray,
    n_clusters: int,
    max_iter: int = 20,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = data.shape[0]
    init_idx = rng.choice(n, size=n_clusters, replace=False)
    centers = data[init_idx].copy()
    labels = np.zeros(n, dtype=np.int64)
    for _ in range(max_iter):
        dists = np.linalg.norm(data[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for k in range(n_clusters):
            mask = labels == k
            if mask.any():
                centers[k] = data[mask].mean(axis=0)
    return labels, centers
