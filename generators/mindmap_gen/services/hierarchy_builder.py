import asyncio
import re

from teacherlm_core.schemas.chunk import Chunk

from ..schemas import MindMap, MindMapNode
from .llm_service import get_llm_service

_MAX_CHARS_PER_THEME = 18_000
_MAX_CHUNKS_PER_THEME = 14

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> set[str]:
    return {t.lower() for t in _TOKEN_RE.findall(text) if len(t) > 2}


def _filter_chunks_for_theme(
    theme: str,
    chunks: list[Chunk],
    top_k: int = _MAX_CHUNKS_PER_THEME,
) -> list[Chunk]:
    """Rank chunks by token-overlap with the theme label, return top-k.

    Cheap keyword match — avoids loading a reranker model just to bucket
    already-retrieved chunks under each branch.
    """
    theme_tokens = _tokenize(theme)
    if not theme_tokens:
        return chunks[:top_k]

    scored: list[tuple[float, int, Chunk]] = []
    for idx, ch in enumerate(chunks):
        chunk_tokens = _tokenize(ch.text)
        if not chunk_tokens:
            continue
        overlap = len(theme_tokens & chunk_tokens)
        # Combine overlap with the chunk's own retrieval score so that
        # well-retrieved-but-low-overlap chunks still surface.
        score = overlap + 0.1 * ch.score
        scored.append((score, idx, ch))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    if not scored or scored[0][0] <= 0:
        return _broad_sample(chunks, target=top_k)

    selected: list[Chunk] = []
    seen: set[str] = set()
    for _, idx, chunk in scored[: max(4, top_k // 2)]:
        for candidate in _with_neighbors(chunks, idx):
            if candidate.chunk_id in seen:
                continue
            seen.add(candidate.chunk_id)
            selected.append(candidate)
            if len(selected) >= top_k:
                return selected

    return selected or _broad_sample(chunks, target=top_k)


def _with_neighbors(chunks: list[Chunk], idx: int) -> list[Chunk]:
    current = chunks[idx]
    out: list[Chunk] = []
    current_file = current.metadata.get("file_id")
    for candidate_idx in (idx - 1, idx, idx + 1):
        if candidate_idx < 0 or candidate_idx >= len(chunks):
            continue
        candidate = chunks[candidate_idx]
        if candidate.metadata.get("file_id") == current_file:
            out.append(candidate)
    return out


def _broad_sample(chunks: list[Chunk], target: int) -> list[Chunk]:
    if len(chunks) <= target:
        return list(chunks)
    stride = max(1, len(chunks) // target)
    return chunks[::stride][:target]


def _combine(chunks: list[Chunk], max_chars: int = _MAX_CHARS_PER_THEME) -> str:
    parts: list[str] = []
    used = 0
    for ch in chunks:
        block = f"[{ch.source}] {ch.text}".strip()
        if used + len(block) + 2 > max_chars:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


async def _build_branch(theme: str, chunks: list[Chunk]) -> MindMapNode:
    relevant = _filter_chunks_for_theme(theme, chunks)
    text = _combine(relevant)
    expansion = await get_llm_service().expand_subtopic(theme, text)
    return MindMapNode(text=theme, children=expansion.subtopics)


async def build(
    themes: list[str],
    context_chunks: list[Chunk],
    size_config: dict,
    central_topic: str,
) -> MindMap:
    """Expand each theme into a subtree, in parallel, return full MindMap."""
    branches = await asyncio.gather(
        *(_build_branch(theme, context_chunks) for theme in themes)
    )
    return MindMap(central_topic=central_topic, branches=list(branches))
