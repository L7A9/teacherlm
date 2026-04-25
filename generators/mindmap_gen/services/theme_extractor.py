from teacherlm_core.schemas.chunk import Chunk

from .llm_service import get_llm_service

# Rough cap to keep theme-extraction prompt under ~8k tokens
# (~4 chars per token heuristic).
_MAX_CHARS = 32_000


def _combine_chunks(chunks: list[Chunk], max_chars: int = _MAX_CHARS) -> str:
    """Concatenate chunk text with source attribution, capped at max_chars."""
    parts: list[str] = []
    used = 0
    for ch in chunks:
        block = f"[{ch.source}] {ch.text}".strip()
        if used + len(block) + 2 > max_chars:
            break
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


async def extract(chunks: list[Chunk], n_branches: int) -> list[str]:
    """Extract n_branches main themes from the provided chunks."""
    text = _combine_chunks(chunks)
    result = await get_llm_service().extract_themes(text, n_branches)
    return result.themes
