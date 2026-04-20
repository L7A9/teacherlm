from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field

from config import Settings, get_settings


# Split on sentence-ending punctuation followed by whitespace. Keeps the
# terminating punctuation attached to the preceding sentence.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'\(\[])")

# Paragraph boundary — used as a hard split hint for very long documents.
_PARAGRAPH_SPLIT = re.compile(r"\n{2,}")

# Cheap token estimator. BGE / SentencePiece tokens average ~1.3 per whitespace
# word for English prose; rounding up keeps us safely under the 512-token
# bucket without pulling in a full tokenizer dependency.
def _approx_tokens(text: str) -> int:
    if not text:
        return 0
    words = text.split()
    return max(1, int(len(words) * 1.3))


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    text: str
    source: str
    index: int
    token_count: int
    metadata: dict[str, str] = field(default_factory=dict)


class ChunkingService:
    """Semantic chunking: paragraph → sentence split → merge to target size with overlap."""

    def __init__(self, settings: Settings | None = None) -> None:
        s = settings or get_settings()
        self._max_tokens = s.chunk_max_tokens
        self._overlap_tokens = s.chunk_overlap_tokens

    def chunk_text(self, text: str, *, source: str) -> list[Chunk]:
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        if not sentences:
            return []

        chunks: list[Chunk] = []
        buffer: list[str] = []
        buffer_tokens = 0
        index = 0

        for sentence in sentences:
            sent_tokens = _approx_tokens(sentence)

            # A single sentence larger than the bucket — emit it alone rather
            # than losing it.
            if sent_tokens >= self._max_tokens:
                if buffer:
                    chunks.append(self._finalize(buffer, source, index, buffer_tokens))
                    index += 1
                chunks.append(self._finalize([sentence], source, index, sent_tokens))
                index += 1
                buffer, buffer_tokens = self._carry_overlap([sentence])
                continue

            if buffer_tokens + sent_tokens > self._max_tokens and buffer:
                chunks.append(self._finalize(buffer, source, index, buffer_tokens))
                index += 1
                buffer, buffer_tokens = self._carry_overlap(buffer)

            buffer.append(sentence)
            buffer_tokens += sent_tokens

        if buffer:
            chunks.append(self._finalize(buffer, source, index, buffer_tokens))

        return chunks

    # --- internals ---

    def _split_sentences(self, text: str) -> list[str]:
        sentences: list[str] = []
        for paragraph in _PARAGRAPH_SPLIT.split(text):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            parts = _SENTENCE_SPLIT.split(paragraph)
            sentences.extend(part.strip() for part in parts if part.strip())
        return sentences

    def _finalize(self, pieces: list[str], source: str, index: int, tokens: int) -> Chunk:
        return Chunk(
            chunk_id=str(uuid.uuid4()),
            text=" ".join(pieces).strip(),
            source=source,
            index=index,
            token_count=tokens,
            metadata={"chunker": "semantic-sentence-v1"},
        )

    def _carry_overlap(self, prev: list[str]) -> tuple[list[str], int]:
        """Return the tail of `prev` whose token count is ≈ overlap_tokens."""
        if self._overlap_tokens <= 0 or not prev:
            return [], 0

        carry: list[str] = []
        total = 0
        for sentence in reversed(prev):
            sent_tokens = _approx_tokens(sentence)
            if total + sent_tokens > self._overlap_tokens and carry:
                break
            carry.insert(0, sentence)
            total += sent_tokens
            if total >= self._overlap_tokens:
                break
        return carry, total


_chunker: ChunkingService | None = None


def get_chunker() -> ChunkingService:
    global _chunker
    if _chunker is None:
        _chunker = ChunkingService()
    return _chunker
