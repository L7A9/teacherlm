import re

from rank_bm25 import BM25Okapi

from teacherlm_core.schemas.chunk import Chunk

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Thin wrapper around rank_bm25.BM25Okapi over a Chunk corpus."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._tokenized = [tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    def __len__(self) -> int:
        return len(self._chunks)

    def query(self, query: str, top_k: int = 20) -> list[Chunk]:
        """Return the top-k chunks with BM25 scores written into `Chunk.score`."""
        if self._bm25 is None or not self._chunks:
            return []
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            zip(scores, self._chunks, strict=True),
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
