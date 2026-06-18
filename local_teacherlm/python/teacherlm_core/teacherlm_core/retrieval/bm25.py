from __future__ import annotations

import math
import re
from collections import Counter

from teacherlm_core.schemas.chunk import Chunk

try:
    from rank_bm25 import BM25Okapi
except Exception:  # noqa: BLE001
    BM25Okapi = None


_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25 over chunks with a deterministic token-overlap fallback."""

    def __init__(self, chunks: list[Chunk]) -> None:
        self._chunks = chunks
        self._tokenized = [tokenize(_searchable_text(chunk)) for chunk in chunks]
        self._bm25 = BM25Okapi(self._tokenized) if BM25Okapi and self._tokenized else None

    def query(self, query: str, top_k: int = 20) -> list[Chunk]:
        if not self._chunks or not query.strip():
            return []
        if self._bm25 is not None:
            scores = self._bm25.get_scores(tokenize(query))
        else:
            scores = [_fallback_score(tokenize(query), tokens) for tokens in self._tokenized]
        ranked = sorted(
            zip(scores, self._chunks, strict=True),
            key=lambda pair: pair[0],
            reverse=True,
        )[:top_k]
        return [
            Chunk(
                text=chunk.text,
                source=chunk.source,
                score=float(score),
                chunk_id=chunk.chunk_id,
                metadata={**chunk.metadata, "retrieval_via": "bm25"},
            )
            for score, chunk in ranked
            if float(score) > 0.0
        ]


def _searchable_text(chunk: Chunk) -> str:
    metadata = chunk.metadata or {}
    parts = [chunk.text, str(metadata.get("section_title", ""))]
    heading_path = metadata.get("heading_path")
    if isinstance(heading_path, list):
        parts.extend(str(item) for item in heading_path)
    elif heading_path:
        parts.append(str(heading_path))
    for key in ("key_concepts", "generated_questions", "formula_labels", "table_captions"):
        value = metadata.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value)
    return "\n".join(part for part in parts if part)


def _fallback_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    if not query_tokens or not doc_tokens:
        return 0.0
    query = Counter(query_tokens)
    doc = Counter(doc_tokens)
    overlap = sum(min(query[token], doc[token]) for token in query)
    return overlap / math.sqrt(max(1, len(doc_tokens)))

