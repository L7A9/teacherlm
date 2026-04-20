from teacherlm_core.confidence.groundedness import _STOPWORDS
from teacherlm_core.retrieval.bm25 import tokenize
from teacherlm_core.schemas.chunk import Chunk


def score_coverage(query: str, chunks: list[Chunk]) -> float:
    """Fraction of distinct query keywords present in the retrieved chunks.

    Returns 0.0 when the query has no content-bearing keywords or the
    chunk set is empty. A score of 1.0 means every query keyword shows up
    at least once across the top chunks.
    """
    query_keywords = {t for t in tokenize(query) if t not in _STOPWORDS}
    if not query_keywords or not chunks:
        return 0.0

    chunk_vocab: set[str] = set()
    for c in chunks:
        chunk_vocab.update(tokenize(c.text))

    hits = len(query_keywords & chunk_vocab)
    return hits / len(query_keywords)
