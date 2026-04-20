import re

from teacherlm_core.retrieval.bm25 import tokenize
from teacherlm_core.schemas.chunk import Chunk

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")

_STOPWORDS: frozenset[str] = frozenset(
    """
    a an the and or but if then else when while of to in on at by for with
    from as is are was were be been being do does did have has had i you he
    she it we they them me my your his her our their this that these those
    not no yes so too very can could should would may might will shall just
    than into over under about above below out up down off again further
    here there where which who whom what why how all any both each few more
    most other some such own same also only s t d ll ve re m
    """.split()
)


async def score_groundedness(response: str, chunks: list[Chunk]) -> float:
    """Mean per-sentence overlap between response content terms and chunk terms.

    Returns 0.0 when there are no content-bearing sentences or no chunks.
    A score near 1.0 means nearly every non-stopword term in the response
    also appears in the retrieved chunks.
    """
    if not response.strip() or not chunks:
        return 0.0

    chunk_vocab: set[str] = set()
    for c in chunks:
        chunk_vocab.update(tokenize(c.text))
    chunk_vocab -= _STOPWORDS
    if not chunk_vocab:
        return 0.0

    sentences = _split_sentences(response)
    sentence_scores: list[float] = []
    for sent in sentences:
        content_tokens = [t for t in tokenize(sent) if t not in _STOPWORDS]
        if not content_tokens:
            continue
        hits = sum(1 for t in content_tokens if t in chunk_vocab)
        sentence_scores.append(hits / len(content_tokens))

    if not sentence_scores:
        return 0.0
    return sum(sentence_scores) / len(sentence_scores)


def _split_sentences(text: str) -> list[str]:
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return [p.strip() for p in parts if p.strip()]
