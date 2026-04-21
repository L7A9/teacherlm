from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from ..config import get_settings
from ..schemas import BasicCard, Card, ClozeCard


@lru_cache
def _embedder() -> TextEmbedding:
    return TextEmbedding(model_name=get_settings().embedding_model)


def _card_text(card: Card) -> str:
    """Canonical string used for similarity. Combines prompt + answer so that
    two cards aren't mistaken as duplicates just because they share a concept."""
    if isinstance(card, BasicCard):
        return f"{card.front}\n{card.back}"
    if isinstance(card, ClozeCard):
        return card.text
    return str(card)


def _embed(texts: list[str]) -> np.ndarray:
    vectors = list(_embedder().embed(texts))
    matrix = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def dedupe_cards(cards: list[Card], *, threshold: float | None = None) -> list[Card]:
    """Greedy dedup by cosine similarity on fastembed vectors.

    Walks cards in order; keeps a card only if its similarity to every already-
    kept card is below `threshold`. Threshold defaults to settings.dedupe_similarity.
    """
    if not cards:
        return []
    settings = get_settings()
    thresh = settings.dedupe_similarity if threshold is None else threshold

    texts = [_card_text(c) for c in cards]
    vecs = _embed(texts)

    kept_idx: list[int] = []
    for i in range(len(cards)):
        if not kept_idx:
            kept_idx.append(i)
            continue
        sims = vecs[i] @ vecs[kept_idx].T
        if float(np.max(sims)) < thresh:
            kept_idx.append(i)
    return [cards[i] for i in kept_idx]
