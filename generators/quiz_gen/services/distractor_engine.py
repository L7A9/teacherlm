from __future__ import annotations

import asyncio
import re
from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

from teacherlm_core.schemas.chunk import Chunk

from ..config import get_settings
from ..schemas import MCQ


_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-']{2,}")


@lru_cache
def _embedder() -> TextEmbedding:
    return TextEmbedding(model_name=get_settings().embedding_model)


def _embed(texts: list[str]) -> np.ndarray:
    vectors = list(_embedder().embed(texts))
    return np.asarray(vectors, dtype=np.float32)


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / norms


def _candidate_phrases(chunks: list[Chunk], chunk_id: str, pool_size: int) -> list[str]:
    """Pull short phrase candidates from sibling chunks (and the source chunk).

    We prefer multi-word noun phrases by joining adjacent capitalized/lowercase
    tokens, plus single salient tokens. This keeps things dependency-light.
    """
    seen: set[str] = set()
    out: list[str] = []

    def _push(text: str) -> None:
        cleaned = text.strip().strip(".,;:").strip()
        if not cleaned or len(cleaned) > 60:
            return
        key = cleaned.lower()
        if key in seen:
            return
        seen.add(key)
        out.append(cleaned)

    # Same-source chunk first (closest semantic neighborhood).
    ordered = sorted(chunks, key=lambda c: 0 if c.chunk_id == chunk_id else 1)
    for chunk in ordered:
        tokens = _TOKEN_RE.findall(chunk.text)
        # Bigrams
        for i in range(len(tokens) - 1):
            _push(f"{tokens[i]} {tokens[i + 1]}")
            if len(out) >= pool_size * 4:
                break
        # Unigrams
        for tok in tokens:
            _push(tok)
            if len(out) >= pool_size * 4:
                break
        if len(out) >= pool_size * 4:
            break

    return out[: pool_size * 4]


def _select_distractors(
    correct: str,
    candidates: list[str],
    *,
    n: int,
    sim_min: float,
    sim_max: float,
) -> list[str]:
    """Pick n candidates whose cosine similarity to `correct` is in [sim_min, sim_max].

    Falls back to the closest-but-not-identical candidates if the band is empty.
    """
    correct_lower = correct.strip().lower()
    pool = [c for c in candidates if c.strip().lower() != correct_lower]
    if not pool:
        return []

    matrix = _normalize(_embed([correct, *pool]))
    correct_vec = matrix[0]
    pool_vecs = matrix[1:]
    sims = pool_vecs @ correct_vec  # cosine because vectors are L2-normalized

    in_band_idx = np.where((sims >= sim_min) & (sims <= sim_max))[0]
    if len(in_band_idx) >= n:
        # Highest similarity within the band → hardest negatives first.
        ranked = in_band_idx[np.argsort(-sims[in_band_idx])]
        chosen = ranked[:n]
    else:
        # Not enough in-band — top up by sliding outward (closer to correct first).
        ranked_all = np.argsort(-sims)
        chosen = []
        for idx in ranked_all:
            if sims[idx] >= 0.999:  # near-duplicate of the correct answer
                continue
            chosen.append(idx)
            if len(chosen) >= n:
                break

    # De-duplicate by surface form just in case.
    seen: set[str] = set()
    out: list[str] = []
    for idx in chosen:
        text = pool[idx]
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


async def enhance_mcq_distractors(
    mcq: MCQ,
    chunks: list[Chunk],
) -> MCQ:
    """Replace MCQ options with [correct, *3 hard negatives], shuffled deterministically."""
    settings = get_settings()
    if not (0 <= mcq.correct_index < len(mcq.options)):
        return mcq

    correct = mcq.options[mcq.correct_index]
    candidates = _candidate_phrases(
        chunks=chunks,
        chunk_id=mcq.source_chunk_id,
        pool_size=settings.distractor_pool_size,
    )
    if not candidates:
        return mcq

    distractors = await asyncio.to_thread(
        _select_distractors,
        correct,
        candidates,
        n=settings.distractors_per_mcq,
        sim_min=settings.distractor_sim_min,
        sim_max=settings.distractor_sim_max,
    )
    if len(distractors) < settings.distractors_per_mcq:
        # Not enough usable distractors — keep original options rather than degrade.
        return mcq

    options = [correct, *distractors]
    # Stable shuffle keyed on the question so reruns produce the same order.
    order = sorted(range(len(options)), key=lambda i: hash((mcq.question, i)))
    shuffled = [options[i] for i in order]
    correct_index = shuffled.index(correct)

    return mcq.model_copy(update={"options": shuffled, "correct_index": correct_index})
