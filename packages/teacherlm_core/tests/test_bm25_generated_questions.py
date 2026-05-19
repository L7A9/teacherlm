from __future__ import annotations

import sys
from pathlib import Path


CORE_DIR = Path(__file__).resolve().parents[1]
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

from teacherlm_core.retrieval.bm25 import BM25Index
from teacherlm_core.schemas.chunk import Chunk


def test_bm25_matches_generated_questions_metadata() -> None:
    chunks = [
        Chunk(
            text="Singular value decomposition factorizes the user-item matrix.",
            source="lecture.pdf",
            score=0.0,
            chunk_id="svd",
            metadata={"generated_questions": ["What are the equations for SVD?"]},
        ),
        Chunk(
            text="Recurrent neural networks process ordered sequences.",
            source="lecture.pdf",
            score=0.0,
            chunk_id="rnn",
            metadata={"generated_questions": ["How do RNN hidden states work?"]},
        ),
    ]

    [hit, *_] = BM25Index(chunks).query("explain svd equations", top_k=2)

    assert hit.chunk_id == "svd"
    assert hit.text == chunks[0].text
