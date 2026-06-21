from __future__ import annotations

import asyncio
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "teacherlm_core"))
sys.path.insert(0, str(ROOT / "local_api"))


def _paragraph(label: str, terms: str) -> str:
    return " ".join([f"{label} {terms} explains a coherent mechanism and its supported consequences."] * 4)


def test_hybrid_chunker_uses_embedding_valley_for_topic_shift(monkeypatch) -> None:
    import local_api.services.ingestion as ingestion

    text = "\n\n".join(
        [
            _paragraph("Vectors", "basis dimension linear algebra"),
            _paragraph("Matrices", "basis transformation linear algebra"),
            _paragraph("Reactions", "atoms bonds chemical stoichiometry"),
            _paragraph("Molecules", "atoms bonds chemical compounds"),
        ]
    )
    calls: list[list[str]] = []

    class FakeVectors:
        async def embed_passages(self, passages):
            calls.append(list(passages))
            return [[1.0, 0.0], [0.99, 0.01], [0.0, 1.0], [0.01, 0.99]]

    monkeypatch.setattr(ingestion, "get_vector_service", lambda: FakeVectors())
    drafts = asyncio.run(
        ingestion._hybrid_chunk_text(
            text,
            max_chars=1100,
            overlap_chars=100,
            min_semantic_chars=300,
            similarity_threshold=0.72,
        )
    )

    assert len(calls) == 1
    assert len(drafts) == 2
    assert drafts[0]["boundary_type"] == "semantic"
    assert "Matrices" in drafts[0]["text"]
    assert "Reactions" not in drafts[0]["text"]
    assert "Reactions" in drafts[1]["text"]


def test_structured_units_stay_attached_during_semantic_splitting() -> None:
    import local_api.services.ingestion as ingestion

    text = "\n\n".join(
        [
            _paragraph("Force", "mass acceleration Newtonian mechanics"),
            "$$F = ma$$",
            _paragraph("Motion", "mass acceleration Newtonian mechanics"),
            _paragraph("Cells", "membrane nucleus biological tissue"),
        ]
    )
    units = ingestion._semantic_units(text, max_chars=700)
    drafts = ingestion._assemble_semantic_chunks(
        units,
        [0.05, 0.05, 0.10],
        max_chars=900,
        overlap_chars=80,
        min_semantic_chars=250,
        similarity_threshold=0.72,
    )

    formula_chunk = next(draft["text"] for draft in drafts if "F = ma" in draft["text"])
    assert "Force" in formula_chunk
    assert "Motion" in formula_chunk
    assert "Cells" not in formula_chunk


def test_oversized_prose_is_split_without_truncating_the_tail() -> None:
    import local_api.services.ingestion as ingestion

    text = " ".join(f"token{index}" for index in range(500)) + " FINAL_SENTINEL"
    units = ingestion._semantic_units(text, max_chars=320)

    assert len(units) > 1
    assert all(len(unit["text"]) <= 320 for unit in units)
    reconstructed = " ".join(unit["text"] for unit in units)
    assert re.findall(r"token\d+", reconstructed) == [f"token{index}" for index in range(500)]
    assert reconstructed.endswith("FINAL_SENTINEL")


def test_short_coherent_section_skips_semantic_embedding(monkeypatch) -> None:
    import local_api.services.ingestion as ingestion

    class UnexpectedVectors:
        async def embed_passages(self, _passages):
            raise AssertionError("short sections should not invoke semantic boundary embeddings")

    monkeypatch.setattr(ingestion, "get_vector_service", lambda: UnexpectedVectors())
    drafts = asyncio.run(
        ingestion._hybrid_chunk_text(
            "A short coherent section explains one idea without changing topics.",
            max_chars=1800,
            overlap_chars=180,
            min_semantic_chars=600,
            similarity_threshold=0.72,
        )
    )

    assert drafts == [
        {
            "text": "A short coherent section explains one idea without changing topics.",
            "boundary_type": "section_end",
            "boundary_similarity": None,
        }
    ]
