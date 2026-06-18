from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "teacherlm_core"))

from teacherlm_core.retrieval import rrf_fuse, should_skip_hyde
from teacherlm_core.schemas import Chunk


def test_rrf_fuses_rankings_by_chunk_id() -> None:
    a = Chunk(text="alpha", source="a.md", score=1.0, chunk_id="a")
    b = Chunk(text="beta", source="b.md", score=1.0, chunk_id="b")
    fused = rrf_fuse([[a, b], [b]], top_k=2)
    assert [chunk.chunk_id for chunk in fused] == ["b", "a"]
    assert fused[0].metadata["retrieval_score_type"] == "rrf"


def test_hyde_skips_formula_queries() -> None:
    assert should_skip_hyde("solve f(x)=x^2 + 1")
    assert not should_skip_hyde("why is retrieval augmented generation useful")

