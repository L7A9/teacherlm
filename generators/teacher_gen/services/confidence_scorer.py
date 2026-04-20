from teacherlm_core.confidence.coverage import score_coverage
from teacherlm_core.confidence.groundedness import score_groundedness
from teacherlm_core.schemas.chunk import Chunk


def _overall(groundedness: float, coverage: float) -> float:
    return 0.7 * groundedness + 0.3 * coverage


def _label(overall: float) -> str:
    if overall >= 0.75:
        return "high"
    if overall >= 0.45:
        return "medium"
    return "low"


async def compute(
    response: str,
    chunks: list[Chunk],
    query: str,
) -> dict:
    groundedness = await score_groundedness(response, chunks)
    coverage = score_coverage(query, chunks)
    overall = _overall(groundedness, coverage)
    return {
        "groundedness": round(groundedness, 3),
        "coverage": round(coverage, 3),
        "overall": round(overall, 3),
        "label": _label(overall),
        "chunks_used": len(chunks),
    }
