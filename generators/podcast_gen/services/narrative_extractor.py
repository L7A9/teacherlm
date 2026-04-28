from __future__ import annotations

from teacherlm_core.schemas.chunk import Chunk

from ..config import get_settings
from ..schemas import NarrativeArc
from .llm_service import LLMService, build_system_prompt


def _friendly_source(chunk: Chunk) -> str:
    """A label safe to read aloud — just the source filename, no IDs."""
    return chunk.source or "uploaded material"


def format_context_for_speech(chunks: list[Chunk], max_chars: int = 9000) -> str:
    """Render chunks with human-friendly source labels only — no chunk_ids,
    no scores, no machine metadata. The LLM is told to cite by these labels
    (or by the topic the excerpt covers), so anything it leaks into the
    spoken script will still sound natural.
    """
    parts: list[str] = []
    used = 0
    for c in chunks:
        block = (
            f"=== From: {_friendly_source(c)} ===\n"
            f"{c.text.strip()}\n"
        )
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts) if parts else "(no source excerpts available)"


async def extract_narrative_arc(
    chunks: list[Chunk],
    *,
    topic_focus: str,
    language_hint: str,
    llm: LLMService,
) -> NarrativeArc:
    """Extract a teaching-order narrative arc the script can follow."""
    s = get_settings()
    system = build_system_prompt(
        "narrative_arc.txt",
        min_points=s.min_key_points,
        max_points=s.max_key_points,
        topic_focus=topic_focus or "(no specific topic — give a balanced overview)",
        context_block=format_context_for_speech(chunks),
        language_hint=language_hint,
    )
    arc = await llm.extract_structured(
        system=system,
        user_message=(
            "Plan the podcast arc now. Return JSON only, matching the schema."
        ),
        schema=NarrativeArc,
    )
    if len(arc.key_points) > s.max_key_points:
        arc.key_points = arc.key_points[: s.max_key_points]
    if not arc.key_points:
        arc.key_points = [
            (chunks[0].text.strip().split(".")[0][:160] + ".")
            if chunks
            else "Overview of the uploaded material."
        ]
    return arc
