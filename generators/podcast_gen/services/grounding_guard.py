from __future__ import annotations

import re
from typing import Any

from ..schemas import PodcastScript, Segment


_NO_MATERIALS_RE = re.compile(
    r"\bno\s+materials?\s+available\b|"
    r"\bno\s+(?:uploaded\s+)?(?:course\s+)?materials?\b|"
    r"\bno\s+source\s+excerpts?\b|"
    r"\bno\s+context\s+chunks?\b",
    re.IGNORECASE,
)


def usable_context_chunks(chunks: list[Any]) -> list[Any]:
    return [
        chunk
        for chunk in chunks
        if len(" ".join(str(getattr(chunk, "text", "") or "").split())) >= 24
    ]


def script_claims_no_materials(script: PodcastScript) -> bool:
    text = " ".join(
        [
            script.title,
            script.summary,
            *(segment.text for segment in script.segments[:8]),
        ]
    )
    return bool(_NO_MATERIALS_RE.search(text))


def deterministic_script_from_arc(arc: Any) -> PodcastScript:
    segments = [
        Segment(
            speaker="host_a",
            text=f"Welcome. Today we are going to unpack {arc.title} using the uploaded course material.",
        ),
        Segment(
            speaker="host_b",
            text=arc.intro or "We will move through the main ideas in a clear study order.",
        ),
    ]
    for point in arc.key_points:
        segments.append(
            Segment(
                speaker="host_a",
                text=f"What should I understand about {point}?",
            )
        )
        segments.append(
            Segment(
                speaker="host_b",
                text=(
                    f"The course material highlights this point: {point}. "
                    "The useful study move is to connect the definition, the example, and the practical step."
                ),
            )
        )
    segments.extend(
        [
            Segment(
                speaker="host_a",
                text="So the main path is to follow the course structure and keep linking each idea to its examples.",
            ),
            Segment(
                speaker="host_b",
                text=arc.conclusion or "Exactly. That gives you a grounded overview to review before going deeper.",
            ),
        ]
    )
    return PodcastScript(title=arc.title, summary=arc.intro or arc.title, segments=segments)
