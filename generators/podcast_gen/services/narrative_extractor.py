from __future__ import annotations

import re

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
    for c in _usable_chunks(chunks):
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
    if _needs_grounded_fallback(arc, chunks):
        arc = _fallback_arc_from_sources(chunks, language_hint=language_hint)
    if len(arc.key_points) > s.max_key_points:
        arc.key_points = arc.key_points[: s.max_key_points]
    if not arc.key_points:
        arc = _fallback_arc_from_sources(chunks, language_hint=language_hint)
    return arc


_NO_MATERIALS_RE = re.compile(
    r"\b(no|not enough|missing|without|zero)\s+"
    r"(?:uploaded\s+)?(?:course\s+)?(?:source\s+)?materials?\b|"
    r"\bno\s+source\s+excerpts?\b|"
    r"\bno\s+context\s+chunks?\b|"
    r"\bno\s+materials?\s+available\b",
    re.IGNORECASE,
)


def _usable_chunks(chunks: list[Chunk]) -> list[Chunk]:
    return [chunk for chunk in chunks if _is_usable_source_text(chunk.text)]


def _is_usable_source_text(text: str | None) -> bool:
    cleaned = " ".join(str(text or "").split())
    if len(cleaned) < 24:
        return False
    if cleaned == "(no source excerpts available)":
        return False
    alpha = sum(1 for char in cleaned if char.isalpha())
    return alpha >= 10


def _looks_like_no_materials_arc(arc: NarrativeArc) -> bool:
    text = " ".join([arc.title, arc.intro, *arc.key_points, arc.conclusion])
    return bool(_NO_MATERIALS_RE.search(text))


_PROMPT_LEAK_TITLE_RE = re.compile(
    r"\b("
    r"content\s+planning|podcast\s+planning|planning\s+a\s+podcast|"
    r"educational\s+podcast|narrative\s+arc|source\s+excerpts?|"
    r"planification\s+de\s+contenu|planification\s+du\s+podcast|"
    r"arc\s+narratif|extraits?\s+sources?"
    r")\b",
    re.IGNORECASE,
)


def _needs_grounded_fallback(arc: NarrativeArc, chunks: list[Chunk]) -> bool:
    usable = _usable_chunks(chunks)
    if not usable:
        return False
    if _looks_like_no_materials_arc(arc):
        return True
    title = " ".join(str(arc.title or "").split())
    if _PROMPT_LEAK_TITLE_RE.search(title):
        return True
    return False


def _fallback_arc_from_sources(chunks: list[Chunk], *, language_hint: str) -> NarrativeArc:
    usable = _usable_chunks(chunks)
    if not usable:
        return NarrativeArc(
            title="Course Podcast",
            intro="This episode needs uploaded course material before it can be grounded.",
            key_points=["No source excerpts were available for the podcast."],
            conclusion="Upload course files, then generate the podcast again.",
            sources=[],
        )

    title = _fallback_title(usable)
    points = _fallback_key_points(usable, limit=get_settings().max_key_points)
    sources = _dedupe([_friendly_source(chunk) for chunk in usable if _friendly_source(chunk)])
    if "fran" in language_hint.casefold():
        return NarrativeArc(
            title=f"Podcast du cours: {title}",
            intro=f"Dans cet episode, nous allons parcourir les idees principales du cours autour de {title}.",
            key_points=points,
            conclusion="Le fil conducteur est de relier les notions du support aux exemples et aux etapes pratiques.",
            sources=sources,
        )
    return NarrativeArc(
        title=f"Course Podcast: {title}",
        intro=f"In this episode, we will walk through the main course ideas around {title}.",
        key_points=points,
        conclusion="The big takeaway is to connect the source concepts to their examples and practical steps.",
        sources=sources,
    )


def _fallback_title(chunks: list[Chunk]) -> str:
    for chunk in chunks:
        heading = str((chunk.metadata or {}).get("heading_path") or "").strip()
        if heading:
            leaf = heading.split(">")[-1].strip()
            if leaf and len(leaf) > 3:
                return leaf[:90]
    for chunk in chunks:
        first = _first_sentence(chunk.text)
        if first:
            return first[:90]
    return "uploaded course material"


def _fallback_key_points(chunks: list[Chunk], *, limit: int) -> list[str]:
    points: list[str] = []
    for chunk in chunks:
        heading = str((chunk.metadata or {}).get("heading_path") or "").strip()
        sentence = _first_sentence(chunk.text)
        if heading and sentence:
            point = f"{heading.split('>')[-1].strip()}: {sentence}"
        else:
            point = sentence or heading
        point = " ".join(point.split())[:180].strip(" :-")
        if point:
            points.append(point)
        if len(_dedupe(points)) >= limit:
            break
    return _dedupe(points)[:limit] or ["Overview of the uploaded course material."]


def _first_sentence(text: str | None) -> str:
    cleaned = " ".join(str(text or "").split())
    if not cleaned:
        return ""
    match = re.search(r"(.{40,220}?[.!?])(?:\s|$)", cleaned)
    if match:
        return match.group(1).strip()
    return cleaned[:180].strip()


def _dedupe(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        key = value.casefold().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(value)
    return out
