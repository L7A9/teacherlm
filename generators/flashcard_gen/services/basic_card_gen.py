from __future__ import annotations

import logging
import re

from teacherlm_core.schemas.chunk import Chunk

from ..schemas import BasicCard, BasicCardBatch, PrioritizedConcept
from .llm_service import LLMService, build_system_prompt


logger = logging.getLogger(__name__)


# Strip markup that the LLM would otherwise copy verbatim onto the card back.
# Tables, code fences, and display-math blocks don't belong on flashcards; we
# remove them from the context the LLM sees so it grounds on prose only.
_HTML_BLOCK_RE = re.compile(
    r"<\s*(table|ul|ol|pre|code|img)[^>]*>.*?<\s*/\s*\1\s*>",
    re.IGNORECASE | re.DOTALL,
)
_DISPLAY_MATH_RE = re.compile(r"\$\$.*?\$\$", re.DOTALL)
_MARKDOWN_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_MULTI_BLANK_RE = re.compile(r"\n{3,}")


def _sanitize_chunk_text(text: str) -> str:
    text = _HTML_BLOCK_RE.sub("[table omitted]", text)
    text = _DISPLAY_MATH_RE.sub("[equation omitted]", text)
    text = _MARKDOWN_TABLE_ROW_RE.sub("", text)
    return _MULTI_BLANK_RE.sub("\n\n", text).strip()


def _concepts_block(prioritized: list[PrioritizedConcept]) -> str:
    lines: list[str] = []
    for idx, p in enumerate(prioritized, 1):
        definition = p.concept.definition or "(no inline definition; use the chunks)"
        lines.append(
            f"{idx}. {p.concept.name}\n"
            f"   context: {p.concept.context_sentence}\n"
            f"   hint: {definition}"
        )
    return "\n".join(lines)


def _chunks_block(chunks: list[Chunk], wanted_ids: set[str]) -> str:
    keep = [c for c in chunks if c.chunk_id in wanted_ids] or chunks
    return "\n\n".join(
        f"[chunk_id={c.chunk_id} source={c.source}]\n{_sanitize_chunk_text(c.text)}"
        for c in keep
    )


async def generate_basic_cards(
    prioritized: list[PrioritizedConcept],
    chunks: list[Chunk],
    llm: LLMService,
) -> list[BasicCard]:
    """Turn prioritized concepts into BasicCards via ollama format=BasicCardBatch.

    We ask for the whole batch in one call so the model can vary its phrasing
    across cards. The LLM-provided front/back are paired back to the input
    concepts by index; any shortfall is dropped silently.
    """
    if not prioritized:
        return []

    wanted_ids = {p.concept.source_chunk_id for p in prioritized}
    system = build_system_prompt(
        "card_generation.txt",
        concepts_block=_concepts_block(prioritized),
        chunks_block=_chunks_block(chunks, wanted_ids),
    )
    user = f"Write {len(prioritized)} flashcards — one per listed concept, in order."

    try:
        batch = await llm.generate_structured(
            system=system,
            user_message=user,
            schema=BasicCardBatch,
        )
    except Exception:
        logger.exception("basic card generation failed for %d concepts", len(prioritized))
        return []

    out: list[BasicCard] = []
    for draft, prio in zip(batch.cards, prioritized):
        front = draft.front.strip()
        back = draft.back.strip()
        if not front or not back:
            continue
        if _contains_markup(front) or _contains_markup(back):
            logger.debug("dropping card for %r: markup leaked through", prio.concept.name)
            continue
        out.append(
            BasicCard(
                front=front,
                back=back,
                concept=prio.concept.name,
                source_chunk_id=prio.concept.source_chunk_id,
            )
        )
    return out


_MARKUP_LEAK_RE = re.compile(r"<\s*/?\s*(table|tr|td|th|tbody|thead|pre|code|img)\b", re.IGNORECASE)


def _contains_markup(text: str) -> bool:
    if _MARKUP_LEAK_RE.search(text):
        return True
    if _DISPLAY_MATH_RE.search(text):
        return True
    if _MARKDOWN_TABLE_ROW_RE.search(text):
        return True
    return False
