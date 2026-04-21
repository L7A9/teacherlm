from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator

from teacherlm_core.schemas.generator_io import (
    GeneratorInput,
    GeneratorOutput,
    LearnerUpdates,
)

from .config import get_settings
from .schemas import Card, FlashcardDeck
from .services.artifact_store import get_artifact_store
from .services.basic_card_gen import generate_basic_cards
from .services.cloze_card_gen import generate_cloze_cards
from .services.concept_miner import mine_concepts
from .services.deduplicator import dedupe_cards
from .services.exporter import export_deck
from .services.llm_service import get_llm_service
from .services.priority_selector import prioritize
from .services.sm2_scheduler import schedule_cards


logger = logging.getLogger(__name__)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _resolve_card_count(options: dict) -> int:
    s = get_settings()
    raw = (
        options.get("card_count")
        or options.get("count")
        or options.get("n_cards")
        or s.default_card_count
    )
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = s.default_card_count
    return max(s.min_card_count, min(n, s.max_card_count))


def _split_counts(total: int, basic_ratio: float) -> tuple[int, int]:
    n_basic = round(total * basic_ratio)
    n_basic = max(0, min(n_basic, total))
    n_cloze = total - n_basic
    return n_basic, n_cloze


def _resolve_title(options: dict, learner_concepts: list[str]) -> str:
    if topic := options.get("topic"):
        return f"Flashcards: {topic}"
    if learner_concepts:
        return f"Flashcards: {learner_concepts[0]}"
    return "Flashcards"


def _intro(
    total: int,
    struggling: list[str],
    stats: dict,
) -> str:
    if total == 0:
        return (
            "I couldn't pull enough concept-worthy material from your sources — "
            "try uploading more, or chat with me about the topic first."
        )
    if struggling:
        focus = ", ".join(struggling[:3])
        return (
            f"Here's a {total}-card deck focused on areas we've been working on "
            f"({focus}). Review a few at a time — spaced repetition beats cramming."
        )
    return (
        f"Here's a {total}-card deck covering the key concepts from your sources. "
        f"I grouped {stats.get('basic', 0)} Q&A-style cards with "
        f"{stats.get('cloze', 0)} fill-in-the-blank cards."
    )


async def run(inp: GeneratorInput) -> AsyncIterator[str]:
    settings = get_settings()
    llm = get_llm_service()
    options = dict(inp.options or {})

    n_target = _resolve_card_count(options)
    n_basic, n_cloze = _split_counts(n_target, settings.basic_ratio)
    learner = inp.learner_state

    yield _sse(
        "progress",
        {"stage": "mining_concepts", "chunks": len(inp.context_chunks)},
    )
    mined = mine_concepts(inp.context_chunks)
    yield _sse("progress", {"stage": "mined", "count": len(mined)})

    prioritized = prioritize(mined, learner, limit=n_target)
    yield _sse(
        "progress",
        {
            "stage": "prioritized",
            "kept": len(prioritized),
            "struggling": sum(1 for p in prioritized if p.source == "struggling"),
        },
    )

    if not prioritized:
        empty_response = _intro(0, learner.struggling_concepts, {})
        yield _sse("token", {"delta": empty_response})
        yield _sse(
            "done",
            GeneratorOutput(
                response=empty_response,
                generator_id=settings.generator_id,
                output_type=settings.output_type,
                sources=inp.context_chunks,
                metadata={"reason": "no_concepts_mined"},
            ).model_dump(),
        )
        return

    basic_slice = prioritized[:n_basic]
    cloze_slice = prioritized[n_basic : n_basic + n_cloze] or prioritized[:n_cloze]

    yield _sse("progress", {"stage": "generating_basic", "target": len(basic_slice)})
    basic_cards = await generate_basic_cards(basic_slice, inp.context_chunks, llm)

    yield _sse("progress", {"stage": "generating_cloze", "target": len(cloze_slice)})
    cloze_cards = generate_cloze_cards(cloze_slice)

    all_cards: list[Card] = [*basic_cards, *cloze_cards]
    yield _sse("progress", {"stage": "deduplicating", "before": len(all_cards)})
    deduped = dedupe_cards(all_cards, threshold=settings.dedupe_similarity)
    yield _sse("progress", {"stage": "deduplicated", "after": len(deduped)})

    scheduled = schedule_cards(deduped)

    stats = {
        "total": len(scheduled),
        "basic": sum(1 for c in scheduled if c.type == "basic"),
        "cloze": sum(1 for c in scheduled if c.type == "cloze"),
        "dropped_duplicates": len(all_cards) - len(deduped),
        "concepts_mined": len(mined),
    }

    deck = FlashcardDeck(
        title=_resolve_title(options, learner.struggling_concepts),
        intro_message=_intro(len(scheduled), learner.struggling_concepts, stats),
        cards=scheduled,
        stats=stats,
    )

    yield _sse("progress", {"stage": "exporting"})
    store = get_artifact_store()
    artifacts = await export_deck(
        deck,
        conversation_id=inp.conversation_id,
        store=store,
    )
    yield _sse(
        "progress",
        {"stage": "exported", "artifacts": [a.filename for a in artifacts]},
    )

    response_text = deck.intro_message
    yield _sse("token", {"delta": response_text})

    concepts_covered = sorted({c.concept for c in scheduled if c.concept})
    output = GeneratorOutput(
        response=response_text,
        generator_id=settings.generator_id,
        output_type=settings.output_type,
        artifacts=artifacts,
        sources=inp.context_chunks,
        learner_updates=LearnerUpdates(concepts_covered=concepts_covered),
        metadata={
            "deck_data": deck.model_dump(),
            "stats": stats,
        },
    )
    yield _sse("done", output.model_dump())
