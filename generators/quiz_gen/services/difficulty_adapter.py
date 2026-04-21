from __future__ import annotations

import random
from itertools import cycle

from teacherlm_core.schemas.learner_state import LearnerState

from ..config import get_settings
from ..schemas import (
    BloomLevel,
    ConceptCard,
    ExtractedConcepts,
    QuestionKind,
    QuestionSlot,
    QuizPlan,
    SlotKind,
)


# Bloom levels we draw from, ordered easiest → hardest. Used as fallback rotation.
_BLOOM_ORDER: list[BloomLevel] = ["remember", "understand", "apply", "analyze"]

# Default per-slot question-type rotation. Keeps things varied without parameters.
_KIND_ROTATION: list[QuestionKind] = ["mcq", "mcq", "true_false", "fill_blank"]


def _flatten(extracted: ExtractedConcepts) -> dict[str, ConceptCard]:
    """Flatten all extracted concepts to a name → card map."""
    out: dict[str, ConceptCard] = {}
    for level in _BLOOM_ORDER:
        for card in getattr(extracted, level):
            out.setdefault(card.name, card)
    return out


def _split_counts(n_total: int, settings) -> dict[SlotKind, int]:
    """Largest-remainder allocation across the three slot kinds."""
    weights: dict[SlotKind, float] = {
        "struggling": settings.mix_struggling,
        "coverage": settings.mix_coverage,
        "stretch": settings.mix_stretch,
    }
    raw = {k: n_total * w for k, w in weights.items()}
    floors = {k: int(v) for k, v in raw.items()}
    remainder = n_total - sum(floors.values())
    # Distribute the remainder by largest fractional part.
    order = sorted(raw, key=lambda k: raw[k] - floors[k], reverse=True)
    for k in order[:remainder]:
        floors[k] += 1
    return floors


def _pick_card(
    name: str,
    flat: dict[str, ConceptCard],
    fallback_pool: list[ConceptCard],
    rng: random.Random,
) -> ConceptCard | None:
    if name in flat:
        return flat[name]
    if not fallback_pool:
        return None
    # Best-effort: match by case-insensitive substring before falling back to random.
    needle = name.lower()
    matches = [c for c in fallback_pool if needle in c.name.lower() or c.name.lower() in needle]
    return rng.choice(matches) if matches else rng.choice(fallback_pool)


def _stretch_level(card: ConceptCard) -> BloomLevel:
    """Push to a harder level when generating stretch questions."""
    idx = _BLOOM_ORDER.index(card.bloom_level)
    return _BLOOM_ORDER[min(idx + 1, len(_BLOOM_ORDER) - 1)]


def _slots_for(
    n: int,
    cards: list[ConceptCard],
    slot_kind: SlotKind,
    kinds: cycle,
    rng: random.Random,
    *,
    bump_difficulty: bool = False,
) -> list[QuestionSlot]:
    """Round-robin n slots across the given concept cards."""
    if n <= 0 or not cards:
        return []
    out: list[QuestionSlot] = []
    pool = cycle(cards)
    for _ in range(n):
        card = next(pool)
        bloom = _stretch_level(card) if bump_difficulty else card.bloom_level
        out.append(
            QuestionSlot(
                concept=card.name,
                bloom_level=bloom,
                kind=next(kinds),
                slot_kind=slot_kind,
            )
        )
    rng.shuffle(out)
    return out


def plan_question_mix(
    learner_state: LearnerState,
    extracted: ExtractedConcepts,
    n_total: int,
    *,
    seed: int | None = None,
    allowed_kinds: list[QuestionKind] | None = None,
) -> QuizPlan:
    """Plan a quiz with a 60/30/10 split across struggling / coverage / stretch.

    Resilient to a cold-start learner: empty struggling/understood lists are
    handled by reallocating to coverage so we never produce a 0-question quiz.

    When ``allowed_kinds`` is provided, only those question kinds are used in
    the rotation — otherwise the default mix (mcq-heavy) is used.
    """
    settings = get_settings()
    rng = random.Random(seed)
    flat = _flatten(extracted)
    all_cards = list(flat.values())

    if not all_cards:
        return QuizPlan(slots=[], total=0, counts={"struggling": 0, "coverage": 0, "stretch": 0})

    counts = _split_counts(n_total, settings)
    rotation = list(allowed_kinds) if allowed_kinds else list(_KIND_ROTATION)

    # --- pick concept pools ---
    struggling_cards: list[ConceptCard] = []
    for name in learner_state.struggling_concepts:
        card = _pick_card(name, flat, all_cards, rng)
        if card and card not in struggling_cards:
            struggling_cards.append(card)

    understood_cards: list[ConceptCard] = []
    for name in learner_state.understood_concepts:
        card = _pick_card(name, flat, all_cards, rng)
        if card and card not in understood_cards:
            understood_cards.append(card)

    coverage_cards = list(all_cards)
    rng.shuffle(coverage_cards)

    # --- reallocate empty buckets to coverage ---
    if not struggling_cards:
        counts["coverage"] += counts["struggling"]
        counts["struggling"] = 0
    if not understood_cards:
        counts["coverage"] += counts["stretch"]
        counts["stretch"] = 0

    kinds = cycle(rotation)

    slots: list[QuestionSlot] = []
    slots.extend(_slots_for(counts["struggling"], struggling_cards, "struggling", kinds, rng))
    slots.extend(_slots_for(counts["coverage"], coverage_cards, "coverage", kinds, rng))
    slots.extend(
        _slots_for(
            counts["stretch"],
            understood_cards,
            "stretch",
            kinds,
            rng,
            bump_difficulty=True,
        )
    )

    return QuizPlan(slots=slots, total=len(slots), counts=counts)
