from __future__ import annotations

from teacherlm_core.schemas.learner_state import LearnerState

from ..config import get_settings
from ..schemas import MinedConcept, PrioritizedConcept, PrioritySource


def _matches(concept_name: str, label: str) -> bool:
    """Case-insensitive containment match — handles variant surface forms."""
    c = concept_name.lower()
    l = label.lower()
    return c == l or c in l or l in c


def _is_mastered(
    concept_name: str,
    learner_state: LearnerState,
    threshold: float,
) -> bool:
    for understood in learner_state.understood_concepts:
        if _matches(concept_name, understood):
            mastery = learner_state.mastery_scores.get(understood, 1.0)
            if mastery >= threshold:
                return True
    return False


def _is_struggling(concept_name: str, learner_state: LearnerState) -> bool:
    return any(
        _matches(concept_name, s) for s in learner_state.struggling_concepts
    )


def _score(
    concept: MinedConcept,
    *,
    struggling: bool,
    struggling_boost: float,
    coverage_boost: float,
) -> tuple[float, PrioritySource]:
    # Base salience: occurrence frequency (log-ish dampened so one repeated term
    # doesn't eclipse the rest of the material).
    base = 1.0 + min(concept.occurrences, 5) * 0.2
    # Definitional sentences are gold for cards — boost them.
    if concept.definition:
        base += 0.5

    if struggling:
        return base * struggling_boost, "struggling"
    return base * coverage_boost, "coverage"


def prioritize(
    mined: list[MinedConcept],
    learner_state: LearnerState,
    *,
    limit: int | None = None,
) -> list[PrioritizedConcept]:
    """Rank mined concepts by learner-state relevance.

    - Drops concepts the learner has already mastered (mastery >= threshold).
    - Boosts concepts the learner is struggling with.
    - Falls back to occurrence-based salience otherwise.
    - Returns at most `limit` items (if provided), sorted by priority desc.
    """
    settings = get_settings()
    out: list[PrioritizedConcept] = []

    for concept in mined:
        if _is_mastered(concept.name, learner_state, settings.mastery_skip_threshold):
            continue
        struggling = _is_struggling(concept.name, learner_state)
        priority, source = _score(
            concept,
            struggling=struggling,
            struggling_boost=settings.struggling_boost,
            coverage_boost=settings.coverage_boost,
        )
        out.append(
            PrioritizedConcept(concept=concept, priority=priority, source=source)
        )

    out.sort(key=lambda p: p.priority, reverse=True)
    if limit is not None:
        out = out[:limit]
    return out
