from __future__ import annotations

from teacherlm_core.schemas.generator_io import LearnerUpdates
from teacherlm_core.schemas.learner_state import LearnerState

from local_api.db import get_store


class LearnerService:
    def load(self, conversation_id: str) -> LearnerState:
        return LearnerState.model_validate(get_store().load_learner_state(conversation_id))

    def apply_updates(self, conversation_id: str, updates: LearnerUpdates) -> LearnerState:
        state = self.load(conversation_id)
        understood = set(state.understood_concepts)
        struggling = set(state.struggling_concepts)
        mastery = dict(state.mastery_scores)

        for concept in updates.concepts_covered:
            mastery.setdefault(concept, 0.15)
        for concept in updates.concepts_demonstrated:
            understood.add(concept)
            struggling.discard(concept)
            mastery[concept] = max(mastery.get(concept, 0.0), 0.75)
        for concept in updates.concepts_struggled:
            struggling.add(concept)
            mastery[concept] = min(mastery.get(concept, 0.35), 0.35)

        next_state = state.model_copy(
            update={
                "understood_concepts": sorted(understood),
                "struggling_concepts": sorted(struggling),
                "mastery_scores": mastery,
                "session_turns": state.session_turns + 1,
            }
        )
        get_store().save_learner_state(conversation_id, next_state.model_dump())
        return next_state


_learner_service: LearnerService | None = None


def get_learner_service() -> LearnerService:
    global _learner_service
    if _learner_service is None:
        _learner_service = LearnerService()
    return _learner_service

