from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from teacherlm_core.schemas.generator_io import LearnerUpdates
from teacherlm_core.schemas.learner_state import LearnerState

from db.models import LearnerStateRecord


# Mastery math (per root CLAUDE.md):
#   demonstrated: mastery += 0.2 * (1 - mastery)
#   struggled:    mastery *= 0.7
# Classification thresholds for the understood / struggling lists.
_DEMONSTRATE_STEP = 0.2
_STRUGGLE_DECAY = 0.7
_UNDERSTOOD_THRESHOLD = 0.7
_STRUGGLING_THRESHOLD = 0.3


class LearnerTracker:
    """Loads and updates per-conversation learner state.

    State is persisted in LearnerStateRecord.state_json as a superset of the
    `LearnerState` schema fields plus a private `_encounters` map recording
    how many times each concept has been covered.
    """

    async def load_state(self, session: AsyncSession, conversation_id: uuid.UUID) -> LearnerState:
        raw = await self._load_raw(session, conversation_id)
        return self._to_learner_state(conversation_id, raw)

    async def apply_updates(
        self,
        session: AsyncSession,
        conversation_id: uuid.UUID,
        updates: LearnerUpdates,
        *,
        bump_turn: bool = True,
    ) -> LearnerState:
        record = await self._get_or_create(session, conversation_id)
        raw = dict(record.state_json or {})

        mastery: dict[str, float] = dict(raw.get("mastery_scores", {}))
        encounters: dict[str, int] = dict(raw.get("_encounters", {}))
        prev_mastery = dict(mastery)

        for concept in updates.concepts_covered:
            encounters[concept] = encounters.get(concept, 0) + 1
            mastery.setdefault(concept, 0.0)

        for concept in updates.concepts_demonstrated:
            current = mastery.get(concept, 0.0)
            mastery[concept] = min(1.0, current + _DEMONSTRATE_STEP * (1.0 - current))
            encounters.setdefault(concept, 0)

        for concept in updates.concepts_struggled:
            current = mastery.get(concept, 0.0)
            mastery[concept] = max(0.0, current * _STRUGGLE_DECAY)
            encounters.setdefault(concept, 0)

        understood = sorted(c for c, m in mastery.items() if m >= _UNDERSTOOD_THRESHOLD)
        struggling = sorted(c for c, m in mastery.items() if m <= _STRUGGLING_THRESHOLD and c in encounters)

        session_turns = int(raw.get("session_turns", 0)) + (1 if bump_turn else 0)
        prev_tsp = int(raw.get("turns_since_progress", 0))
        progressed = any(mastery.get(c, 0.0) > prev_mastery.get(c, 0.0) for c in mastery)
        turns_since_progress = 0 if progressed else prev_tsp + (1 if bump_turn else 0)

        raw.update(
            {
                "understood_concepts": understood,
                "struggling_concepts": struggling,
                "mastery_scores": mastery,
                "session_turns": session_turns,
                "turns_since_progress": turns_since_progress,
                "_encounters": encounters,
            }
        )
        record.state_json = raw
        await session.flush()
        return self._to_learner_state(conversation_id, raw)

    async def reset(self, session: AsyncSession, conversation_id: uuid.UUID) -> None:
        record = await self._get_or_create(session, conversation_id)
        record.state_json = {}
        await session.flush()

    # --- internals ---

    async def _load_raw(self, session: AsyncSession, conversation_id: uuid.UUID) -> dict[str, Any]:
        result = await session.execute(
            select(LearnerStateRecord).where(LearnerStateRecord.conversation_id == conversation_id)
        )
        record = result.scalar_one_or_none()
        return dict(record.state_json or {}) if record else {}

    async def _get_or_create(
        self, session: AsyncSession, conversation_id: uuid.UUID
    ) -> LearnerStateRecord:
        result = await session.execute(
            select(LearnerStateRecord).where(LearnerStateRecord.conversation_id == conversation_id)
        )
        record = result.scalar_one_or_none()
        if record is None:
            record = LearnerStateRecord(conversation_id=conversation_id, state_json={})
            session.add(record)
            await session.flush()
        return record

    @staticmethod
    def _to_learner_state(conversation_id: uuid.UUID, raw: dict[str, Any]) -> LearnerState:
        return LearnerState(
            conversation_id=str(conversation_id),
            understood_concepts=list(raw.get("understood_concepts", [])),
            struggling_concepts=list(raw.get("struggling_concepts", [])),
            mastery_scores=dict(raw.get("mastery_scores", {})),
            session_turns=int(raw.get("session_turns", 0)),
            turns_since_progress=int(raw.get("turns_since_progress", 0)),
        )


_tracker: LearnerTracker | None = None


def get_learner_tracker() -> LearnerTracker:
    global _tracker
    if _tracker is None:
        _tracker = LearnerTracker()
    return _tracker
