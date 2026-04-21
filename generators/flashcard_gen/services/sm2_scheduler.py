from __future__ import annotations

from datetime import UTC, datetime

from ..config import get_settings
from ..schemas import Card, SM2Meta


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def initial_sm2() -> SM2Meta:
    settings = get_settings()
    return SM2Meta(
        ease_factor=settings.sm2_initial_ease,
        interval_days=settings.sm2_initial_interval_days,
        repetitions=0,
        due_at=_now_iso(),
    )


def schedule_cards(cards: list[Card]) -> list[Card]:
    """Attach fresh SM-2 metadata to each card. Idempotent — if `sm2` is
    already set, it's left alone (useful if we later re-run pipeline stages).
    """
    out: list[Card] = []
    for card in cards:
        if card.sm2 is not None:
            out.append(card)
            continue
        out.append(card.model_copy(update={"sm2": initial_sm2()}))
    return out
