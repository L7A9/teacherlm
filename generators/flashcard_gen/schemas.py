from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


CardType = Literal["basic", "cloze"]
PrioritySource = Literal["struggling", "coverage", "salient"]


class SM2Meta(BaseModel):
    """Per-card SuperMemo-2 state. Fresh cards start unseen (reps=0, interval=0)."""

    ease_factor: float = 2.5
    interval_days: int = 0
    repetitions: int = 0
    due_at: str  # ISO-8601 UTC


class BasicCard(BaseModel):
    type: Literal["basic"] = "basic"
    front: str
    back: str
    concept: str
    source_chunk_id: str
    sm2: SM2Meta | None = None


class ClozeCard(BaseModel):
    type: Literal["cloze"] = "cloze"
    # Anki cloze syntax: "{{c1::photosynthesis}} is the process by which..."
    text: str
    answer: str  # the text that was blanked out (for non-Anki renderers)
    concept: str
    source_chunk_id: str
    sm2: SM2Meta | None = None


Card = Annotated[Union[BasicCard, ClozeCard], Field(discriminator="type")]


class FlashcardDeck(BaseModel):
    title: str
    intro_message: str
    cards: list[Card]
    stats: dict = Field(default_factory=dict)


# ---------- internal planning + mining ----------


class MinedConcept(BaseModel):
    """A candidate concept pulled from chunks by the miner."""

    name: str
    context_sentence: str
    definition: str | None = None
    source_chunk_id: str
    occurrences: int = 1


class PrioritizedConcept(BaseModel):
    concept: MinedConcept
    priority: float
    source: PrioritySource


# ---------- LLM structured output ----------


class BasicCardDraft(BaseModel):
    """What the LLM emits per concept — schema-constrained via ollama format=."""

    front: str
    back: str


class BasicCardBatch(BaseModel):
    """Top-level container so ollama format= has a single object to populate."""

    cards: list[BasicCardDraft]
