from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


BloomLevel = Literal["remember", "understand", "apply", "analyze"]
QuestionKind = Literal["mcq", "true_false", "fill_blank"]
SlotKind = Literal["struggling", "coverage", "stretch"]


# ---------- question types ----------


class MCQ(BaseModel):
    type: Literal["mcq"] = "mcq"
    bloom_level: BloomLevel
    question: str
    options: list[str] = Field(min_length=2)
    correct_index: int = Field(ge=0)
    explanation: str
    concept: str
    source_chunk_id: str


class TrueFalse(BaseModel):
    type: Literal["true_false"] = "true_false"
    bloom_level: BloomLevel
    question: str
    answer: bool
    explanation: str
    concept: str
    source_chunk_id: str


class FillBlank(BaseModel):
    type: Literal["fill_blank"] = "fill_blank"
    bloom_level: BloomLevel
    question: str  # contains "____" placeholder
    answer: str
    accepted_answers: list[str] = Field(default_factory=list)
    explanation: str
    concept: str
    source_chunk_id: str


Question = Annotated[
    Union[MCQ, TrueFalse, FillBlank],
    Field(discriminator="type"),
]


# ---------- top-level quiz ----------


class QuizOutput(BaseModel):
    title: str
    intro_message: str  # teacher voice
    questions: list[Question]
    bloom_distribution: dict[str, int] = Field(default_factory=dict)


# ---------- planning + extraction ----------


class ConceptCard(BaseModel):
    """One concept extracted from chunks, tagged by Bloom's level."""

    name: str
    bloom_level: BloomLevel
    description: str = ""
    source_chunk_ids: list[str] = Field(default_factory=list)


class ExtractedConcepts(BaseModel):
    """Structured-output target for the concept extractor."""

    remember: list[ConceptCard] = Field(default_factory=list)
    understand: list[ConceptCard] = Field(default_factory=list)
    apply: list[ConceptCard] = Field(default_factory=list)
    analyze: list[ConceptCard] = Field(default_factory=list)


class QuestionSlot(BaseModel):
    """A planned question — concept + Bloom level + type — before generation."""

    concept: str
    bloom_level: BloomLevel
    kind: QuestionKind
    slot_kind: SlotKind  # why we're asking it: struggling / coverage / stretch


class QuizPlan(BaseModel):
    slots: list[QuestionSlot]
    total: int
    counts: dict[SlotKind, int]
