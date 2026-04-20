from typing import Literal
from pydantic import BaseModel, Field


ResponseMode = Literal["explain", "guide", "quiz_back", "affirm"]


class QueryAnalysis(BaseModel):
    intent: Literal[
        "new_question",
        "clarification",
        "confusion",
        "confirmation",
        "follow_up",
    ]
    confusion_level: float = Field(ge=0.0, le=1.0)
    targets_concept: str | None = None
    requires_direct_answer: bool


class ConceptExtraction(BaseModel):
    covered: list[str] = Field(default_factory=list)
    demonstrated_understanding: list[str] = Field(default_factory=list)
    showed_confusion: list[str] = Field(default_factory=list)


class TeacherResponse(BaseModel):
    response: str
    mode: ResponseMode
    confidence: dict
    learner_updates: ConceptExtraction
    sources: list[dict]
