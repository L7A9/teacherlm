from pydantic import BaseModel, Field


class LearnerState(BaseModel):
    conversation_id: str
    understood_concepts: list[str] = Field(default_factory=list)
    struggling_concepts: list[str] = Field(default_factory=list)
    mastery_scores: dict[str, float] = Field(default_factory=dict)
    session_turns: int = 0
    turns_since_progress: int = 0
