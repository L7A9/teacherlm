from pydantic import BaseModel, Field


class KnownConcept(BaseModel):
    id: str
    name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    bloom_level: str = "understand"
    importance: float = 0.5
    course_parts: list[dict[str, str]] = Field(default_factory=list)


class ConceptProgress(BaseModel):
    concept_id: str
    name: str
    mastery: float = 0.0
    encounters: int = 0
    struggle_evidence: int = 0


class LearningPhase(BaseModel):
    id: str
    title: str
    summary: str = ""
    order_index: int = 0
    objective_ids: list[str] = Field(default_factory=list)


class ObjectiveProgress(BaseModel):
    objective_id: str
    phase_id: str
    objective_text: str
    bloom_level: str = "understand"
    mastery: float = 0.0
    encounters: int = 0
    struggle_evidence: int = 0
    concept_ids: list[str] = Field(default_factory=list)
    order_index: int = 0


class PhaseProgress(BaseModel):
    phase_id: str
    title: str
    mastery: float = 0.0
    objectives_total: int = 0
    objectives_mastered: int = 0
    struggle_evidence: int = 0
    order_index: int = 0


class LearnerState(BaseModel):
    conversation_id: str
    understood_concepts: list[str] = Field(default_factory=list)
    struggling_concepts: list[str] = Field(default_factory=list)
    mastery_scores: dict[str, float] = Field(default_factory=dict)
    session_turns: int = 0
    turns_since_progress: int = 0
    known_concepts: list[KnownConcept] = Field(default_factory=list)
    concept_progress: list[ConceptProgress] = Field(default_factory=list)
    learning_phases: list[LearningPhase] = Field(default_factory=list)
    objective_progress: list[ObjectiveProgress] = Field(default_factory=list)
    phase_progress: list[PhaseProgress] = Field(default_factory=list)
    remediation_paths: list[dict] = Field(default_factory=list)
