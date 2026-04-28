from pydantic import BaseModel, Field

from teacherlm_core.schemas.chunk import Chunk
from teacherlm_core.schemas.learner_state import LearnerState


class GeneratorInput(BaseModel):
    conversation_id: str
    user_message: str
    context_chunks: list[Chunk]
    learner_state: LearnerState
    chat_history: list[dict]
    options: dict = Field(default_factory=dict)


class GeneratorArtifact(BaseModel):
    type: str
    url: str
    filename: str
    # Storage key (e.g. MinIO object key) so the platform can re-sign URLs
    # when serving message history — presigned URLs expire (default 1h) but
    # the underlying object survives container restarts.
    key: str | None = None


class LearnerUpdates(BaseModel):
    concepts_covered: list[str] = Field(default_factory=list)
    concepts_demonstrated: list[str] = Field(default_factory=list)
    concepts_struggled: list[str] = Field(default_factory=list)


class GeneratorOutput(BaseModel):
    response: str
    generator_id: str
    output_type: str
    artifacts: list[GeneratorArtifact] = Field(default_factory=list)
    sources: list[Chunk] = Field(default_factory=list)
    learner_updates: LearnerUpdates = Field(default_factory=LearnerUpdates)
    metadata: dict = Field(default_factory=dict)
