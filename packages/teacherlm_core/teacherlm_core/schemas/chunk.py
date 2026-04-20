from pydantic import BaseModel, Field


class Chunk(BaseModel):
    text: str
    source: str
    score: float
    chunk_id: str
    metadata: dict = Field(default_factory=dict)
