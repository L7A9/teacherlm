from functools import lru_cache
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="TEACHER_GEN_",
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    generator_id: str = "teacher_gen"
    output_type: str = "text"
    version: str = "0.1.0"

    host: str = "0.0.0.0"
    port: int = 8001

    ollama_host: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices("TEACHER_GEN_OLLAMA_HOST", "OLLAMA_HOST"),
    )
    chat_model: str = Field(
        default="llama3.1:8b-instruct-q4_K_M",
        validation_alias=AliasChoices("TEACHER_GEN_CHAT_MODEL", "OLLAMA_CHAT_MODEL"),
    )
    analysis_model: str = Field(
        default="llama3.1:8b-instruct-q4_K_M",
        validation_alias=AliasChoices(
            "TEACHER_GEN_ANALYSIS_MODEL",
            "OLLAMA_ANALYSIS_MODEL",
            "OLLAMA_CHAT_MODEL",
        ),
    )
    extraction_model: str = Field(
        default="llama3.1:8b-instruct-q4_K_M",
        validation_alias=AliasChoices(
            "TEACHER_GEN_EXTRACTION_MODEL",
            "OLLAMA_EXTRACTION_MODEL",
            "OLLAMA_CHAT_MODEL",
        ),
    )

    chat_temperature: float = 0.4
    analysis_temperature: float = 0.1
    extraction_temperature: float = 0.1

    max_context_chunks: int = 8

    confusion_guide_threshold: float = 0.7
    stuck_turns_threshold: int = 4

    # Cross-encoder logit below which we consider retrieval off-topic and refuse.
    # bge-reranker-base: clearly relevant ~>0, clearly irrelevant <-3.
    min_relevance_score: float = -2.0

    request_timeout_s: float = 120.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
