from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class CoreSettings(BaseSettings):
    """Runtime configuration shared by the platform and all generators.

    Values are loaded from environment variables (prefix `TEACHERLM_`) or a
    local `.env` file. Example: `TEACHERLM_OLLAMA_MODEL=llama3.1:8b`.
    """

    model_config = SettingsConfigDict(
        env_prefix="TEACHERLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Ollama ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"
    ollama_timeout_s: float = 120.0

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "teacherlm_chunks"

    # --- Embedding / reranking ---
    embedding_model: str = "intfloat/multilingual-e5-large"
    reranker_model: str = "BAAI/bge-reranker-base"

    # --- Retrieval defaults ---
    retrieval_top_k: int = Field(default=10, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=50)
    bm25_top_k: int = Field(default=20, ge=1, le=200)
    dense_top_k: int = Field(default=20, ge=1, le=200)

    # --- Confidence thresholds ---
    groundedness_warn_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    coverage_warn_threshold: float = Field(default=0.4, ge=0.0, le=1.0)


def get_settings() -> CoreSettings:
    """Construct a fresh CoreSettings instance from the current environment."""
    return CoreSettings()
