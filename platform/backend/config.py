from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    app_name: str = "teacherlm-backend"
    environment: str = Field(default="development")
    debug: bool = True
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ]
    )

    # --- Database ---
    database_url: str = "postgresql+asyncpg://teacherlm:teacherlm@localhost:5432/teacherlm"

    # --- Redis (arq) ---
    redis_url: str = "redis://localhost:6379/0"

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None

    # --- MinIO ---
    minio_endpoint: str = "localhost:9000"
    # Browser-facing endpoint used when re-signing artifact URLs returned to
    # the frontend. Defaults to minio_endpoint when unset; override in compose
    # so the backend signs against `localhost:9000` (browser) instead of the
    # in-network `minio:9000` (only reachable inside the compose network).
    minio_public_endpoint: str | None = None
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "teacherlm"
    minio_secure: bool = False
    artifact_url_ttl_s: int = 3600

    # --- LlamaCloud ---
    llama_cloud_api_key: str = ""
    llama_cloud_base_url: str | None = None
    llama_cloud_poll_interval_s: float = 2.0
    llama_cloud_timeout_s: float = 600.0

    # --- Embeddings ---
    embedding_model: str = "intfloat/multilingual-e5-large"
    embedding_dim: int = 1024
    embedding_batch_size: int = 8
    embedding_model_candidates: list[str] = Field(
        default_factory=lambda: [
            "BAAI/bge-m3",
            "intfloat/multilingual-e5-large",
            "BAAI/bge-small-en-v1.5",
        ]
    )

    # --- Chunking ---
    chunk_max_tokens: int = 512
    chunk_overlap_tokens: int = 50

    # --- Retrieval ---
    retrieval_top_k: int = 8
    retrieval_dense_candidate_k: int = 80
    retrieval_sparse_candidate_k: int = 80
    retrieval_context_expansion_enabled: bool = True
    retrieval_neighbor_window: int = 1
    retrieval_parent_max_tokens: int = 900
    retrieval_expansion_max_chars: int = 4500
    retrieval_rerank_enabled: bool = True
    retrieval_rerank_warmup_enabled: bool = True
    retrieval_reranker_model: str = "BAAI/bge-reranker-base"
    retrieval_rerank_candidate_k: int = 50
    retrieval_rerank_top_k: int = 8
    retrieval_rerank_modes: list[str] = Field(
        default_factory=lambda: [
            "semantic_topk",
            "coverage_broad",
            "narrative_arc",
            "topic_clusters",
            "relationship_dense",
        ]
    )

    # --- Course context ---
    course_context_max_chunks: int = 5000
    course_context_chunk_budget: int = 96
    course_context_section_budget: int = 48

    # --- Ingestion worker ---
    ingestion_max_jobs: int = 1
    ingestion_job_timeout_s: int = 3600

    # --- Generators registry ---
    generators_registry_path: Path = Path(__file__).resolve().parents[2] / "generators_registry.json"

    # --- Ollama (shared with teacherlm_core) ---
    ollama_host: str = "http://localhost:11434"
    ollama_chat_model: str = "llama3.1:8b-instruct-q4_K_M"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
