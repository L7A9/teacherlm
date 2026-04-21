from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="QUIZ_GEN_",
        env_file=".env",
        extra="ignore",
    )

    generator_id: str = "quiz_gen"
    output_type: str = "quiz"
    version: str = "0.1.0"

    host: str = "0.0.0.0"
    port: int = 8002

    ollama_host: str = "http://localhost:11434"
    chat_model: str = "llama3.1:8b-instruct-q4_K_M"
    extraction_model: str = "llama3.1:8b-instruct-q4_K_M"
    generation_model: str = "llama3.1:8b-instruct-q4_K_M"

    chat_temperature: float = 0.4
    extraction_temperature: float = 0.1
    generation_temperature: float = 0.3

    # Quiz shape defaults
    default_question_count: int = 8
    min_question_count: int = 3
    max_question_count: int = 30

    # Difficulty mix (must sum to 1.0)
    mix_struggling: float = 0.6
    mix_coverage: float = 0.3
    mix_stretch: float = 0.1

    # Distractor engine
    # Off by default: the fastembed-bigram candidate pool often surfaces
    # fragments ("of the", "by which") that aren't answer-shaped, which ends up
    # degrading the LLM's original options rather than improving them.
    enhance_distractors: bool = False
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    distractor_sim_min: float = 0.4
    distractor_sim_max: float = 0.7
    distractor_pool_size: int = 24
    distractors_per_mcq: int = 3

    # MinIO (artifact storage)
    minio_endpoint: str = "localhost:9000"
    # Endpoint used when signing URLs handed to the browser. Defaults to
    # `minio_endpoint` if unset. In docker-compose, set this to the host the
    # browser can reach (e.g. "localhost:9000") while `minio_endpoint` stays
    # the in-network hostname ("minio:9000").
    minio_public_endpoint: str | None = None
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "teacherlm"
    minio_secure: bool = False
    artifact_url_ttl_s: int = 3600

    request_timeout_s: float = 240.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
