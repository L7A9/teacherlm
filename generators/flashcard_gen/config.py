from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FLASHCARD_GEN_",
        env_file=".env",
        extra="ignore",
    )

    generator_id: str = "flashcard_gen"
    output_type: str = "flashcards"
    version: str = "0.1.0"

    host: str = "0.0.0.0"
    port: int = 8005

    ollama_host: str = "http://localhost:11434"
    chat_model: str = "llama3.1:8b-instruct-q4_K_M"
    generation_model: str = "llama3.1:8b-instruct-q4_K_M"

    chat_temperature: float = 0.4
    generation_temperature: float = 0.3

    # Deck shape
    default_card_count: int = 12
    min_card_count: int = 3
    max_card_count: int = 60

    # Mix of basic (LLM) vs cloze (spaCy) cards.
    basic_ratio: float = 0.65
    cloze_ratio: float = 0.35

    # Concept mining
    spacy_model: str = "en_core_web_sm"
    min_concept_chars: int = 3
    max_concept_chars: int = 64
    # NER types we keep. PERSON / ORG / GPE / LOC / NORP / FAC are deliberately
    # excluded by default — on academic / course material they overwhelmingly
    # catch author names, university affiliations, and city/country mentions
    # from cover pages, which are not what the student wants to study.
    # Noun-chunk extraction + the definition regex cover real domain terms.
    ner_keep_types: tuple[str, ...] = (
        "PRODUCT",
        "EVENT",
        "WORK_OF_ART",
        "LAW",
    )

    # Learner-state thresholds
    mastery_skip_threshold: float = 0.85
    struggling_boost: float = 2.0
    coverage_boost: float = 1.0

    # Deduplication
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    dedupe_similarity: float = 0.85

    # SM-2 defaults
    sm2_initial_ease: float = 2.5
    sm2_initial_interval_days: int = 0

    # MinIO (artifact storage)
    minio_endpoint: str = "localhost:9000"
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
