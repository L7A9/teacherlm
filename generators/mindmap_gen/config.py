from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        populate_by_name=True,
    )

    OLLAMA_URL: str = Field(
        default="http://localhost:11434",
        validation_alias=AliasChoices(
            "OLLAMA_URL",
            "MINDMAP_GEN_OLLAMA_HOST",
            "OLLAMA_HOST",
        ),
    )
    MODEL_NAME: str = Field(
        default="llama3.1:8b",
        validation_alias=AliasChoices(
            "MODEL_NAME",
            "MINDMAP_GEN_MODEL",
            "OLLAMA_CHAT_MODEL",
        ),
    )
    ARTIFACTS_DIR: str = "./artifacts"
    DEFAULT_SIZE: str = "standard"
    MAX_NODES: int = 60
    LLM_CALL_TIMEOUT_S: float = 120.0
    LLM_KEEPALIVE_INTERVAL_S: float = 15.0
    # Public origin under which `/artifacts/<filename>` is reachable from a
    # browser. Compose maps the container port to the host, so the default
    # works for local dev. Override in production / non-localhost deployments.
    PUBLIC_BASE_URL: str = "http://localhost:8008"


settings = Settings()
