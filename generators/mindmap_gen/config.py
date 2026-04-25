from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OLLAMA_URL: str = "http://localhost:11434"
    MODEL_NAME: str = "llama3.1:8b"
    ARTIFACTS_DIR: str = "./artifacts"
    DEFAULT_SIZE: str = "standard"
    MAX_NODES: int = 60
    # Public origin under which `/artifacts/<filename>` is reachable from a
    # browser. Compose maps the container port to the host, so the default
    # works for local dev. Override in production / non-localhost deployments.
    PUBLIC_BASE_URL: str = "http://localhost:8008"


settings = Settings()
