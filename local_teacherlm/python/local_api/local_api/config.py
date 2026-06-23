from __future__ import annotations

import os
import sys
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _runtime_root() -> Path:
    """Return the source root or PyInstaller's extracted resource directory."""
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parents[3]


PROJECT_ROOT = _runtime_root()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "teacherlm-local-api"
    host: str = "127.0.0.1"
    port: int = 8765
    debug: bool = True
    app_data_dir: Path | None = Field(default=None, alias="TEACHERLM_APP_DATA_DIR")
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:1420",
            "http://127.0.0.1:1420",
            "tauri://localhost",
            "http://tauri.localhost",
        ]
    )

    generators_registry_path: Path = PROJECT_ROOT / "generators_registry.json"

    default_ollama_base_url: str = "http://localhost:11434"
    default_ollama_model: str = "llama3.2"
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
    retrieval_top_k: int = 16
    retrieval_candidate_k: int = 80
    retrieval_dense_candidate_k: int = 80
    retrieval_sparse_candidate_k: int = 80
    retrieval_context_expansion_enabled: bool = True
    retrieval_neighbor_window: int = 1
    retrieval_expansion_max_chars: int = 4500
    retrieval_hyde_enabled: bool = True
    retrieval_hyde_max_chars: int = 900
    retrieval_rerank_enabled: bool = True
    retrieval_reranker_model: str = "BAAI/bge-reranker-base"
    retrieval_rerank_candidate_k: int = 50
    retrieval_rerank_top_k: int = 16
    retrieval_graph_enabled: bool = True
    chunk_max_chars: int = 1800
    chunk_overlap_chars: int = 180
    chunk_semantic_min_chars: int = 600
    chunk_semantic_similarity_threshold: float = 0.72
    generated_questions_per_chunk: int = 4
    llama_cloud_api_key: str = ""
    llama_cloud_base_url: str | None = None
    llama_cloud_poll_interval_s: float = 2.0
    llama_cloud_timeout_s: float = 600.0

    @property
    def data_dir(self) -> Path:
        if self.app_data_dir is not None:
            return self.app_data_dir
        if os.name == "nt":
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        else:
            base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
        return base / "TeacherLM"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "teacherlm.db"

    @property
    def secret_key_path(self) -> Path:
        return self.data_dir / "secrets.key"

    @property
    def models_dir(self) -> Path:
        return self.data_dir / "models"

    @property
    def embedding_cache_dir(self) -> Path:
        return self.models_dir / "embeddings"

    @property
    def reranker_cache_dir(self) -> Path:
        return self.models_dir / "rerankers"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
