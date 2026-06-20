from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ConversationCreate(BaseModel):
    title: str = "New course"


class ConversationUpdate(BaseModel):
    title: str | None = None


class ChatRequest(BaseModel):
    message: str
    source_file_ids: list[str] = Field(default_factory=list)
    options: dict = Field(default_factory=dict)


class GenerateRequest(BaseModel):
    output_type: Literal["quiz", "mindmap", "podcast", "presentation", "report", "chart"]
    prompt: str = ""
    source_file_ids: list[str] = Field(default_factory=list)
    options: dict = Field(default_factory=dict)


class CourseQuizAnswer(BaseModel):
    question_id: str
    option_id: str


class CourseQuizSubmission(BaseModel):
    answers: list[CourseQuizAnswer] = Field(default_factory=list)


class ProviderWrite(BaseModel):
    display_name: str
    provider_type: Literal[
        "ollama",
        "openai",
        "anthropic",
        "openai_compatible",
        "anthropic_compatible",
    ] = "ollama"
    base_url: str = "http://localhost:11434"
    model_name: str = "llama3.2"
    api_key: str | None = None
    is_default_chat: bool = False
    is_default_embedding: bool = False


class ProviderPatch(BaseModel):
    display_name: str | None = None
    provider_type: str | None = None
    base_url: str | None = None
    model_name: str | None = None
    api_key: str | None = None
    is_default_chat: bool | None = None
    is_default_embedding: bool | None = None


class ProviderRead(BaseModel):
    id: str
    display_name: str
    provider_type: str
    base_url: str
    model_name: str
    api_key_set: bool = False
    is_default_chat: bool = False
    is_default_embedding: bool = False
    status: str = "unknown"


class ParserSettingsRead(BaseModel):
    llama_cloud_api_key_set: bool = False
    use_local_parsers_only: bool = True
    status: str = "local"


class ParserSettingsUpdate(BaseModel):
    llama_cloud_api_key: str | None = None
    clear_llama_cloud_api_key: bool = False
    use_local_parsers_only: bool | None = None


class CourseBuilderSettingsRead(BaseModel):
    sequential_unlocking_enabled: bool = True


class CourseBuilderSettingsUpdate(BaseModel):
    sequential_unlocking_enabled: bool | None = None


class RetrievalSettingsRead(BaseModel):
    embedding_model: str
    embedding_dim: int
    embedding_batch_size: int
    embedding_model_candidates: list[str] = Field(default_factory=list)
    retrieval_top_k: int
    retrieval_dense_candidate_k: int
    retrieval_sparse_candidate_k: int
    retrieval_hyde_enabled: bool
    retrieval_rerank_enabled: bool
    retrieval_reranker_model: str
    retrieval_graph_enabled: bool
    index_status: dict = Field(default_factory=dict)


class RetrievalSettingsUpdate(BaseModel):
    embedding_model: str | None = None
    embedding_dim: int | None = None
    embedding_batch_size: int | None = None
    retrieval_top_k: int | None = None
    retrieval_dense_candidate_k: int | None = None
    retrieval_sparse_candidate_k: int | None = None
    retrieval_hyde_enabled: bool | None = None
    retrieval_rerank_enabled: bool | None = None
    retrieval_reranker_model: str | None = None
    retrieval_graph_enabled: bool | None = None


class RuntimeSettingsRead(BaseModel):
    default_chat_provider: ProviderRead | None = None
    default_embedding_provider: ProviderRead | None = None
    parser: ParserSettingsRead = Field(default_factory=ParserSettingsRead)
    retrieval: RetrievalSettingsRead | None = None
