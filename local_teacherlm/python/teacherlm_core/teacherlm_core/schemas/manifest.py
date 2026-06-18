from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


GeneratorTransport = Literal[
    "local_inprocess",
    "local_http",
    "local_process",
    "mcp_stdio",
    "mcp_http",
]

OutputType = Literal[
    "text",
    "quiz",
    "report",
    "presentation",
    "chart",
    "podcast",
    "mindmap",
]

RetrievalMode = Literal[
    "semantic_topk",
    "coverage_broad",
    "narrative_arc",
    "topic_clusters",
    "relationship_dense",
]


class GeneratorPermissions(BaseModel):
    read_context: bool = True
    read_learner_state: bool = True
    write_artifacts: bool = False
    network: bool = False


class GeneratorManifest(BaseModel):
    model_config = ConfigDict(extra="allow")

    generator_id: str
    display_name: str
    contract_version: str = "1.0.0"
    output_type: OutputType
    enabled: bool = True
    transport: GeneratorTransport = "local_inprocess"
    endpoint: str = "inprocess"
    retrieval_mode: RetrievalMode = "semantic_topk"
    artifact_types: list[str] = Field(default_factory=list)
    permissions: GeneratorPermissions = Field(default_factory=GeneratorPermissions)
    options_schema: dict = Field(default_factory=dict)
    capabilities: list[str] = Field(default_factory=list)
    is_chat_default: bool = False

