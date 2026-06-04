from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


LlmProvider = Literal["ollama", "openai", "anthropic", "openai_compatible"]


class LlmRuntimeSettingsRead(BaseModel):
    enabled: bool = False
    provider: LlmProvider = "ollama"
    model: str = ""
    api_link: str = ""
    api_key_set: bool = False


class ParserRuntimeSettingsRead(BaseModel):
    api_key_set: bool = False


class RuntimeSettingsRead(BaseModel):
    llm: LlmRuntimeSettingsRead = Field(default_factory=LlmRuntimeSettingsRead)
    parser: ParserRuntimeSettingsRead = Field(default_factory=ParserRuntimeSettingsRead)


class LlmRuntimeSettingsUpdate(BaseModel):
    enabled: bool | None = None
    provider: LlmProvider | None = None
    model: str | None = None
    api_link: str | None = None
    api_key: str | None = None


class ParserRuntimeSettingsUpdate(BaseModel):
    api_key: str | None = None


class RuntimeSettingsUpdate(BaseModel):
    llm: LlmRuntimeSettingsUpdate | None = None
    parser: ParserRuntimeSettingsUpdate | None = None
