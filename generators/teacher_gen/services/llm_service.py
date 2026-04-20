from collections.abc import AsyncIterator
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
from teacherlm_core.llm.ollama_client import OllamaClient

from ..config import get_settings

T = TypeVar("T", bound=BaseModel)

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache
def load_teacher_voice() -> str:
    return (files("teacherlm_core.prompts") / "teacher_voice.txt").read_text(
        encoding="utf-8"
    )


@lru_cache
def load_local_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def build_system_prompt(local_prompt_name: str, **fmt: object) -> str:
    voice = load_teacher_voice()
    body = load_local_prompt(local_prompt_name).format(**fmt)
    return f"{voice}\n\n---\n\n{body}"


class LLMService:
    def __init__(self) -> None:
        s = get_settings()
        self.chat = OllamaClient(s.ollama_host, s.chat_model)
        self.analysis = OllamaClient(s.ollama_host, s.analysis_model)
        self.extraction = OllamaClient(s.ollama_host, s.extraction_model)
        self._s = s

    async def stream_reply(
        self,
        system: str,
        chat_history: list[dict],
        user_message: str,
    ) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": system},
            *chat_history,
            {"role": "user", "content": user_message},
        ]
        async for chunk in self.chat.stream_chat(
            messages=messages,
            options={"temperature": self._s.chat_temperature},
        ):
            yield chunk

    async def analyze_structured(
        self,
        system: str,
        user_message: str,
        schema: type[T],
    ) -> T:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        return await self.analysis.chat_structured(
            messages=messages,
            schema=schema,
            options={"temperature": self._s.analysis_temperature},
        )

    async def extract_structured(
        self,
        system: str,
        user_message: str,
        schema: type[T],
    ) -> T:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        return await self.extraction.chat_structured(
            messages=messages,
            schema=schema,
            options={"temperature": self._s.extraction_temperature},
        )


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()
