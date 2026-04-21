from collections.abc import AsyncIterator
from functools import lru_cache
from importlib.resources import files
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel
from teacherlm_core.llm.ollama_client import OllamaClient
from teacherlm_core.llm.structured import generate_structured

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
    """Compose: shared teacher voice + local prompt formatted with kwargs."""
    voice = load_teacher_voice()
    body = load_local_prompt(local_prompt_name).format(**fmt)
    return f"{voice}\n\n---\n\n{body}"


class LLMService:
    """Quiz-specific Ollama wrapper.

    Three logical roles, all backed by configurable models:
      - chat:       free-form teacher voice (intro message)
      - extraction: low-temp structured extraction (concepts)
      - generation: medium-temp structured generation (questions)
    """

    def __init__(self) -> None:
        s = get_settings()
        self._s = s
        self.chat = OllamaClient(s.ollama_host, s.chat_model)
        self.extraction = OllamaClient(s.ollama_host, s.extraction_model)
        self.generation = OllamaClient(s.ollama_host, s.generation_model)

    async def stream_reply(
        self,
        system: str,
        user_message: str,
    ) -> AsyncIterator[str]:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user_message},
        ]
        async for chunk in self.chat.stream_chat(
            messages=messages,
            options={"temperature": self._s.chat_temperature},
        ):
            yield chunk

    async def reply(self, system: str, user_message: str) -> str:
        response = await self.chat.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_message},
            ],
            stream=False,
            options={"temperature": self._s.chat_temperature},
        )
        return response["message"]["content"].strip()

    async def extract_structured(
        self,
        system: str,
        user_message: str,
        schema: type[T],
    ) -> T:
        return await generate_structured(
            client=self.extraction,
            schema=schema,
            system_prompt=system,
            user_prompt=user_message,
        )

    async def generate_structured(
        self,
        system: str,
        user_message: str,
        schema: type[T],
    ) -> T:
        return await generate_structured(
            client=self.generation,
            schema=schema,
            system_prompt=system,
            user_prompt=user_message,
        )


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()
