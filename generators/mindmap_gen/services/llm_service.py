from functools import lru_cache
from importlib.resources import files
from pathlib import Path

import httpx

from teacherlm_core.llm.ollama_client import OllamaClient
from teacherlm_core.llm.runtime import (
    build_llm_client_kwargs,
    get_current_llm_options,
    has_llm_override,
)
from teacherlm_core.llm.structured import generate_structured

from ..config import settings
from ..schemas import CourseOutline, SubtopicExpansion, ThemeList

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


@lru_cache
def _load_teacher_voice() -> str:
    return (files("teacherlm_core.prompts") / "teacher_voice.txt").read_text(
        encoding="utf-8"
    )


@lru_cache
def _load_local_prompt(name: str) -> str:
    return (PROMPTS_DIR / name).read_text(encoding="utf-8")


def _build_system_prompt(local_prompt_name: str, **fmt: object) -> str:
    voice = _load_teacher_voice()
    body = _load_local_prompt(local_prompt_name).format(**fmt)
    return f"{voice}\n\n---\n\n{body}"


class LLMService:
    """Mind-map specific Ollama wrapper.

    Exposes the two structured generation calls the pipeline needs:
      - extract_themes:   high-level branch names from full content
      - expand_subtopic:  sub-topics + leaves under one branch
    """

    def __init__(self, override: dict | None = None) -> None:
        cfg = build_llm_client_kwargs(
            default_base_url=settings.OLLAMA_URL,
            default_model=settings.MODEL_NAME,
            options=override,
        )
        self._client = OllamaClient(
            str(cfg["base_url"]),
            str(cfg["model"]),
            provider=str(cfg["provider"]),
            api_key=cfg["api_key"],
        )

    async def is_available(self, timeout_s: float = 3.0) -> tuple[bool, str | None]:
        if self._client.provider != "ollama":
            return True, None
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                response = await client.get(f"{self._client.base_url.rstrip('/')}/api/tags")
                response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 - provider readiness boundary
            return False, str(exc)
        return True, None

    async def extract_themes(
        self, chunks_text: str, n_branches: int
    ) -> ThemeList:
        system = _build_system_prompt(
            "theme_extraction.txt", n_branches=n_branches
        )
        return await generate_structured(
            client=self._client,
            schema=ThemeList,
            system_prompt=system,
            user_prompt=chunks_text,
        )

    async def build_course_outline(
        self, chunks_text: str, n_branches: int
    ) -> CourseOutline:
        system = _build_system_prompt(
            "course_outline.txt", n_branches=n_branches
        )
        return await generate_structured(
            client=self._client,
            schema=CourseOutline,
            system_prompt=system,
            user_prompt=chunks_text,
            options={"temperature": 0.15, "num_ctx": 8192, "num_predict": 1800},
        )

    async def build_batch_outline(self, batch_text: str) -> CourseOutline:
        system = _build_system_prompt("batch_outline.txt")
        return await generate_structured(
            client=self._client,
            schema=CourseOutline,
            system_prompt=system,
            user_prompt=batch_text,
            options={"temperature": 0.1, "num_ctx": 8192, "num_predict": 900},
        )

    async def synthesize_course_outline(
        self, extracted_outlines: str, n_branches: int
    ) -> CourseOutline:
        system = _build_system_prompt(
            "course_synthesis.txt", n_branches=n_branches
        )
        return await generate_structured(
            client=self._client,
            schema=CourseOutline,
            system_prompt=system,
            user_prompt=extracted_outlines,
            options={"temperature": 0.1, "num_ctx": 8192, "num_predict": 1800},
        )

    async def expand_subtopic(
        self, theme: str, relevant_chunks: str
    ) -> SubtopicExpansion:
        system = _build_system_prompt("subtopic_expansion.txt", theme=theme)
        return await generate_structured(
            client=self._client,
            schema=SubtopicExpansion,
            system_prompt=system,
            user_prompt=relevant_chunks,
        )

    async def infer_central_topic(self, chunks_text: str) -> str:
        response = await self._client.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Infer the overall course subject of these documents. "
                        "Prefer an exact title, repeated heading, or central "
                        "discipline from the source over a narrow subtopic. "
                        "Answer in the source language, 2-6 words. Return ONLY "
                        "the subject, no prose."
                    ),
                },
                {"role": "user", "content": chunks_text},
            ],
            stream=False,
            options={"temperature": 0.2},
        )
        return response["message"]["content"].strip().strip('"').strip("'")


@lru_cache
def _default_llm_service() -> LLMService:
    return LLMService()


def get_llm_service() -> LLMService:
    override = get_current_llm_options()
    if has_llm_override(override):
        return LLMService(override)
    return _default_llm_service()
