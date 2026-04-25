from functools import lru_cache
from importlib.resources import files
from pathlib import Path

from teacherlm_core.llm.ollama_client import OllamaClient
from teacherlm_core.llm.structured import generate_structured

from ..config import settings
from ..schemas import SubtopicExpansion, ThemeList

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

    def __init__(self) -> None:
        self._client = OllamaClient(settings.OLLAMA_URL, settings.MODEL_NAME)

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
                        "What is the overall subject of these documents? "
                        "Answer in 2-4 words. Return ONLY the subject, no prose."
                    ),
                },
                {"role": "user", "content": chunks_text},
            ],
            stream=False,
            options={"temperature": 0.2},
        )
        return response["message"]["content"].strip().strip('"').strip("'")


@lru_cache
def get_llm_service() -> LLMService:
    return LLMService()
