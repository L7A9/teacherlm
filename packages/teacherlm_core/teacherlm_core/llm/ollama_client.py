from collections.abc import AsyncIterator
from typing import Any

from ollama import AsyncClient
from pydantic import BaseModel


class OllamaClient:
    """Async wrapper around ollama.AsyncClient with Pydantic-aware helpers."""

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url
        self.model = model
        self._client = AsyncClient(host=base_url)

    async def chat(
        self,
        messages: list[dict],
        stream: bool = False,
        format: str | dict | None = None,
        options: dict | None = None,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
        }
        if format is not None:
            kwargs["format"] = format
        if options is not None:
            kwargs["options"] = options
        return await self._client.chat(**kwargs)

    async def chat_structured[T: BaseModel](
        self,
        messages: list[dict],
        schema: type[T],
        options: dict | None = None,
    ) -> T:
        response = await self.chat(
            messages=messages,
            stream=False,
            format=schema.model_json_schema(),
            options=options,
        )
        content = response["message"]["content"]
        return schema.model_validate_json(content)

    async def stream_chat(
        self,
        messages: list[dict],
        options: dict | None = None,
    ) -> AsyncIterator[str]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if options is not None:
            kwargs["options"] = options
        async for part in await self._client.chat(**kwargs):
            delta = part.get("message", {}).get("content", "")
            if delta:
                yield delta
