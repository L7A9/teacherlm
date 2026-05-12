from collections.abc import AsyncIterator
import json
from typing import Any

import httpx
from ollama import AsyncClient
from pydantic import BaseModel

from .language import inject_language_directive


class OllamaClient:
    """Async chat wrapper for Ollama plus cloud providers.

    The class name stays `OllamaClient` for backwards compatibility with the
    generators, but it can also call OpenAI-compatible APIs and Anthropic.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        provider: str = "ollama",
        api_key: str | None = None,
    ) -> None:
        self.base_url = base_url
        self.model = model
        self.provider = self._normalize_provider(provider)
        self.api_key = api_key
        self._client = AsyncClient(host=base_url) if self.provider == "ollama" else None

    async def chat(
        self,
        messages: list[dict],
        stream: bool = False,
        format: str | dict | None = None,
        options: dict | None = None,
    ) -> Any:
        if self.provider == "ollama":
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": inject_language_directive(messages),
                "stream": stream,
            }
            if format is not None:
                kwargs["format"] = format
            if options is not None:
                kwargs["options"] = options
            return await self._client.chat(**kwargs)  # type: ignore[union-attr]

        if self.provider == "anthropic":
            return await self._anthropic_chat(
                messages=messages,
                stream=stream,
                format=format,
                options=options,
            )

        payload = self._openai_payload(
            messages=messages,
            stream=stream,
            format=format,
            options=options,
        )
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                self._chat_completions_url(),
                headers=self._openai_headers(),
                json=payload,
            )
            await self._raise_for_status(response)
            data = response.json()
        content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {"message": {"content": content}}

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
        content = _coerce_json_content(response["message"]["content"])
        return schema.model_validate_json(content)

    async def stream_chat(
        self,
        messages: list[dict],
        options: dict | None = None,
    ) -> AsyncIterator[str]:
        if self.provider == "ollama":
            kwargs: dict[str, Any] = {
                "model": self.model,
                "messages": inject_language_directive(messages),
                "stream": True,
            }
            if options is not None:
                kwargs["options"] = options
            async for part in await self._client.chat(**kwargs):  # type: ignore[union-attr]
                delta = part.get("message", {}).get("content", "")
                if delta:
                    yield delta
            return

        if self.provider == "anthropic":
            async for delta in self._anthropic_stream_chat(
                messages=messages,
                options=options,
            ):
                yield delta
            return

        payload = self._openai_payload(
            messages=messages,
            stream=True,
            format=None,
            options=options,
        )
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                self._chat_completions_url(),
                headers=self._openai_headers(),
                json=payload,
            ) as response:
                await self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    chunk = line[6:].strip()
                    if chunk == "[DONE]":
                        break
                    try:
                        data = json.loads(chunk)
                    except ValueError:
                        continue
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta

    async def _anthropic_chat(
        self,
        *,
        messages: list[dict],
        stream: bool,
        format: str | dict | None,
        options: dict | None,
    ) -> dict[str, dict[str, str]]:
        payload = self._anthropic_payload(
            messages=messages,
            stream=stream,
            format=format,
            options=options,
        )
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                self._anthropic_messages_url(),
                headers=self._anthropic_headers(),
                json=payload,
            )
            await self._raise_for_status(response)
            data = response.json()
        return {"message": {"content": _anthropic_text(data)}}

    async def _anthropic_stream_chat(
        self,
        *,
        messages: list[dict],
        options: dict | None,
    ) -> AsyncIterator[str]:
        payload = self._anthropic_payload(
            messages=messages,
            stream=True,
            format=None,
            options=options,
        )
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                self._anthropic_messages_url(),
                headers=self._anthropic_headers(),
                json=payload,
            ) as response:
                await self._raise_for_status(response)
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    raw = line[6:].strip()
                    if not raw or raw == "[DONE]":
                        continue
                    try:
                        data = json.loads(raw)
                    except ValueError:
                        continue
                    if data.get("type") == "error":
                        error = data.get("error") or {}
                        raise RuntimeError(error.get("message") or str(error))
                    if data.get("type") != "content_block_delta":
                        continue
                    delta = data.get("delta") or {}
                    text = delta.get("text")
                    if isinstance(text, str) and text:
                        yield text

    def _openai_payload(
        self,
        *,
        messages: list[dict],
        stream: bool,
        format: str | dict | None,
        options: dict | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": inject_language_directive(messages),
            "stream": stream,
        }
        if options:
            for source, target in {
                "temperature": "temperature",
                "top_p": "top_p",
                "num_predict": "max_tokens",
                "max_tokens": "max_tokens",
            }.items():
                if source in options:
                    payload[target] = options[source]
        if isinstance(format, dict):
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "teacherlm_structured_output",
                    "schema": format,
                    "strict": False,
                },
            }
        elif format == "json":
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _anthropic_payload(
        self,
        *,
        messages: list[dict],
        stream: bool,
        format: str | dict | None,
        options: dict | None,
    ) -> dict[str, Any]:
        system, chat_messages = _anthropic_messages(inject_language_directive(messages))
        if isinstance(format, dict):
            system = (
                f"{system}\n\nReturn ONLY valid JSON conforming to this JSON "
                f"schema. Do not use Markdown fences or prose:\n"
                f"{json.dumps(format, ensure_ascii=False)}"
            ).strip()
        elif format == "json":
            system = (
                f"{system}\n\nReturn ONLY valid JSON. Do not use Markdown "
                "fences or prose."
            ).strip()

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": chat_messages,
            "max_tokens": _max_tokens(options),
            "stream": stream,
        }
        if system:
            payload["system"] = system
        if options:
            if "temperature" in options:
                payload["temperature"] = options["temperature"]
            if "top_p" in options:
                payload["top_p"] = options["top_p"]
        return payload

    def _chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/chat/completions"):
            return base
        return f"{base}/chat/completions"

    def _anthropic_messages_url(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1/messages") or base.endswith("/messages"):
            return base
        return f"{base}/v1/messages"

    def _openai_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _anthropic_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01",
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    async def _raise_for_status(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text.strip()
            raise RuntimeError(
                f"{response.status_code} from {self.provider} provider: {detail}"
            ) from exc

    @staticmethod
    def _normalize_provider(value: str) -> str:
        provider = value.lower().strip().replace("-", "_")
        if provider in {"openai_compatible", "openai_compat", "openai_compat_api"}:
            return "openai_compatible"
        if provider in {"anthropic", "claude"}:
            return "anthropic"
        return provider


def _anthropic_messages(messages: list[dict]) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    chat_messages: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role") or "user")
        content = str(msg.get("content") or "")
        if role == "system":
            system_parts.append(content)
        elif role in {"user", "assistant"}:
            if chat_messages and chat_messages[-1]["role"] == role:
                chat_messages[-1]["content"] += f"\n\n{content}"
            else:
                chat_messages.append({"role": role, "content": content})
    if not chat_messages:
        chat_messages.append({"role": "user", "content": ""})
    return "\n\n".join(system_parts), chat_messages


def _anthropic_text(data: dict[str, Any]) -> str:
    parts: list[str] = []
    for block in data.get("content") or []:
        if isinstance(block, dict) and block.get("type") == "text":
            text = block.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


def _max_tokens(options: dict | None) -> int:
    if not options:
        return 2048
    raw = options.get("max_tokens", options.get("num_predict", 2048))
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return 2048


def _coerce_json_content(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    starts = [pos for pos in (text.find("{"), text.find("[")) if pos >= 0]
    ends = [pos for pos in (text.rfind("}"), text.rfind("]")) if pos >= 0]
    if starts and ends:
        start = min(starts)
        end = max(ends)
        if end > start:
            return text[start : end + 1]
    return text
