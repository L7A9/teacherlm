from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, SecretStr


ProviderType = Literal[
    "ollama",
    "openai",
    "anthropic",
    "openai_compatible",
    "anthropic_compatible",
]


class LLMMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class LLMProviderConfig(BaseModel):
    provider_id: str
    display_name: str
    provider_type: ProviderType = "ollama"
    base_url: str = "http://localhost:11434"
    model_name: str = "llama3.2"
    api_key: SecretStr | None = None
    timeout_s: float = 60.0
    metadata: dict = Field(default_factory=dict)


class LLMProviderError(RuntimeError):
    pass


async def complete_text(
    provider: LLMProviderConfig,
    messages: list[LLMMessage],
    *,
    json_schema: dict | None = None,
    temperature: float = 0.2,
) -> str:
    """Route a text completion through a configured provider.

    Ollama uses its native `format=` argument for structured output. External
    providers are optional and only used after the student configures keys.
    """

    match provider.provider_type:
        case "ollama":
            return await _complete_ollama(provider, messages, json_schema, temperature)
        case "openai" | "openai_compatible":
            return await _complete_openai(provider, messages, json_schema, temperature)
        case "anthropic" | "anthropic_compatible":
            return await _complete_anthropic(provider, messages, json_schema, temperature)
        case other:
            raise LLMProviderError(f"unsupported provider type: {other}")


async def _complete_ollama(
    provider: LLMProviderConfig,
    messages: list[LLMMessage],
    json_schema: dict | None,
    temperature: float,
) -> str:
    try:
        from ollama import AsyncClient
    except Exception as exc:  # noqa: BLE001
        raise LLMProviderError("ollama package is not installed") from exc

    client = AsyncClient(host=provider.base_url)
    payload = [message.model_dump() for message in messages]
    response = await client.chat(
        model=provider.model_name,
        messages=payload,
        options={"temperature": temperature},
        format=json_schema if json_schema is not None else None,
    )
    content = response.get("message", {}).get("content", "")
    if not content:
        raise LLMProviderError("Ollama returned an empty response")
    return str(content)


async def _complete_openai(
    provider: LLMProviderConfig,
    messages: list[LLMMessage],
    json_schema: dict | None,
    temperature: float,
) -> str:
    api_key = provider.api_key.get_secret_value() if provider.api_key else None
    if not api_key:
        raise LLMProviderError("OpenAI-compatible provider requires an API key")
    try:
        from openai import AsyncOpenAI
    except Exception as exc:  # noqa: BLE001
        raise LLMProviderError("openai package is not installed") from exc

    client = AsyncOpenAI(api_key=api_key, base_url=provider.base_url)
    kwargs: dict = {}
    if json_schema is not None:
        kwargs["response_format"] = {"type": "json_object"}
    response = await client.chat.completions.create(
        model=provider.model_name,
        messages=[message.model_dump() for message in messages],
        temperature=temperature,
        **kwargs,
    )
    content = response.choices[0].message.content or ""
    if not content:
        raise LLMProviderError("OpenAI-compatible provider returned an empty response")
    return content


async def _complete_anthropic(
    provider: LLMProviderConfig,
    messages: list[LLMMessage],
    json_schema: dict | None,
    temperature: float,
) -> str:
    api_key = provider.api_key.get_secret_value() if provider.api_key else None
    if not api_key:
        raise LLMProviderError("Anthropic-compatible provider requires an API key")
    try:
        from anthropic import AsyncAnthropic
    except Exception as exc:  # noqa: BLE001
        raise LLMProviderError("anthropic package is not installed") from exc

    system_parts = [message.content for message in messages if message.role == "system"]
    user_messages = [
        {"role": message.role, "content": message.content}
        for message in messages
        if message.role != "system"
    ]
    if json_schema is not None:
        user_messages.append(
            {
                "role": "user",
                "content": "Return a valid JSON object matching the requested schema.",
            }
        )

    client = AsyncAnthropic(api_key=api_key, base_url=provider.base_url)
    response = await client.messages.create(
        model=provider.model_name,
        max_tokens=2048,
        temperature=temperature,
        system="\n\n".join(system_parts) or None,
        messages=user_messages,
    )
    text = "\n".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )
    if not text:
        raise LLMProviderError("Anthropic-compatible provider returned an empty response")
    return text

