import asyncio

import httpx
import pytest

from teacherlm_core.llm.ollama_client import OllamaClient
from teacherlm_core.llm.runtime import build_llm_client_kwargs


def test_llm_override_selects_openai_compatible_provider() -> None:
    cfg = build_llm_client_kwargs(
        default_base_url="http://localhost:11434",
        default_model="local-model",
        options={
            "enabled": True,
            "provider": "openai_compatible",
            "model": "mistral-large-latest",
            "base_url": "https://api.mistral.ai/v1",
            "api_key": "secret",
        },
    )

    assert cfg == {
        "provider": "openai_compatible",
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-large-latest",
        "api_key": "secret",
    }


def test_llm_override_disabled_keeps_ollama_default() -> None:
    cfg = build_llm_client_kwargs(
        default_base_url="http://localhost:11434",
        default_model="local-model",
        options={"enabled": False, "provider": "openai", "model": "gpt-4.1-mini"},
    )

    assert cfg["provider"] == "ollama"
    assert cfg["base_url"] == "http://localhost:11434"
    assert cfg["model"] == "local-model"


def test_streaming_status_error_reads_body_before_text() -> None:
    client = OllamaClient(
        base_url="https://api.example.test/v1",
        model="test-model",
        provider="openai_compatible",
    )
    response = httpx.Response(
        status_code=400,
        request=httpx.Request("POST", "https://api.example.test/v1/chat/completions"),
        stream=httpx.ByteStream(b'{"error":"bad request"}'),
    )

    async def run() -> None:
        await client._raise_for_status(response)

    with pytest.raises(RuntimeError) as caught:
        asyncio.run(run())

    assert "400 from openai_compatible provider" in str(caught.value)
    assert "bad request" in str(caught.value)
