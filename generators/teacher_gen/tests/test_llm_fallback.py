from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path


TEACHER_GEN_DIR = Path(__file__).resolve().parents[1]
CORE_DIR = Path(__file__).resolve().parents[3] / "packages" / "teacherlm_core"
for path in (TEACHER_GEN_DIR.parent, CORE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from teacher_gen.services.llm_service import LLMService  # noqa: E402


class _FailingCloudClient:
    provider = "openai_compatible"
    model = "cloud-model"

    async def stream_chat(self, **_kwargs):  # noqa: ANN202
        raise RuntimeError(
            '429 from openai_compatible provider: {"type":"rate_limited","code":"1300"}'
        )
        yield ""


class _LocalClient:
    provider = "ollama"
    model = "local-model"

    async def stream_chat(self, **_kwargs):  # noqa: ANN202
        yield "local answer"


class TeacherGenLlmFallbackTests(unittest.TestCase):
    def test_rate_limited_cloud_provider_streams_local_fallback_notice(self) -> None:
        service = LLMService()
        service.chat = _FailingCloudClient()
        service.fallback_chat = _LocalClient()

        async def collect() -> str:
            parts = []
            async for delta in service.stream_reply(
                system="system",
                chat_history=[],
                user_message="question",
            ):
                parts.append(delta)
            return "".join(parts)

        response = asyncio.run(collect())

        self.assertTrue(service.last_chat_used_fallback)
        self.assertIn("local fallback LLM", response)
        self.assertIn("local answer", response)


if __name__ == "__main__":
    unittest.main()
