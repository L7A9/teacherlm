from __future__ import annotations

import asyncio
import uuid
import unittest
from unittest.mock import patch

from config import Settings
from routers.chat import _direct_reply_fallback
from services.interaction_router import (
    InteractionDecision,
    InteractionRouter,
    _dedupe_labels,
    _format_router_input,
)


class InteractionRouterTests(unittest.TestCase):
    def test_dedupe_labels_compacts_and_limits_visible_topics(self) -> None:
        labels = _dedupe_labels(
            ["  Lecture_04_V2.pdf  ", "lecture 04 V2.pdf", "SVD", "x", "Neural CF"],
            limit=3,
        )

        self.assertEqual(labels, ["Lecture 04 V2.pdf", "SVD", "Neural CF"])

    def test_router_input_contains_summary_history_and_student_message(self) -> None:
        prompt = _format_router_input(
            user_message="explain SVD equations",
            course_summary="Uploaded files: Lecture 04. Main visible topics: SVD, RNN, NCF.",
            chat_history=[{"role": "user", "content": f"old {i}"} for i in range(8)],
            learner_state={
                "understood_concepts": ["embeddings"],
                "struggling_concepts": ["matrix factorization"],
            },
        )

        self.assertIn("SVD, RNN, NCF", prompt)
        self.assertIn("Student message:\nexplain SVD equations", prompt)
        self.assertIn("struggling=[matrix factorization]", prompt)
        self.assertNotIn("old 0", prompt)
        self.assertIn("old 7", prompt)

    def test_client_uses_enabled_llm_override_from_settings_page(self) -> None:
        router = InteractionRouter(
            settings=Settings(
                ollama_host="http://local-ollama:11434",
                ollama_chat_model="local-model",
            )
        )

        client = router._client(
            {
                "llm": {
                    "enabled": True,
                    "provider": "openai_compatible",
                    "base_url": "https://example.test/v1",
                    "model": "settings-model",
                    "api_key": "secret",
                }
            }
        )

        self.assertEqual(client.provider, "openai_compatible")
        self.assertEqual(client.base_url, "https://example.test/v1")
        self.assertEqual(client.model, "settings-model")
        self.assertEqual(client.api_key, "secret")

    def test_route_falls_back_to_retrieve_when_llm_router_fails(self) -> None:
        class FailingClient:
            async def chat_structured(self, **_kwargs):  # noqa: ANN202
                raise RuntimeError("boom")

        router = InteractionRouter(settings=Settings())
        router._client = lambda _options: FailingClient()  # type: ignore[method-assign]

        async def run() -> InteractionDecision:
            with patch(
                "services.interaction_router.build_course_summary",
                return_value="Uploaded files: Lecture 04. Main visible topics: SVD.",
            ):
                return await router.route(
                    conversation_id=uuid.uuid4(),
                    user_message="explain SVD",
                    chat_history=[],
                    learner_state={},
                    options={},
                )

        decision = asyncio.run(run())

        self.assertEqual(decision.action, "retrieve")
        self.assertEqual(decision.retrieval_query, "explain SVD")

    def test_direct_reply_fallbacks_are_source_safe(self) -> None:
        self.assertIn("outside the uploaded course files", _direct_reply_fallback("outside_files"))
        self.assertIn("course materials", _direct_reply_fallback("conversational_reply"))

