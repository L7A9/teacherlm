from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
import unittest
import uuid

from routers.coursebuilder import generate_coursebuilder, rebuild_coursebuilder
from schemas.coursebuilder import CourseBuilderGenerateRequest, CourseBuilderRead


class _Session:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def get(self, model: object, pk: uuid.UUID) -> object:
        return object()

    async def commit(self) -> None:
        self.events.append("commit")


class _Service:
    def __init__(self, events: list[str]) -> None:
        self.events = events
        self.failed: list[str] = []
        self.queued_options: dict | None = None

    async def file_counts(self, session: object, conversation_id: uuid.UUID) -> tuple[int, int]:
        return 1, 0

    async def queue_course(
        self,
        session: object,
        conversation_id: uuid.UUID,
        *,
        llm_options: dict | None = None,
        restart_queued: bool = False,
    ) -> object:
        self.events.append("queue")
        self.queued_options = llm_options
        return SimpleNamespace(id=uuid.uuid4(), generation_metadata={"generation_id": "generation-1"})

    async def get_course(self, session: object, conversation_id: uuid.UUID) -> CourseBuilderRead:
        self.events.append("get")
        return CourseBuilderRead(conversation_id=conversation_id, total_file_count=1)

    async def mark_course_failed(
        self,
        session: object,
        conversation_id: uuid.UUID,
        message: str,
        *,
        course_id: uuid.UUID | None = None,
    ) -> None:
        self.events.append("failed")
        self.failed.append(message)


class CourseBuilderRouterTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_commits_queued_course_before_enqueueing_worker(self) -> None:
        events: list[str] = []
        conversation_id = uuid.uuid4()
        arq = SimpleNamespace(enqueue_job=AsyncMock(side_effect=lambda *args, **kwargs: events.append("enqueue")))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(arq_pool=arq)))

        with patch("routers.coursebuilder.get_coursebuilder_service", return_value=_Service(events)):
            await generate_coursebuilder(
                conversation_id,
                CourseBuilderGenerateRequest(options={"language": "fr"}),
                request,  # type: ignore[arg-type]
                _Session(events),  # type: ignore[arg-type]
            )

        self.assertEqual(events, ["queue", "commit", "enqueue", "get"])
        arq.enqueue_job.assert_awaited_once_with(
            "build_coursebuilder_course",
            str(conversation_id),
            {"language": "fr"},
            "generation-1",
            _job_id=f"coursebuilder:{conversation_id}:generation-1",
        )

    async def test_generate_uses_backend_resolved_llm_but_queues_sanitized_options(self) -> None:
        events: list[str] = []
        conversation_id = uuid.uuid4()
        arq = SimpleNamespace(enqueue_job=AsyncMock(side_effect=lambda *args, **kwargs: events.append("enqueue")))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(arq_pool=arq)))
        service = _Service(events)

        class RuntimeSettings:
            def sanitize_client_options(self, options: dict | None) -> dict:
                clean = dict(options or {})
                clean.pop("llm", None)
                return clean

            async def resolve_options(self, session: object, options: dict | None) -> dict:
                return {
                    **dict(options or {}),
                    "llm": {
                        "enabled": True,
                        "provider": "openai_compatible",
                        "model": "db-model",
                        "base_url": "https://db.example/v1",
                        "api_key": "db-secret",
                    },
                }

        with (
            patch("routers.coursebuilder.get_coursebuilder_service", return_value=service),
            patch("routers.coursebuilder.get_runtime_settings_service", return_value=RuntimeSettings()),
        ):
            await generate_coursebuilder(
                conversation_id,
                CourseBuilderGenerateRequest(
                    options={
                        "language": "fr",
                        "llm": {"enabled": True, "provider": "openai", "api_key": "client-secret"},
                    }
                ),
                request,  # type: ignore[arg-type]
                _Session(events),  # type: ignore[arg-type]
            )

        self.assertEqual(service.queued_options["llm"]["api_key"], "db-secret")  # type: ignore[index]
        arq.enqueue_job.assert_awaited_once_with(
            "build_coursebuilder_course",
            str(conversation_id),
            {"language": "fr"},
            "generation-1",
            _job_id=f"coursebuilder:{conversation_id}:generation-1",
        )

    async def test_rebuild_commits_queued_course_before_enqueueing_worker(self) -> None:
        events: list[str] = []
        conversation_id = uuid.uuid4()
        arq = SimpleNamespace(enqueue_job=AsyncMock(side_effect=lambda *args, **kwargs: events.append("enqueue")))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(arq_pool=arq)))

        with patch("routers.coursebuilder.get_coursebuilder_service", return_value=_Service(events)):
            await rebuild_coursebuilder(
                conversation_id,
                CourseBuilderGenerateRequest(options={"language": "en"}),
                request,  # type: ignore[arg-type]
                _Session(events),  # type: ignore[arg-type]
            )

        self.assertEqual(events, ["queue", "commit", "enqueue", "get"])
        arq.enqueue_job.assert_awaited_once_with(
            "build_coursebuilder_course",
            str(conversation_id),
            {"language": "en"},
            "generation-1",
            _job_id=f"coursebuilder:{conversation_id}:generation-1",
        )

    async def test_generate_marks_failed_when_enqueue_fails_after_commit(self) -> None:
        events: list[str] = []
        conversation_id = uuid.uuid4()
        async def fail_enqueue(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
            events.append("enqueue")
            raise RuntimeError("redis down")

        arq = SimpleNamespace(enqueue_job=AsyncMock(side_effect=fail_enqueue))
        request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(arq_pool=arq)))
        service = _Service(events)

        with patch("routers.coursebuilder.get_coursebuilder_service", return_value=service):
            with self.assertRaises(Exception) as caught:
                await generate_coursebuilder(
                    conversation_id,
                    CourseBuilderGenerateRequest(options={"language": "fr"}),
                    request,  # type: ignore[arg-type]
                    _Session(events),  # type: ignore[arg-type]
                )

        self.assertEqual(getattr(caught.exception, "status_code", None), 503)
        self.assertEqual(events, ["queue", "commit", "enqueue", "failed", "commit"])
        self.assertTrue(service.failed)


if __name__ == "__main__":
    unittest.main()
