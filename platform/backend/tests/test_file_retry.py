from __future__ import annotations

import uuid
import unittest
import sys
import types
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from db.models import UploadedFile
from routers.files import _sync_coursebuilder_after_file_delete, retry_file
from schemas.file import FileRetryRequest


class _ScalarResult:
    def __init__(self, value: object | None) -> None:
        self._value = value

    def scalar_one_or_none(self) -> object | None:
        return self._value


class _ListResult:
    def __init__(self, values: list[object]) -> None:
        self._values = values

    def scalars(self) -> "_ListResult":
        return self

    def all(self) -> list[object]:
        return self._values


class _Document:
    cleaned_text_path = "cleaned/file.md"


class _Session:
    def __init__(self, record: UploadedFile, document: object | None = None) -> None:
        self.record = record
        self.document = document
        self.deleted: list[object] = []
        self.flushed = False
        self.committed = False
        self.refreshed: UploadedFile | None = None

    async def get(self, model: object, pk: uuid.UUID) -> UploadedFile | None:
        return self.record if pk == self.record.id else None

    async def execute(self, statement: object) -> _ScalarResult:
        return _ScalarResult(self.document)

    async def delete(self, item: object) -> None:
        self.deleted.append(item)

    async def flush(self) -> None:
        self.flushed = True

    async def refresh(self, record: UploadedFile) -> None:
        self.refreshed = record

    async def commit(self) -> None:
        self.committed = True


class _Storage:
    def __init__(self) -> None:
        self.deleted: list[str] = []

    async def delete(self, key: str) -> None:
        self.deleted.append(key)


class _Vectors:
    def __init__(self) -> None:
        self.deleted: list[tuple[uuid.UUID, str]] = []

    async def delete_by_file(self, conversation_id: uuid.UUID, file_id: str) -> None:
        self.deleted.append((conversation_id, file_id))


class _CourseBuilderSyncSession:
    def __init__(self, files: list[UploadedFile]) -> None:
        self.files = files
        self.commits = 0

    async def execute(self, statement: object) -> _ListResult:
        return _ListResult(self.files)

    async def commit(self) -> None:
        self.commits += 1


class _CourseBuilderService:
    def __init__(self) -> None:
        self.cleared: list[uuid.UUID] = []
        self.queued: list[tuple[uuid.UUID, dict | None]] = []
        self.failed: list[str] = []

    async def clear_course(self, session: object, conversation_id: uuid.UUID) -> None:
        self.cleared.append(conversation_id)

    async def queue_course(
        self,
        session: object,
        conversation_id: uuid.UUID,
        *,
        llm_options: dict | None = None,
        restart_queued: bool = False,
    ) -> object:
        self.queued.append((conversation_id, llm_options))
        return types.SimpleNamespace(
            id=uuid.uuid4(),
            generation_metadata={"generation_id": "generation-1"},
        )

    async def generate_course(self, *args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return None

    async def mark_course_failed(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.failed.append(str(kwargs.get("message") or args[2] if len(args) > 2 else "failed"))


class FileRetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_retry_failed_file_resets_artifacts_and_requeues_ingestion(self) -> None:
        conversation_id = uuid.uuid4()
        record = UploadedFile(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            filename="lecture.pdf",
            file_id="uploads/lecture.pdf",
            status="failed",
            chunk_count=12,
            parsed_markdown_path="parsed/file.md",
            error="ParserError: broken",
        )
        document = _Document()
        session = _Session(record, document)
        storage = _Storage()
        vectors = _Vectors()
        arq = type("Arq", (), {"enqueue_job": AsyncMock()})()
        vector_module = types.ModuleType("services.vector_service")
        vector_module.get_vector_service = lambda: vectors

        with (
            patch("routers.files.get_storage", return_value=storage),
            patch.dict(sys.modules, {"services.vector_service": vector_module}),
        ):
            response = await retry_file(
                conversation_id,
                record.id,
                FileRetryRequest(options={"language": "fr"}),
                session,  # type: ignore[arg-type]
                arq,
            )

        self.assertIs(response, record)
        self.assertEqual(record.status, "uploaded")
        self.assertIsNone(record.error)
        self.assertEqual(record.chunk_count, 0)
        self.assertIsNone(record.parsed_markdown_path)
        self.assertEqual(storage.deleted, ["parsed/file.md", "cleaned/file.md"])
        self.assertEqual(vectors.deleted, [(conversation_id, "uploads/lecture.pdf")])
        self.assertEqual(session.deleted, [document])
        self.assertTrue(session.flushed)
        self.assertTrue(session.committed)
        self.assertIs(session.refreshed, record)
        arq.enqueue_job.assert_awaited_once_with(
            "ingest_file",
            str(record.id),
            {"language": "fr"},
        )

    async def test_retry_rejects_non_failed_file(self) -> None:
        conversation_id = uuid.uuid4()
        record = UploadedFile(
            id=uuid.uuid4(),
            conversation_id=conversation_id,
            filename="lecture.pdf",
            file_id="uploads/lecture.pdf",
            status="ready",
        )
        session = _Session(record)
        arq = type("Arq", (), {"enqueue_job": AsyncMock()})()

        with self.assertRaises(HTTPException) as caught:
            await retry_file(
                conversation_id,
                record.id,
                None,
                session,  # type: ignore[arg-type]
                arq,
            )

        self.assertEqual(caught.exception.status_code, 409)
        arq.enqueue_job.assert_not_awaited()

    async def test_delete_sync_clears_coursebuilder_when_no_files_remain(self) -> None:
        conversation_id = uuid.uuid4()
        service = _CourseBuilderService()
        session = _CourseBuilderSyncSession([])
        request = types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace(arq_pool=None)))

        with patch("routers.files.get_coursebuilder_service", return_value=service):
            await _sync_coursebuilder_after_file_delete(
                session,  # type: ignore[arg-type]
                conversation_id,
                request,  # type: ignore[arg-type]
                llm_options={"language": "fr"},
            )

        self.assertEqual(service.cleared, [conversation_id])
        self.assertEqual(service.queued, [])

    async def test_delete_sync_requeues_coursebuilder_when_remaining_files_are_ready(self) -> None:
        conversation_id = uuid.uuid4()
        service = _CourseBuilderService()
        session = _CourseBuilderSyncSession(
            [
                UploadedFile(
                    id=uuid.uuid4(),
                    conversation_id=conversation_id,
                    filename="remaining.pdf",
                    file_id="uploads/remaining.pdf",
                    status="ready",
                )
            ]
        )
        arq = types.SimpleNamespace(enqueue_job=AsyncMock())
        request = types.SimpleNamespace(app=types.SimpleNamespace(state=types.SimpleNamespace(arq_pool=arq)))

        with patch("routers.files.get_coursebuilder_service", return_value=service):
            await _sync_coursebuilder_after_file_delete(
                session,  # type: ignore[arg-type]
                conversation_id,
                request,  # type: ignore[arg-type]
                llm_options={"language": "fr"},
            )

        self.assertEqual(service.queued, [(conversation_id, {"language": "fr"})])
        self.assertEqual(session.commits, 1)
        arq.enqueue_job.assert_awaited_once_with(
            "build_coursebuilder_course",
            str(conversation_id),
            {"language": "fr"},
            "generation-1",
            _job_id=f"coursebuilder:{conversation_id}:generation-1",
        )


if __name__ == "__main__":
    unittest.main()
