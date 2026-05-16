from __future__ import annotations

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_DIR.parents[1]
for path in (BACKEND_DIR, REPO_ROOT / "packages" / "teacherlm_core"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


async def _load_files(conversation_id: str | None) -> list[Any]:
    from sqlalchemy import select

    from db.models import UploadedFile
    from db.session import session_scope

    async with session_scope() as session:
        stmt = select(UploadedFile).where(UploadedFile.parsed_markdown_path.is_not(None))
        if conversation_id:
            stmt = stmt.where(UploadedFile.conversation_id == uuid.UUID(conversation_id))
        result = await session.execute(stmt.order_by(UploadedFile.created_at.asc()))
        return list(result.scalars().all())


async def _update_file(file_pk: uuid.UUID, *, status: str, chunk_count: int | None = None, error: str | None = None) -> None:
    from db.models import UploadedFile
    from db.session import session_scope

    async with session_scope() as session:
        record = await session.get(UploadedFile, file_pk)
        if record is None:
            return
        record.status = status
        if chunk_count is not None:
            record.chunk_count = chunk_count
        record.error = error


async def _reindex_file(record: Any) -> dict[str, Any]:
    from services.chunking_service import get_chunker
    from services.course_content_store import get_course_content_store
    from services.course_structure_service import get_course_structure_extractor
    from services.document_cleaning_service import get_document_cleaner
    from db.session import session_scope
    from services.storage_service import get_storage
    from services.vector_service import get_vector_service

    storage = get_storage()
    cleaner = get_document_cleaner()
    extractor = get_course_structure_extractor()
    chunker = get_chunker()
    content_store = get_course_content_store()
    vectors = get_vector_service()

    await _update_file(record.id, status="chunking")
    markdown = await storage.get_text(str(record.parsed_markdown_path))
    cleaned, stats = cleaner.clean_markdown_with_stats(markdown)
    cleaned_key = storage.cleaned_text_key(record.conversation_id, record.file_id)
    await storage.put_text(cleaned_key, cleaned)
    document = extractor.extract(
        cleaned,
        conversation_id=record.conversation_id,
        source_file_id=record.file_id,
        source_filename=record.filename,
    )
    chunks = chunker.chunk_course_document(document, source_file_id=record.file_id)

    async with session_scope() as session:
        await content_store.replace_document(
            session,
            conversation_id=record.conversation_id,
            uploaded_file_id=record.id,
            source_file_id=record.file_id,
            source_filename=record.filename,
            raw_markdown_path=record.parsed_markdown_path,
            cleaned_text_path=cleaned_key,
            cleaned_text=cleaned,
            document=document,
            chunks=chunks,
        )

    await _update_file(record.id, status="embedding")
    await vectors.delete_by_file(record.conversation_id, record.file_id)
    if chunks:
        await vectors.ensure_collection(record.conversation_id)
        upserted = await vectors.upsert_chunks(
            record.conversation_id,
            chunks,
            file_id=record.file_id,
        )
    else:
        upserted = 0

    await _update_file(record.id, status="ready", chunk_count=upserted)
    return {
        "file_id": str(record.id),
        "filename": record.filename,
        "chunks": upserted,
        "sections": len(document.sections),
        "cleaning": {
            "original_lines": stats.original_lines,
            "kept_lines": stats.kept_lines,
            "removed_lines": stats.removed_lines,
        },
    }


async def _amain() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild Qdrant chunks from stored parsed markdown."
    )
    parser.add_argument(
        "conversation_id",
        nargs="?",
        help="Conversation UUID to reindex. Omit with --all to reindex every parsed file.",
    )
    parser.add_argument("--all", action="store_true", help="Reindex all files with parsed markdown.")
    args = parser.parse_args()

    if not args.conversation_id and not args.all:
        parser.error("provide conversation_id or --all")

    files = await _load_files(args.conversation_id)
    results = []
    for record in files:
        try:
            results.append(await _reindex_file(record))
        except Exception as exc:  # noqa: BLE001
            await _update_file(record.id, status="failed", error=f"{type(exc).__name__}: {exc}")
            results.append(
                {
                    "file_id": str(record.id),
                    "filename": record.filename,
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    from services.vector_service import get_vector_service

    await get_vector_service().aclose()
    print(
        json.dumps(
            {
                "conversation_id": args.conversation_id,
                "file_count": len(files),
                "total_chunks": sum(int(item.get("chunks", 0)) for item in results),
                "files": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    asyncio.run(_amain())
