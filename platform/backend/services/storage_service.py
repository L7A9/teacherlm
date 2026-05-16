from __future__ import annotations

import asyncio
import io
import uuid
from datetime import timedelta
from pathlib import PurePosixPath

from minio import Minio
from minio.error import S3Error

from config import Settings, get_settings


class StorageService:
    """Async-friendly wrapper over the (sync) MinIO SDK.

    Object keys follow the convention:
      conversations/{conversation_id}/uploads/{uuid}_{filename}
      conversations/{conversation_id}/parsed/{uuid}.md
      conversations/{conversation_id}/artifacts/{uuid}_{filename}
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        # Pin region so presign skips the AWS-style location probe (which
        # fails when the sign client points at a host only reachable from
        # the browser, e.g. localhost:9000).
        region = "us-east-1"
        self._client = Minio(
            endpoint=self._settings.minio_endpoint,
            access_key=self._settings.minio_access_key,
            secret_key=self._settings.minio_secret_key,
            secure=self._settings.minio_secure,
            region=region,
        )
        public_endpoint = (
            self._settings.minio_public_endpoint or self._settings.minio_endpoint
        )
        if public_endpoint == self._settings.minio_endpoint:
            self._sign_client = self._client
        else:
            # Signature includes the Host header, so we must sign against the
            # hostname the browser will hit, not the in-network one.
            self._sign_client = Minio(
                endpoint=public_endpoint,
                access_key=self._settings.minio_access_key,
                secret_key=self._settings.minio_secret_key,
                secure=self._settings.minio_secure,
                region=region,
            )
        self._bucket = self._settings.minio_bucket

    # --- bucket lifecycle ---

    async def ensure_bucket(self) -> None:
        def _ensure() -> None:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)

        await asyncio.to_thread(_ensure)

    # --- key helpers ---

    @staticmethod
    def upload_key(conversation_id: uuid.UUID | str, filename: str) -> str:
        safe = PurePosixPath(filename).name
        return f"conversations/{conversation_id}/uploads/{uuid.uuid4()}_{safe}"

    @staticmethod
    def parsed_key(conversation_id: uuid.UUID | str, file_id: str) -> str:
        stem = PurePosixPath(file_id).stem or str(uuid.uuid4())
        return f"conversations/{conversation_id}/parsed/{stem}.md"

    @staticmethod
    def cleaned_text_key(conversation_id: uuid.UUID | str, file_id: str) -> str:
        stem = PurePosixPath(file_id).stem or str(uuid.uuid4())
        return f"conversations/{conversation_id}/cleaned/{stem}.md"

    @staticmethod
    def artifact_key(conversation_id: uuid.UUID | str, filename: str) -> str:
        safe = PurePosixPath(filename).name
        return f"conversations/{conversation_id}/artifacts/{uuid.uuid4()}_{safe}"

    # --- operations ---

    async def put_bytes(self, key: str, data: bytes, content_type: str = "application/octet-stream") -> str:
        def _put() -> None:
            self._client.put_object(
                bucket_name=self._bucket,
                object_name=key,
                data=io.BytesIO(data),
                length=len(data),
                content_type=content_type,
            )

        await asyncio.to_thread(_put)
        return key

    async def put_text(self, key: str, text: str, content_type: str = "text/markdown; charset=utf-8") -> str:
        return await self.put_bytes(key, text.encode("utf-8"), content_type=content_type)

    async def get_bytes(self, key: str) -> bytes:
        def _get() -> bytes:
            response = self._client.get_object(self._bucket, key)
            try:
                return response.read()
            finally:
                response.close()
                response.release_conn()

        return await asyncio.to_thread(_get)

    async def get_text(self, key: str) -> str:
        return (await self.get_bytes(key)).decode("utf-8")

    async def delete(self, key: str) -> None:
        def _delete() -> None:
            try:
                self._client.remove_object(self._bucket, key)
            except S3Error as exc:
                if exc.code != "NoSuchKey":
                    raise

        await asyncio.to_thread(_delete)

    async def presigned_get_url(
        self, key: str, expires_seconds: int | None = None
    ) -> str:
        ttl = expires_seconds or self._settings.artifact_url_ttl_s

        def _sign() -> str:
            return self._sign_client.presigned_get_object(
                self._bucket,
                key,
                expires=timedelta(seconds=ttl),
            )

        return await asyncio.to_thread(_sign)


_storage: StorageService | None = None


def get_storage() -> StorageService:
    global _storage
    if _storage is None:
        _storage = StorageService()
    return _storage
