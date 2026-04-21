from __future__ import annotations

import asyncio
import io
import uuid
from datetime import timedelta
from functools import lru_cache
from pathlib import PurePosixPath

from minio import Minio

from ..config import get_settings


class ArtifactStore:
    """Small async wrapper around MinIO for uploading quiz JSON artifacts.

    Mirrors the platform's storage_service key conventions so artifacts land
    next to platform-uploaded files in the same bucket:
      conversations/{conversation_id}/artifacts/{uuid}_{filename}
    """

    def __init__(self) -> None:
        s = get_settings()
        self._settings = s
        # Pin region so presign never probes the endpoint for `_get_region`:
        # the sign client targets a hostname only reachable from the browser
        # (e.g. localhost:9000), not from inside the container.
        region = "us-east-1"
        self._client = Minio(
            endpoint=s.minio_endpoint,
            access_key=s.minio_access_key,
            secret_key=s.minio_secret_key,
            secure=s.minio_secure,
            region=region,
        )
        # Separate client for presigning when the browser-facing host differs
        # from the in-network host. Signature includes the Host header, so we
        # must sign against the hostname the browser will hit.
        public_endpoint = s.minio_public_endpoint or s.minio_endpoint
        if public_endpoint == s.minio_endpoint:
            self._sign_client = self._client
        else:
            self._sign_client = Minio(
                endpoint=public_endpoint,
                access_key=s.minio_access_key,
                secret_key=s.minio_secret_key,
                secure=s.minio_secure,
                region=region,
            )
        self._bucket = s.minio_bucket

    async def ensure_bucket(self) -> None:
        def _ensure() -> None:
            if not self._client.bucket_exists(self._bucket):
                self._client.make_bucket(self._bucket)

        await asyncio.to_thread(_ensure)

    @staticmethod
    def artifact_key(conversation_id: str, filename: str) -> str:
        safe = PurePosixPath(filename).name
        return f"conversations/{conversation_id}/artifacts/{uuid.uuid4()}_{safe}"

    async def put_bytes(
        self,
        key: str,
        data: bytes,
        content_type: str = "application/octet-stream",
    ) -> None:
        def _put() -> None:
            self._client.put_object(
                bucket_name=self._bucket,
                object_name=key,
                data=io.BytesIO(data),
                length=len(data),
                content_type=content_type,
            )

        await asyncio.to_thread(_put)

    async def presigned_get_url(self, key: str) -> str:
        def _sign() -> str:
            return self._sign_client.presigned_get_object(
                self._bucket,
                key,
                expires=timedelta(seconds=self._settings.artifact_url_ttl_s),
            )

        return await asyncio.to_thread(_sign)

    async def save_json(
        self,
        *,
        conversation_id: str,
        filename: str,
        payload: bytes,
    ) -> tuple[str, str]:
        """Upload JSON bytes and return (key, presigned_url)."""
        await self.ensure_bucket()
        key = self.artifact_key(conversation_id, filename)
        await self.put_bytes(key, payload, content_type="application/json")
        url = await self.presigned_get_url(key)
        return key, url


@lru_cache
def get_artifact_store() -> ArtifactStore:
    return ArtifactStore()
