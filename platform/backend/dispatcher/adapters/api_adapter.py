from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx
from httpx_sse import aconnect_sse

from teacherlm_core.schemas.generator_io import GeneratorInput, GeneratorOutput

from dispatcher.registry import GeneratorEntry


DEFAULT_TIMEOUT = httpx.Timeout(connect=5.0, read=600.0, write=30.0, pool=5.0)


@dataclass(slots=True)
class GeneratorEvent:
    """One SSE event from a generator.

    Known event names: "chunk", "sources", "artifact", "progress", "done", "error".
    """

    event: str
    data: Any


class GeneratorDispatchError(RuntimeError):
    pass


class ApiAdapter:
    """HTTP adapter for generators that expose a POST /run endpoint.

    - `dispatch` performs a one-shot JSON POST and parses GeneratorOutput.
    - `dispatch_stream` opens an SSE connection and yields parsed events.
    """

    def __init__(self, *, timeout: httpx.Timeout | None = None) -> None:
        self._timeout = timeout or DEFAULT_TIMEOUT

    async def dispatch(
        self,
        entry: GeneratorEntry,
        payload: GeneratorInput,
    ) -> GeneratorOutput:
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                response = await client.post(
                    entry.endpoint,
                    json=payload.model_dump(mode="json"),
                    headers={"Accept": "application/json"},
                )
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise GeneratorDispatchError(
                    f"{entry.id} dispatch failed: {exc}"
                ) from exc
            return GeneratorOutput.model_validate(response.json())

    async def dispatch_stream(
        self,
        entry: GeneratorEntry,
        payload: GeneratorInput,
    ) -> AsyncIterator[GeneratorEvent]:
        body = payload.model_dump(mode="json")
        headers = {"Accept": "text/event-stream", "Content-Type": "application/json"}

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            try:
                async with aconnect_sse(
                    client,
                    "POST",
                    entry.endpoint,
                    json=body,
                    headers=headers,
                ) as event_source:
                    async for sse in event_source.aiter_sse():
                        yield GeneratorEvent(
                            event=sse.event or "message",
                            data=_parse_sse_data(sse.data),
                        )
            except httpx.HTTPError as exc:
                raise GeneratorDispatchError(
                    f"{entry.id} stream failed: {exc}"
                ) from exc


def _parse_sse_data(raw: str) -> Any:
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw
