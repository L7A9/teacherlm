import json
from collections.abc import AsyncIterator
from typing import Any


async def safe_sse_stream(source: AsyncIterator[str]) -> AsyncIterator[str]:
    """Forward preformatted SSE frames and turn exceptions into SSE errors."""

    try:
        async for frame in source:
            yield frame
    except Exception as exc:  # noqa: BLE001 - boundary converts to client error
        yield format_sse({"message": _friendly_error(exc)}, event="error")


async def stream_as_sse(
    async_iter: AsyncIterator[str | dict[str, Any]],
    event: str | None = None,
) -> AsyncIterator[str]:
    """Wrap an async iterator of text deltas (or dict payloads) as SSE frames.

    - `str` items are sent as JSON-encoded `{"delta": <text>}` payloads so
      clients can parse them uniformly.
    - `dict` items are sent as JSON-encoded payloads verbatim.
    - Emits a terminal `event: done` frame when the source iterator finishes.
    - If the source raises, emits a final `event: error` frame with the
      message, then re-raises.
    """
    try:
        async for item in async_iter:
            payload = (
                json.dumps({"delta": item}, ensure_ascii=False)
                if isinstance(item, str)
                else json.dumps(item, ensure_ascii=False)
            )
            yield _format_sse(payload, event=event)
    except Exception as exc:
        err_payload = json.dumps({"error": str(exc)}, ensure_ascii=False)
        yield _format_sse(err_payload, event="error")
    else:
        yield _format_sse("{}", event="done")


def format_sse(data: dict[str, Any] | str, event: str | None = None) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return _format_sse(payload, event=event)


def _format_sse(data: str, event: str | None = None) -> str:
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    for line in data.splitlines() or [""]:
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"


def _friendly_error(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return "The model provider closed the stream before returning a response."
    return message
