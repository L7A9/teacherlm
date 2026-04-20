import json
from collections.abc import AsyncIterator
from typing import Any


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
        raise
    else:
        yield _format_sse("{}", event="done")


def _format_sse(data: str, event: str | None = None) -> str:
    lines: list[str] = []
    if event:
        lines.append(f"event: {event}")
    for line in data.splitlines() or [""]:
        lines.append(f"data: {line}")
    return "\n".join(lines) + "\n\n"
