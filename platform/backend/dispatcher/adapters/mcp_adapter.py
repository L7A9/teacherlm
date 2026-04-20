from __future__ import annotations

from collections.abc import AsyncIterator

from teacherlm_core.schemas.generator_io import GeneratorInput, GeneratorOutput

from dispatcher.adapters.api_adapter import GeneratorEvent
from dispatcher.registry import GeneratorEntry


class McpAdapter:
    """MCP (Model Context Protocol) adapter — placeholder for plugin-style generators.

    The surface mirrors ApiAdapter so the router can pick either transport by
    looking at `GeneratorEntry.type`. Wire a real MCP client here (e.g. the
    official Python SDK's `mcp.client.sse`/`mcp.client.stdio`) when MCP-based
    generators come online.

    Expected call shape, once implemented:
      - Open an MCP session to `entry.endpoint`.
      - Invoke the `run` tool with arguments = GeneratorInput.model_dump().
      - Translate tool responses / notifications into GeneratorEvent items for
        the streaming path, and aggregate into GeneratorOutput for one-shot.
    """

    async def dispatch(
        self,
        entry: GeneratorEntry,
        payload: GeneratorInput,
    ) -> GeneratorOutput:
        raise NotImplementedError(
            f"McpAdapter.dispatch is not wired yet (generator {entry.id!r}). "
            "Add an MCP client dependency and implement the session/tool call."
        )

    async def dispatch_stream(
        self,
        entry: GeneratorEntry,
        payload: GeneratorInput,
    ) -> AsyncIterator[GeneratorEvent]:
        raise NotImplementedError(
            f"McpAdapter.dispatch_stream is not wired yet (generator {entry.id!r}). "
            "Add an MCP client dependency and translate notifications into GeneratorEvent."
        )
        # Unreachable yield so this stays an async generator in type-checks.
        yield  # type: ignore[unreachable]
