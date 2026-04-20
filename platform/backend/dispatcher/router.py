from __future__ import annotations

from collections.abc import AsyncIterator

from teacherlm_core.schemas.generator_io import GeneratorInput, GeneratorOutput

from dispatcher.adapters.api_adapter import ApiAdapter, GeneratorEvent
from dispatcher.adapters.mcp_adapter import McpAdapter
from dispatcher.registry import GeneratorEntry, GeneratorRegistry, get_registry


class DisabledGeneratorError(RuntimeError):
    pass


class UnsupportedAdapterError(RuntimeError):
    pass


class GeneratorRouter:
    """Resolves a GeneratorEntry (by id or output_type) and dispatches through
    the correct transport adapter."""

    def __init__(
        self,
        registry: GeneratorRegistry | None = None,
        *,
        api_adapter: ApiAdapter | None = None,
        mcp_adapter: McpAdapter | None = None,
    ) -> None:
        self._registry = registry or get_registry()
        self._api = api_adapter or ApiAdapter()
        self._mcp = mcp_adapter or McpAdapter()

    # --- resolution ---

    def resolve_by_id(self, generator_id: str) -> GeneratorEntry:
        entry = self._registry.get(generator_id)
        self._require_enabled(entry)
        return entry

    def resolve_for_output(self, output_type: str) -> GeneratorEntry:
        entry = self._registry.for_output_type(output_type)
        self._require_enabled(entry)
        return entry

    def resolve_chat_default(self) -> GeneratorEntry:
        entry = self._registry.chat_default()
        self._require_enabled(entry)
        return entry

    # --- dispatch ---

    async def dispatch(
        self,
        entry: GeneratorEntry,
        payload: GeneratorInput,
    ) -> GeneratorOutput:
        adapter = self._adapter_for(entry)
        return await adapter.dispatch(entry, payload)

    async def dispatch_stream(
        self,
        entry: GeneratorEntry,
        payload: GeneratorInput,
    ) -> AsyncIterator[GeneratorEvent]:
        adapter = self._adapter_for(entry)
        async for event in adapter.dispatch_stream(entry, payload):
            yield event

    # --- internals ---

    def _adapter_for(self, entry: GeneratorEntry) -> ApiAdapter | McpAdapter:
        match entry.type:
            case "api":
                return self._api
            case "mcp":
                return self._mcp
            case other:
                raise UnsupportedAdapterError(f"generator {entry.id!r}: unknown type {other!r}")

    @staticmethod
    def _require_enabled(entry: GeneratorEntry) -> None:
        if not entry.enabled:
            raise DisabledGeneratorError(f"generator {entry.id!r} is disabled in the registry")


_router: GeneratorRouter | None = None


def get_router() -> GeneratorRouter:
    global _router
    if _router is None:
        _router = GeneratorRouter()
    return _router
