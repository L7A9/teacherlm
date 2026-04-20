from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from config import get_settings


AdapterType = Literal["api", "mcp"]


class GeneratorEntry(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    name: str | None = None
    type: AdapterType = "api"
    endpoint: str
    enabled: bool = True
    output_type: str
    icon: str | None = None
    description: str | None = None
    is_chat_default: bool = False


class GeneratorNotFound(KeyError):
    pass


class GeneratorRegistry:
    """Loads generators_registry.json once and serves lookups.

    The registry file is project-root-relative (see Settings.generators_registry_path).
    Reload by calling `reload()` — useful in tests or after editing the file.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path or get_settings().generators_registry_path
        self._lock = threading.Lock()
        self._entries: list[GeneratorEntry] = []
        self._by_id: dict[str, GeneratorEntry] = {}
        self._loaded = False

    # --- loading ---

    def reload(self) -> None:
        with self._lock:
            self._entries = self._read()
            self._by_id = {e.id: e for e in self._entries}
            self._loaded = True

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self.reload()

    def _read(self) -> list[GeneratorEntry]:
        if not self._path.exists():
            raise FileNotFoundError(f"generators_registry.json not found at {self._path}")
        raw = json.loads(self._path.read_text(encoding="utf-8"))
        items = raw.get("generators", []) if isinstance(raw, dict) else raw
        return [GeneratorEntry.model_validate(item) for item in items]

    # --- lookups ---

    def all(self, *, only_enabled: bool = False) -> list[GeneratorEntry]:
        self._ensure_loaded()
        if only_enabled:
            return [e for e in self._entries if e.enabled]
        return list(self._entries)

    def get(self, generator_id: str) -> GeneratorEntry:
        self._ensure_loaded()
        try:
            return self._by_id[generator_id]
        except KeyError as exc:
            raise GeneratorNotFound(generator_id) from exc

    def for_output_type(self, output_type: str, *, only_enabled: bool = True) -> GeneratorEntry:
        """Return the first (enabled) generator whose output_type matches."""
        self._ensure_loaded()
        for entry in self._entries:
            if entry.output_type != output_type:
                continue
            if only_enabled and not entry.enabled:
                continue
            return entry
        raise GeneratorNotFound(f"no generator for output_type={output_type!r}")

    def chat_default(self) -> GeneratorEntry:
        self._ensure_loaded()
        for entry in self._entries:
            if entry.is_chat_default and entry.enabled:
                return entry
        # Fall back to any enabled generator with output_type="text".
        return self.for_output_type("text")


_registry: GeneratorRegistry | None = None


def get_registry() -> GeneratorRegistry:
    global _registry
    if _registry is None:
        _registry = GeneratorRegistry()
    return _registry
