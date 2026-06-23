from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import httpx
import ollama

from local_api.config import get_settings
from local_api.services.podcast_audio import PodcastModelManager
from local_api.services.settings import get_settings_service


logger = logging.getLogger(__name__)

_COMPONENTS = (
    ("ollama_runtime", "Local AI engine"),
    ("chat_model", "Teacher chat model"),
    ("embeddings", "Course search embeddings"),
    ("reranker", "Answer relevance reranker"),
    ("voice_en", "English podcast voices"),
    ("voice_fr", "French podcast voices"),
)


class RuntimeSetupService:
    """Downloads and verifies every model needed by the offline desktop app."""

    def __init__(self) -> None:
        self._task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._running = False
        self._progress = 0.0
        self._message = "Checking local components"
        self._error: str | None = None
        self._active_component: str | None = None
        self._states = {component_id: "pending" for component_id, _ in _COMPONENTS}
        self._details = {component_id: "" for component_id, _ in _COMPONENTS}

    async def status(self) -> dict[str, Any]:
        if not self._running:
            await self._refresh_installed_state()
        components = [
            {
                "id": component_id,
                "label": label,
                "status": self._states[component_id],
                "detail": self._details[component_id],
            }
            for component_id, label in _COMPONENTS
        ]
        return {
            "ready": all(component["status"] == "ready" for component in components),
            "running": self._running,
            "progress": round(self._progress, 3),
            "message": self._message,
            "error": self._error,
            "active_component": self._active_component,
            "components": components,
        }

    async def start(self) -> dict[str, Any]:
        async with self._lock:
            if self._task is None or self._task.done():
                self._error = None
                self._task = asyncio.create_task(self._run(), name="teacherlm-model-setup")
        return await self.status()

    async def _run(self) -> None:
        self._running = True
        self._progress = 0.01
        self._message = "Starting TeacherLM setup"
        self._error = None
        try:
            await self._prepare_ollama()
            await self._prepare_chat_model()
            await self._prepare_embeddings()
            await self._prepare_reranker()
            await self._prepare_voice("en")
            await self._prepare_voice("fr")
            self._active_component = None
            self._progress = 1.0
            self._message = "TeacherLM is ready"
        except Exception as exc:  # noqa: BLE001 - expose one stable setup boundary.
            logger.exception("TeacherLM local model setup failed")
            if self._active_component:
                self._states[self._active_component] = "error"
                self._details[self._active_component] = str(exc)
            self._error = str(exc) or "Setup could not finish."
            self._message = "Setup needs attention"
        finally:
            self._running = False

    async def _prepare_ollama(self) -> None:
        component_id = "ollama_runtime"
        self._begin(component_id, 0.02, "Starting the local AI engine")
        base_url = get_settings().default_ollama_base_url.rstrip("/")
        for _ in range(60):
            try:
                async with httpx.AsyncClient(timeout=1.0) as client:
                    response = await client.get(f"{base_url}/api/version")
                    if response.is_success:
                        self._complete(component_id, "Local engine is running", 0.05)
                        return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
        raise RuntimeError(
            "The bundled local AI engine did not start. Restart TeacherLM and try again."
        )

    async def _prepare_chat_model(self) -> None:
        component_id = "chat_model"
        provider = get_settings_service().get_default_chat_provider_config()
        model_name = provider.model_name if provider and provider.provider_type == "ollama" else get_settings().default_ollama_model
        base_url = provider.base_url if provider and provider.provider_type == "ollama" else get_settings().default_ollama_base_url
        self._begin(component_id, 0.06, f"Downloading {model_name}")
        client = ollama.AsyncClient(host=base_url)
        installed = await client.list()
        if model_name not in _ollama_model_names(installed):
            stream = await client.pull(model_name, stream=True)
            async for update in stream:
                total = int(getattr(update, "total", 0) or 0)
                completed = int(getattr(update, "completed", 0) or 0)
                status = str(getattr(update, "status", "Downloading the teacher model") or "")
                if total > 0:
                    self._progress = 0.06 + min(0.64, (completed / total) * 0.64)
                self._message = status.capitalize()
        self._mark_installed(component_id, model_name)
        self._complete(component_id, model_name, 0.70)

    async def _prepare_embeddings(self) -> None:
        component_id = "embeddings"
        settings = get_settings()
        self._begin(component_id, 0.71, f"Downloading {settings.embedding_model}")

        def load() -> None:
            from fastembed import TextEmbedding

            model = TextEmbedding(
                model_name=settings.embedding_model,
                cache_dir=str(settings.embedding_cache_dir),
            )
            next(iter(model.embed(["TeacherLM local model verification"])))

        await asyncio.to_thread(load)
        self._mark_installed(component_id, settings.embedding_model)
        self._complete(component_id, settings.embedding_model, 0.82)

    async def _prepare_reranker(self) -> None:
        component_id = "reranker"
        settings = get_settings()
        self._begin(component_id, 0.83, f"Downloading {settings.retrieval_reranker_model}")

        def load() -> None:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            model = TextCrossEncoder(
                model_name=settings.retrieval_reranker_model,
                cache_dir=str(settings.reranker_cache_dir),
            )
            model.rerank("teacher", ["teacher"])

        await asyncio.to_thread(load)
        self._mark_installed(component_id, settings.retrieval_reranker_model)
        self._complete(component_id, settings.retrieval_reranker_model, 0.90)

    async def _prepare_voice(self, language: str) -> None:
        component_id = f"voice_{language}"
        target_progress = 0.95 if language == "en" else 0.99
        label = "English" if language == "en" else "French"
        self._begin(component_id, self._progress, f"Downloading {label} podcast voices")
        manager = PodcastModelManager()
        await manager.ensure_pair(language)  # type: ignore[arg-type]
        self._mark_installed(component_id, "sherpa-onnx-piper-v1")
        self._complete(component_id, f"{label} voices", target_progress)

    async def _refresh_installed_state(self) -> None:
        manifest = self._read_manifest()
        installed = manifest.get("components", {}) if isinstance(manifest, dict) else {}
        for component_id, _ in _COMPONENTS:
            if component_id in {"ollama_runtime", "chat_model"}:
                continue
            if installed.get(component_id) and self._component_cache_exists(component_id):
                self._states[component_id] = "ready"
                self._details[component_id] = str(installed[component_id])
            else:
                self._states[component_id] = "pending"
                self._details[component_id] = "Not downloaded"

        base_url = get_settings().default_ollama_base_url.rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=0.8) as client:
                response = await client.get(f"{base_url}/api/version")
                if response.is_success:
                    self._states["ollama_runtime"] = "ready"
                    self._details["ollama_runtime"] = "Local engine is running"
                    model_name = get_settings().default_ollama_model
                    client = ollama.AsyncClient(host=base_url)
                    names = _ollama_model_names(await client.list())
                    if model_name in names:
                        self._states["chat_model"] = "ready"
                        self._details["chat_model"] = model_name
                    else:
                        self._states["chat_model"] = "pending"
                        self._details["chat_model"] = "Not downloaded"
        except httpx.HTTPError:
            self._states["ollama_runtime"] = "pending"
            self._details["ollama_runtime"] = "Waiting for local engine"
            self._states["chat_model"] = "pending"
            self._details["chat_model"] = "Waiting for local engine"
        except Exception:  # noqa: BLE001 - setup remains retryable when the engine is starting.
            self._states["chat_model"] = "pending"
            self._details["chat_model"] = "Waiting for local engine"

        completed = sum(state == "ready" for state in self._states.values())
        if completed == len(_COMPONENTS):
            self._progress = 1.0
            self._message = "TeacherLM is ready"
            self._error = None
        elif self._progress >= 1.0:
            self._progress = completed / len(_COMPONENTS)

    def _begin(self, component_id: str, progress: float, message: str) -> None:
        self._active_component = component_id
        self._states[component_id] = "downloading"
        self._details[component_id] = message
        self._progress = max(self._progress, progress)
        self._message = message

    def _complete(self, component_id: str, detail: str, progress: float) -> None:
        self._states[component_id] = "ready"
        self._details[component_id] = detail
        self._progress = max(self._progress, progress)

    @property
    def _manifest_path(self) -> Path:
        return get_settings().models_dir / "setup-state.json"

    def _read_manifest(self) -> dict[str, Any]:
        try:
            return json.loads(self._manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            return {}

    def _mark_installed(self, component_id: str, version: str) -> None:
        manifest = self._read_manifest()
        components = manifest.setdefault("components", {})
        components[component_id] = version
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(self._manifest_path)

    @staticmethod
    def _component_cache_exists(component_id: str) -> bool:
        settings = get_settings()
        if component_id == "embeddings":
            path = settings.embedding_cache_dir
        elif component_id == "reranker":
            path = settings.reranker_cache_dir
        else:
            path = settings.models_dir / "tts"
        try:
            return path.is_dir() and any(path.iterdir())
        except OSError:
            return False


def _ollama_model_names(response: Any) -> set[str]:
    models = getattr(response, "models", None)
    if models is None and isinstance(response, dict):
        models = response.get("models", [])
    names: set[str] = set()
    for model in models or []:
        name = getattr(model, "model", None) or getattr(model, "name", None)
        if name is None and isinstance(model, dict):
            name = model.get("model") or model.get("name")
        if name:
            text = str(name)
            names.add(text)
            names.add(text.removesuffix(":latest"))
    return names


_runtime_setup_service: RuntimeSetupService | None = None


def get_runtime_setup_service() -> RuntimeSetupService:
    global _runtime_setup_service
    if _runtime_setup_service is None:
        _runtime_setup_service = RuntimeSetupService()
    return _runtime_setup_service
