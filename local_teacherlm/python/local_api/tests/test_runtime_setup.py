from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "teacherlm_core"))
sys.path.insert(0, str(ROOT / "local_api"))

from local_api.config import get_settings
from local_api.services import runtime_setup


def test_ollama_model_names_normalizes_latest_tag() -> None:
    response = {
        "models": [
            {"model": "llama3.2:latest"},
            {"name": "qwen3:4b"},
        ]
    }
    assert runtime_setup._ollama_model_names(response) == {
        "llama3.2:latest",
        "llama3.2",
        "qwen3:4b",
    }


def test_ollama_download_progress_reads_runtime_marker(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEACHERLM_APP_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    marker = tmp_path / "runtime" / "ollama-download-progress.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("25,100", encoding="utf-8")

    assert runtime_setup._ollama_download_progress() == (25, 100)

    marker.write_text("invalid", encoding="utf-8")
    assert runtime_setup._ollama_download_progress() == (0, 0)
    get_settings.cache_clear()


def test_setup_status_recovers_completed_components(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("TEACHERLM_APP_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    manifest_path = tmp_path / "models" / "setup-state.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(
        json.dumps(
            {
                "components": {
                    "chat_model": "llama3.2",
                    "embeddings": "intfloat/multilingual-e5-large",
                    "reranker": "BAAI/bge-reranker-base",
                    "voice_en": "sherpa-onnx-piper-v1",
                    "voice_fr": "sherpa-onnx-piper-v1",
                }
            }
        ),
        encoding="utf-8",
    )
    for cache in (tmp_path / "models" / "embeddings", tmp_path / "models" / "rerankers", tmp_path / "models" / "tts"):
        cache.mkdir(parents=True)
        (cache / "ready").write_text("ok", encoding="utf-8")

    class Response:
        is_success = True

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, _exc_type, _exc, _traceback):
            return False

        async def get(self, _url):
            return Response()

    class OllamaClient:
        def __init__(self, **_kwargs):
            pass

        async def list(self):
            return {"models": [{"model": "llama3.2:latest"}]}

    monkeypatch.setattr(runtime_setup.httpx, "AsyncClient", lambda **_kwargs: Client())
    monkeypatch.setattr(runtime_setup.ollama, "AsyncClient", OllamaClient)
    status = asyncio.run(runtime_setup.RuntimeSetupService().status())
    assert status["ready"] is True
    assert status["progress"] == 1.0
    assert {component["status"] for component in status["components"]} == {"ready"}
    get_settings.cache_clear()
