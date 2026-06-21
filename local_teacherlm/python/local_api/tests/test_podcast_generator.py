from __future__ import annotations

import asyncio
import io
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "teacherlm_core"))
sys.path.insert(0, str(ROOT / "local_api"))


def _chunks():
    from teacherlm_core.schemas import Chunk

    first = (
        "Photosynthesis converts light energy into chemical energy in plants. "
        "Chlorophyll absorbs light in the chloroplast. "
        "The light-dependent reactions produce energy-carrying molecules. "
        "Those molecules support carbon fixation in the Calvin cycle. "
        "Carbon dioxide supplies the carbon used to form sugars. "
        "Water provides electrons and oxygen is released as a by-product. "
        "The process connects energy capture with the production of organic matter. "
        "Environmental conditions can affect the rate of photosynthesis."
    )
    second = (
        "Cellular respiration releases usable energy from organic molecules. "
        "Glycolysis begins the breakdown of glucose. "
        "Later reactions transfer electrons through carriers. "
        "A proton gradient helps ATP synthase produce ATP. "
        "Oxygen serves as the final electron acceptor in aerobic respiration. "
        "The products include carbon dioxide and water. "
        "Photosynthesis and respiration therefore move matter and energy through living systems."
    )
    return [
        Chunk(
            text=first,
            source="biology.md",
            score=1.0,
            chunk_id="photosynthesis",
            metadata={"key_concepts": ["Photosynthesis", "Chlorophyll"]},
        ),
        Chunk(
            text=second,
            source="biology.md",
            score=0.9,
            chunk_id="respiration",
            metadata={"key_concepts": ["Cellular respiration", "ATP"]},
        ),
    ]


def test_podcast_options_support_minutes_and_legacy_presets() -> None:
    from local_api.services.generators import _podcast_options

    assert _podcast_options({}).duration_minutes == 6
    assert _podcast_options({"duration": "short"}).duration_minutes == 3
    assert _podcast_options({"duration": "long"}).duration_minutes == 15
    assert _podcast_options({"duration": "long", "duration_minutes": 7}).duration_minutes == 7
    with pytest.raises(RuntimeError):
        _podcast_options({"duration_minutes": 2})
    with pytest.raises(RuntimeError):
        _podcast_options({"duration_minutes": 16})
    with pytest.raises(RuntimeError):
        _podcast_options({"topic": "x" * 201})


def test_voice_text_is_chunked_and_transient_generation_is_retried() -> None:
    import numpy as np

    from local_api.services.podcast_audio import _generate_with_retry, _resample_audio, _tts_chunks

    chunks = _tts_chunks("**Introduction** ▶ " + "explication " * 80, max_chars=120)
    assert len(chunks) > 1
    assert all(len(chunk) <= 120 for chunk in chunks)
    assert all("**" not in chunk and "▶" not in chunk for chunk in chunks)

    class FlakyEngine:
        calls = 0

        def generate(self, _text, config):
            self.calls += 1
            assert config.sid == 1
            if self.calls == 1:
                raise RuntimeError("temporary native runtime failure")
            return SimpleNamespace(samples=[0.1, -0.1], sample_rate=22_050)

    engine = FlakyEngine()
    samples, sample_rate = _generate_with_retry(
        engine,
        "Une courte explication.",
        1,
        SimpleNamespace,
        turn_index=2,
        chunk_index=0,
    )

    assert engine.calls == 2
    assert samples.size == 2
    assert sample_rate == 22_050

    french = " ".join(
        _tts_chunks(
            "SVD, RMSE, nDCG@k, Précision@k, user-based, item-based, TF-IDF et CBF.",
            language="fr",
        )
    )
    assert "racine de l'erreur quadratique moyenne" in french
    assert "gain cumulatif actualisé normalisé à k" in french
    assert "approche utilisateur" in french and "approche élément" in french
    assert not any(raw in french for raw in ("RMSE", "nDCG", "user-based", "item-based", "TF-IDF", "CBF"))

    resampled = _resample_audio(np.ones(44_100, dtype=np.float32), 44_100, 22_050)
    assert resampled.size == 22_050


def test_artifact_payload_accepts_lameenc_bytearray() -> None:
    from local_api.services.artifacts import _to_bytes

    encoded = bytearray(b"ID3-podcast-audio")
    result = _to_bytes(encoded)

    assert result == b"ID3-podcast-audio"
    assert isinstance(result, bytes)


def test_fallback_script_is_grounded_alternating_and_near_target(monkeypatch) -> None:
    from teacherlm_core.schemas import GeneratorInput, LearnerState
    from local_api.services import generators

    monkeypatch.setattr(
        generators,
        "get_settings_service",
        lambda: SimpleNamespace(
            get_default_chat_provider_config=lambda: None,
            get_generator_settings=lambda: SimpleNamespace(podcast_audio_enabled=True),
        ),
    )
    chunks = _chunks()
    options = generators._podcast_options({"topic": "Plant energy", "duration_minutes": 3})
    arc = generators._narrative_arc(chunks, options.topic)
    numbered_arc = generators._narrative_arc(
        [
            chunks[0].model_copy(
                update={"text": "1. **Introduction au filtrage collaboratif et aux recommandations personnalisées.**"}
            )
        ],
        "Filtrage collaboratif",
    )
    assert numbered_arc["summary"] != "1."
    assert "Introduction au filtrage collaboratif" in numbered_arc["summary"]
    payload = GeneratorInput(
        conversation_id="podcast-test",
        user_message=options.topic,
        context_chunks=chunks,
        learner_state=LearnerState(conversation_id="podcast-test"),
        chat_history=[],
        options=options.model_dump(),
    )

    script, metadata = asyncio.run(generators._build_podcast_script(payload, chunks, arc, options, "en"))
    turns = script["turns"]
    word_count = sum(len(turn["text"].split()) for turn in turns)
    valid_ids = {chunk.chunk_id for chunk in chunks}

    assert word_count >= round(3 * 135 * 0.85)
    assert word_count <= round(3 * 135 * 1.15)
    assert all(turns[index]["speaker"] != turns[index - 1]["speaker"] for index in range(1, len(turns)))
    assert all(set(turn["source_chunk_ids"]) <= valid_ids for turn in turns)
    assert metadata["backend"] == "deterministic_grounded_fallback"
    assert metadata["target_words"] == 405


def test_audio_failure_returns_completed_transcript(monkeypatch) -> None:
    from teacherlm_core.schemas import GeneratorArtifact, GeneratorInput, LearnerState
    from local_api.services import generators
    from local_api.services.podcast_audio import PodcastAudioError

    created: list[dict] = []

    class FakeArtifacts:
        def create_artifact(self, _conversation_id, artifact_type, filename, payload, **kwargs):
            created.append(
                {
                    "type": artifact_type,
                    "filename": filename,
                    "payload": payload,
                    "mime_type": kwargs.get("mime_type"),
                }
            )
            return GeneratorArtifact(
                type=artifact_type,
                url=f"teacherlm-local://artifact/{artifact_type}",
                filename=filename,
                key=f"artifact-{artifact_type}",
            )

    class FailingAudio:
        async def synthesize(self, *_args, **_kwargs):
            raise PodcastAudioError("model_download_failed", "offline")

    monkeypatch.setattr(generators, "get_artifact_service", lambda: FakeArtifacts())
    monkeypatch.setattr(generators, "get_podcast_audio_service", lambda: FailingAudio())
    monkeypatch.setattr(
        generators,
        "get_settings_service",
        lambda: SimpleNamespace(
            get_default_chat_provider_config=lambda: None,
            get_generator_settings=lambda: SimpleNamespace(podcast_audio_enabled=True),
        ),
    )
    chunks = _chunks()
    payload = GeneratorInput(
        conversation_id="podcast-test",
        user_message="Plant energy",
        context_chunks=chunks,
        learner_state=LearnerState(conversation_id="podcast-test"),
        chat_history=[],
        options={"topic": "Plant energy", "duration_minutes": 3},
    )

    async def collect():
        return [event async for event in generators._podcast(payload)]

    events = asyncio.run(collect())
    done = next(event["data"] for event in events if event["event"] == "done")

    assert not any(event["event"] == "error" for event in events)
    assert [item["type"] for item in created] == ["transcript"]
    assert done["metadata"]["podcast"]["audio_status"] == "failed"
    assert done["metadata"]["podcast"]["audio_error_code"] == "model_download_failed"
    assert done["artifacts"][0]["type"] == "transcript"
    assert "Alex:" in created[0]["payload"] and "Sam:" in created[0]["payload"]


def test_audio_success_returns_mp3_before_transcript(monkeypatch) -> None:
    from teacherlm_core.schemas import GeneratorArtifact, GeneratorInput, LearnerState
    from local_api.services import generators
    from local_api.services.podcast_audio import PodcastAudioResult

    created: list[dict] = []

    class FakeArtifacts:
        def create_artifact(self, _conversation_id, artifact_type, filename, payload, **kwargs):
            created.append({"type": artifact_type, "filename": filename, "payload": payload, **kwargs})
            return GeneratorArtifact(
                type=artifact_type,
                url=f"teacherlm-local://artifact/{artifact_type}",
                filename=filename,
                key=f"artifact-{artifact_type}",
            )

    class SuccessfulAudio:
        async def synthesize(self, turns, language):
            assert language == "en"
            assert {turn["speaker"] for turn in turns} == {"host_a", "host_b"}
            return PodcastAudioResult(
                mp3=b"ID3fake",
                duration_ms=181_000,
                sample_rate=22_050,
                model="english-test-model",
                speaker_ids=(0, 1),
                cache_hit=True,
            )

    monkeypatch.setattr(generators, "get_artifact_service", lambda: FakeArtifacts())
    monkeypatch.setattr(generators, "get_podcast_audio_service", lambda: SuccessfulAudio())
    monkeypatch.setattr(
        generators,
        "get_settings_service",
        lambda: SimpleNamespace(
            get_default_chat_provider_config=lambda: None,
            get_generator_settings=lambda: SimpleNamespace(podcast_audio_enabled=True),
        ),
    )
    chunks = _chunks()
    payload = GeneratorInput(
        conversation_id="podcast-test",
        user_message="Plant energy",
        context_chunks=chunks,
        learner_state=LearnerState(conversation_id="podcast-test"),
        chat_history=[],
        options={"topic": "Plant energy", "duration_minutes": 3},
    )

    async def collect():
        return [event async for event in generators._podcast(payload)]

    done = next(event["data"] for event in asyncio.run(collect()) if event["event"] == "done")

    assert [item["type"] for item in created] == ["transcript", "podcast"]
    assert created[1]["mime_type"] == "audio/mpeg"
    assert done["metadata"]["podcast"]["audio_status"] == "ready"
    assert done["metadata"]["podcast"]["duration_ms"] == 181_000
    assert [artifact["type"] for artifact in done["artifacts"]] == ["podcast", "transcript"]


def test_disabled_audio_returns_transcript_without_loading_voice_service(monkeypatch) -> None:
    from teacherlm_core.schemas import GeneratorArtifact, GeneratorInput, LearnerState
    from local_api.services import generators

    created: list[dict] = []

    class FakeArtifacts:
        def create_artifact(self, _conversation_id, artifact_type, filename, payload, **kwargs):
            created.append({"type": artifact_type, "filename": filename, "payload": payload, **kwargs})
            return GeneratorArtifact(
                type=artifact_type,
                url=f"teacherlm-local://artifact/{artifact_type}",
                filename=filename,
                key=f"artifact-{artifact_type}",
            )

    monkeypatch.setattr(generators, "get_artifact_service", lambda: FakeArtifacts())
    monkeypatch.setattr(
        generators,
        "get_podcast_audio_service",
        lambda: pytest.fail("voice service must not load in transcript-only mode"),
    )
    monkeypatch.setattr(
        generators,
        "get_settings_service",
        lambda: SimpleNamespace(
            get_default_chat_provider_config=lambda: None,
            get_generator_settings=lambda: SimpleNamespace(podcast_audio_enabled=False),
        ),
    )
    payload = GeneratorInput(
        conversation_id="podcast-test",
        user_message="Plant energy",
        context_chunks=_chunks(),
        learner_state=LearnerState(conversation_id="podcast-test"),
        chat_history=[],
        options={"topic": "Plant energy", "duration_minutes": 3},
    )

    async def collect():
        return [event async for event in generators._podcast(payload)]

    events = asyncio.run(collect())
    done = next(event["data"] for event in events if event["event"] == "done")

    assert [item["type"] for item in created] == ["transcript"]
    assert not any(
        isinstance(event.get("data"), dict)
        and event["data"].get("stage") == "podcast_preparing_voices"
        for event in events
    )
    assert done["metadata"]["podcast"]["audio_requested"] is False
    assert done["metadata"]["podcast"]["audio_status"] == "disabled"
    assert done["metadata"]["podcast"]["audio_error_code"] is None
    assert [artifact["type"] for artifact in done["artifacts"]] == ["transcript"]
    assert "no voice model was loaded" in done["response"]


def test_model_cache_and_archive_safety(monkeypatch, tmp_path: Path) -> None:
    from local_api.services.podcast_audio import (
        FRENCH_SECONDARY_BUNDLE,
        PodcastAudioError,
        PodcastModelManager,
        VOICE_BUNDLES,
        _safe_extract,
    )

    bundle = VOICE_BUNDLES["en"]
    ready = tmp_path / bundle.directory_name
    ready.mkdir()
    (ready / bundle.model_name).write_bytes(b"model")
    (ready / "tokens.txt").write_text("a 1", encoding="utf-8")
    (ready / "espeak-ng-data").mkdir()
    (ready / "MODEL_CARD").write_text("license", encoding="utf-8")
    manager = PodcastModelManager(tmp_path)
    monkeypatch.setattr(manager, "_download_and_extract", lambda *_args: pytest.fail("cache should be reused"))
    path, cache_hit = asyncio.run(manager.ensure("en"))
    assert path == ready
    assert cache_hit is True
    assert bundle.speaker_ids == (0, 1)

    french_dirs = []
    for french_bundle in (VOICE_BUNDLES["fr"], FRENCH_SECONDARY_BUNDLE):
        french_ready = tmp_path / french_bundle.directory_name
        french_ready.mkdir()
        (french_ready / french_bundle.model_name).write_bytes(b"model")
        (french_ready / "tokens.txt").write_text("a 1", encoding="utf-8")
        (french_ready / "espeak-ng-data").mkdir()
        (french_ready / "MODEL_CARD").write_text("license", encoding="utf-8")
        french_dirs.append(french_ready)
    pair, pair_cache_hit = asyncio.run(manager.ensure_pair("fr"))
    assert pair == tuple(french_dirs)
    assert pair_cache_hit is True

    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w:bz2") as archive:
        member = tarfile.TarInfo("../escape.txt")
        data = b"unsafe"
        member.size = len(data)
        archive.addfile(member, io.BytesIO(data))
    payload.seek(0)
    with tarfile.open(fileobj=payload, mode="r:bz2") as archive:
        with pytest.raises(PodcastAudioError) as exc:
            _safe_extract(archive, tmp_path / "extract")
    assert exc.value.code == "model_extract_failed"


def test_model_download_rejects_bad_checksum(monkeypatch, tmp_path: Path) -> None:
    from local_api.services import podcast_audio

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    monkeypatch.setattr(
        podcast_audio.urllib.request,
        "urlopen",
        lambda *_args, **_kwargs: FakeResponse(b"not the expected archive"),
    )
    manager = podcast_audio.PodcastModelManager(tmp_path)
    bundle = podcast_audio.VOICE_BUNDLES["fr"]
    final_dir = tmp_path / bundle.directory_name

    with pytest.raises(podcast_audio.PodcastAudioError) as exc:
        manager._download_and_extract(bundle, final_dir)

    assert exc.value.code == "model_checksum_failed"
    assert not final_dir.exists()
