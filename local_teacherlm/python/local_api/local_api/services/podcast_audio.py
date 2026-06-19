from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import shutil
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import numpy as np

from local_api.config import get_settings


logger = logging.getLogger(__name__)

PodcastLanguage = Literal["en", "fr"]


@dataclass(frozen=True)
class VoiceBundle:
    language: PodcastLanguage
    archive_name: str
    url: str
    sha256: str
    directory_name: str
    model_name: str
    speaker_ids: tuple[int, int] = (0, 1)


VOICE_BUNDLES: dict[PodcastLanguage, VoiceBundle] = {
    "en": VoiceBundle(
        language="en",
        archive_name="vits-piper-en_US-libritts_r-medium-int8.tar.bz2",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
            "vits-piper-en_US-libritts_r-medium-int8.tar.bz2"
        ),
        sha256="7e4552e239988f4896872822b56e99e0e9e00958164e3f6bdf5ee14391fbe829",
        directory_name="vits-piper-en_US-libritts_r-medium-int8",
        model_name="en_US-libritts_r-medium.onnx",
    ),
    "fr": VoiceBundle(
        language="fr",
        archive_name="vits-piper-fr_FR-upmc-medium-int8.tar.bz2",
        url=(
            "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
            "vits-piper-fr_FR-upmc-medium-int8.tar.bz2"
        ),
        sha256="ec4930dd3778e19e6ea2bc0d16c5a50a82f428ba7ee43dceab1df7efbaf9f1d3",
        directory_name="vits-piper-fr_FR-upmc-medium-int8",
        model_name="fr_FR-upmc-medium.onnx",
    ),
}

FRENCH_SECONDARY_BUNDLE = VoiceBundle(
    language="fr",
    archive_name="vits-piper-fr_FR-tom-medium-int8.tar.bz2",
    url=(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
        "vits-piper-fr_FR-tom-medium-int8.tar.bz2"
    ),
    sha256="b158f884eb4231b8dddd372c3dc506949630c992a8e48ca692a90323e8bb4e09",
    directory_name="vits-piper-fr_FR-tom-medium-int8",
    model_name="fr_FR-tom-medium.onnx",
    speaker_ids=(0, 0),
)


class PodcastAudioError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class PodcastAudioResult:
    mp3: bytes
    duration_ms: int
    sample_rate: int
    model: str
    speaker_ids: tuple[int, int]
    cache_hit: bool


class PodcastModelManager:
    def __init__(self, models_dir: Path | None = None) -> None:
        self.models_dir = models_dir or (get_settings().data_dir / "models" / "tts")
        self._locks: dict[str, asyncio.Lock] = {}

    async def ensure(self, language: PodcastLanguage) -> tuple[Path, bool]:
        return await self.ensure_bundle(VOICE_BUNDLES[language])

    async def ensure_pair(self, language: PodcastLanguage) -> tuple[tuple[Path, Path], bool]:
        primary_dir, primary_hit = await self.ensure(language)
        if language != "fr":
            return (primary_dir, primary_dir), primary_hit
        secondary_dir, secondary_hit = await self.ensure_bundle(FRENCH_SECONDARY_BUNDLE)
        return (primary_dir, secondary_dir), primary_hit and secondary_hit

    async def ensure_bundle(self, bundle: VoiceBundle) -> tuple[Path, bool]:
        lock = self._locks.setdefault(bundle.directory_name, asyncio.Lock())
        async with lock:
            final_dir = self.models_dir / bundle.directory_name
            if _bundle_is_ready(final_dir, bundle):
                return final_dir, True
            try:
                await asyncio.to_thread(self._download_and_extract, bundle, final_dir)
            except PodcastAudioError:
                raise
            except Exception as exc:  # noqa: BLE001 - normalize the model boundary.
                raise PodcastAudioError("model_download_failed", "The voice model could not be downloaded.") from exc
            return final_dir, False

    def _download_and_extract(self, bundle: VoiceBundle, final_dir: Path) -> None:
        self.models_dir.mkdir(parents=True, exist_ok=True)
        work_dir = Path(tempfile.mkdtemp(prefix=f".{bundle.language}-", dir=self.models_dir))
        archive_path = work_dir / bundle.archive_name
        try:
            digest = hashlib.sha256()
            try:
                with urllib.request.urlopen(bundle.url, timeout=180) as response, archive_path.open("wb") as output:
                    while block := response.read(1024 * 1024):
                        digest.update(block)
                        output.write(block)
            except Exception as exc:  # noqa: BLE001 - converted to a stable public error.
                raise PodcastAudioError("model_download_failed", "The voice model download failed.") from exc
            if digest.hexdigest().lower() != bundle.sha256.lower():
                raise PodcastAudioError("model_checksum_failed", "The downloaded voice model failed verification.")

            extract_dir = work_dir / "extracted"
            extract_dir.mkdir()
            try:
                with tarfile.open(archive_path, mode="r:bz2") as archive:
                    _safe_extract(archive, extract_dir)
            except PodcastAudioError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise PodcastAudioError("model_extract_failed", "The voice model archive could not be opened.") from exc

            extracted_bundle = extract_dir / bundle.directory_name
            if not _bundle_is_ready(extracted_bundle, bundle):
                raise PodcastAudioError("model_extract_failed", "The voice model archive is incomplete.")
            if final_dir.exists():
                shutil.rmtree(final_dir)
            os.replace(extracted_bundle, final_dir)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)


class PodcastAudioService:
    def __init__(self, model_manager: PodcastModelManager | None = None) -> None:
        self.model_manager = model_manager or PodcastModelManager()
        self._engines: dict[str, Any] = {}
        self._engine_locks: dict[str, asyncio.Lock] = {}

    async def synthesize(self, turns: list[dict[str, Any]], language: str) -> PodcastAudioResult:
        if language not in VOICE_BUNDLES:
            raise PodcastAudioError("unsupported_language", "Audio is currently available in English and French.")
        typed_language: PodcastLanguage = language  # type: ignore[assignment]
        bundle = VOICE_BUNDLES[typed_language]
        model_dirs, cache_hit = await self.model_manager.ensure_pair(typed_language)
        lock = self._engine_locks.setdefault(typed_language, asyncio.Lock())
        async with lock:
            try:
                primary_engine = await asyncio.to_thread(self._engine, bundle, model_dirs[0])
                secondary_bundle = FRENCH_SECONDARY_BUNDLE if typed_language == "fr" else bundle
                secondary_engine = await asyncio.to_thread(self._engine, secondary_bundle, model_dirs[1])
                mp3, duration_ms, sample_rate = await asyncio.to_thread(
                    self._synthesize_sync,
                    (primary_engine, secondary_engine),
                    turns,
                    bundle,
                )
            except PodcastAudioError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep transcript fallback stable.
                logger.exception("Podcast speech synthesis failed")
                raise PodcastAudioError("synthesis_failed", "The local voice engine could not create audio.") from exc
        return PodcastAudioResult(
            mp3=mp3,
            duration_ms=duration_ms,
            sample_rate=sample_rate,
            model=(
                f"{bundle.directory_name}+{FRENCH_SECONDARY_BUNDLE.directory_name}"
                if typed_language == "fr"
                else bundle.directory_name
            ),
            speaker_ids=(0, 0) if typed_language == "fr" else bundle.speaker_ids,
            cache_hit=cache_hit,
        )

    def _engine(self, bundle: VoiceBundle, model_dir: Path) -> Any:
        cached = self._engines.get(bundle.directory_name)
        if cached is not None:
            return cached
        try:
            import sherpa_onnx
        except Exception as exc:  # pragma: no cover - depends on optional local runtime.
            raise PodcastAudioError("tts_unavailable", "The local voice runtime is not installed.") from exc

        config = sherpa_onnx.OfflineTtsConfig(
            model=sherpa_onnx.OfflineTtsModelConfig(
                vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                    model=str(model_dir / bundle.model_name),
                    tokens=str(model_dir / "tokens.txt"),
                    data_dir=str(model_dir / "espeak-ng-data"),
                ),
                num_threads=max(1, min(4, os.cpu_count() or 1)),
                provider="cpu",
                debug=False,
            ),
            max_num_sentences=1,
        )
        if not config.validate():
            raise PodcastAudioError("model_invalid", "The cached voice model is invalid.")
        engine = sherpa_onnx.OfflineTts(config)
        self._engines[bundle.directory_name] = engine
        return engine

    @staticmethod
    def _synthesize_sync(
        engines: tuple[Any, Any],
        turns: list[dict[str, Any]],
        bundle: VoiceBundle,
    ) -> tuple[bytes, int, int]:
        try:
            import lameenc
            import sherpa_onnx
        except Exception as exc:  # pragma: no cover - depends on optional local runtime.
            raise PodcastAudioError("tts_unavailable", "The local MP3 runtime is not installed.") from exc

        parts: list[np.ndarray] = []
        sample_rate = int(engines[0].sample_rate)
        turn_pause = np.zeros(int(sample_rate * 0.24), dtype=np.float32)
        sentence_pause = np.zeros(int(sample_rate * 0.08), dtype=np.float32)
        for turn_index, turn in enumerate(turns):
            chunks = _tts_chunks(str(turn.get("text") or ""), language=bundle.language)
            if not chunks:
                continue
            speaker_index = 0 if turn.get("speaker") == "host_a" else 1
            engine = engines[speaker_index]
            speaker_id = bundle.speaker_ids[speaker_index] if bundle.language != "fr" else 0
            if parts:
                parts.append(turn_pause)
            for chunk_index, text in enumerate(chunks):
                if chunk_index:
                    parts.append(sentence_pause)
                samples, generated_sample_rate = _generate_with_retry(
                    engine,
                    text,
                    speaker_id,
                    sherpa_onnx.GenerationConfig,
                    turn_index=turn_index,
                    chunk_index=chunk_index,
                )
                if generated_sample_rate != sample_rate:
                    samples = _resample_audio(samples, generated_sample_rate, sample_rate)
                parts.append(samples)
        if not parts:
            raise PodcastAudioError("synthesis_failed", "The podcast script did not contain speakable turns.")

        combined = np.concatenate(parts)
        combined = np.nan_to_num(combined, nan=0.0, posinf=1.0, neginf=-1.0)
        peak = float(np.max(np.abs(combined))) or 1.0
        combined = np.clip(combined * (0.95 / max(peak, 0.95)), -1.0, 1.0)
        pcm = (combined * 32767.0).astype("<i2").tobytes()
        try:
            encoder = lameenc.Encoder()
            encoder.set_bit_rate(96)
            encoder.set_in_sample_rate(sample_rate)
            encoder.set_channels(1)
            encoder.set_quality(2)
            mp3 = bytes(encoder.encode(pcm) + encoder.flush())
        except Exception as exc:  # noqa: BLE001
            raise PodcastAudioError("encoding_failed", "The generated speech could not be encoded as MP3.") from exc
        if not mp3:
            raise PodcastAudioError("encoding_failed", "The generated MP3 was empty.")
        return mp3, round(len(combined) / sample_rate * 1000), sample_rate


def _bundle_is_ready(path: Path, bundle: VoiceBundle) -> bool:
    return (
        path.is_dir()
        and (path / bundle.model_name).is_file()
        and (path / "tokens.txt").is_file()
        and (path / "espeak-ng-data").is_dir()
        and (path / "MODEL_CARD").is_file()
    )


def _tts_chunks(text: str, max_chars: int = 420, language: str = "") -> list[str]:
    """Remove visual markup and bound each voice-engine request."""
    cleaned = re.sub(r"<[^>]+>", " ", text)
    cleaned = re.sub(r"[*_`#]+", "", cleaned)
    cleaned = cleaned.replace("▶", ". ").replace("•", ". ")
    cleaned = _normalize_pronunciation(cleaned, language)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return []

    sentences = re.split(r"(?<=[.!?;:])\s+", cleaned)
    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) > max_chars:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(_split_long_tts_text(sentence, max_chars))
            continue
        candidate = f"{current} {sentence}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = sentence
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _normalize_pronunciation(text: str, language: str) -> str:
    if language == "fr":
        replacements = (
            (r"\bnDCG\s*@\s*k\b", "gain cumulatif actualisé normalisé à k"),
            (r"\bPrécision\s*@\s*k\b", "précision à k"),
            (r"\bRappel\s*@\s*k\b", "rappel à k"),
            (r"\bTop\s*-\s*N\b", "classement des N meilleurs"),
            (r"\bTF\s*-\s*IDF\b", "té effe, i dé effe"),
            (r"\buser\s*-\s*based\b", "approche utilisateur"),
            (r"\bitem\s*-\s*based\b", "approche élément"),
            (r"\bRMSE\b", "racine de l'erreur quadratique moyenne"),
            (r"\bnDCG\b", "gain cumulatif actualisé normalisé"),
            (r"\bNCF\b", "filtrage collaboratif neuronal"),
            (r"\bCBF\b", "filtrage basé sur le contenu"),
            (r"\bRNN\b", "réseau de neurones récurrent"),
            (r"\bSVD\b", "esse vé dé"),
            (r"\bCF\b", "filtrage collaboratif"),
        )
    elif language == "en":
        replacements = (
            (r"\bnDCG\s*@\s*k\b", "normalized discounted cumulative gain at k"),
            (r"\bPrecision\s*@\s*k\b", "precision at k"),
            (r"\bRecall\s*@\s*k\b", "recall at k"),
            (r"\bTop\s*-\s*N\b", "top N"),
            (r"\bTF\s*-\s*IDF\b", "T F I D F"),
            (r"\bRMSE\b", "root mean squared error"),
            (r"\bnDCG\b", "normalized discounted cumulative gain"),
            (r"\bSVD\b", "S V D"),
            (r"\bRNN\b", "R N N"),
            (r"\bNCF\b", "neural collaborative filtering"),
        )
    else:
        replacements = ()
    for pattern, replacement in replacements:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    if language == "fr":
        text = re.sub(r"(?<=\d)\.(?=\d)", ",", text)
    return text.replace("@", " à " if language == "fr" else " at ")


def _resample_audio(samples: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or samples.size == 0:
        return samples
    target_size = max(1, round(samples.size * target_rate / source_rate))
    source_positions = np.arange(samples.size, dtype=np.float64)
    target_positions = np.linspace(0, samples.size - 1, target_size, dtype=np.float64)
    return np.interp(target_positions, source_positions, samples).astype(np.float32)


def _split_long_tts_text(text: str, max_chars: int) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = word
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def _generate_with_retry(
    engine: Any,
    text: str,
    speaker_id: int,
    generation_config_factory: Any,
    *,
    turn_index: int,
    chunk_index: int,
) -> tuple[np.ndarray, int]:
    last_error: Exception | None = None
    for attempt in range(2):
        try:
            generation = generation_config_factory()
            generation.sid = speaker_id
            generation.speed = 1.0
            generation.silence_scale = 0.15
            audio = engine.generate(text, generation)
            samples = np.asarray(audio.samples, dtype=np.float32)
            if samples.size == 0:
                raise RuntimeError("voice engine returned no samples")
            return samples, int(audio.sample_rate)
        except Exception as exc:  # noqa: BLE001 - retry the local native boundary once.
            last_error = exc
            if attempt == 0:
                logger.warning(
                    "Retrying podcast voice synthesis at turn %s, chunk %s",
                    turn_index + 1,
                    chunk_index + 1,
                )
    raise PodcastAudioError(
        "synthesis_failed",
        f"Voice synthesis failed at turn {turn_index + 1}.",
    ) from last_error


def _safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    root = destination.resolve()
    members = archive.getmembers()
    for member in members:
        target = (destination / member.name).resolve()
        try:
            target.relative_to(root)
        except ValueError as exc:
            raise PodcastAudioError("model_extract_failed", "The voice archive contains an unsafe path.") from exc
        if member.issym() or member.islnk():
            raise PodcastAudioError("model_extract_failed", "The voice archive contains unsupported links.")
    archive.extractall(destination, members=members, filter="data")


@lru_cache(maxsize=1)
def get_podcast_audio_service() -> PodcastAudioService:
    return PodcastAudioService()
