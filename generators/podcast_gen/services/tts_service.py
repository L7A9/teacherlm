from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
import urllib.request
import uuid
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from ..config import get_settings
from ..schemas import PodcastScript, Segment, TTSResult
from .text_sanitizer import sanitize_spoken_text


logger = logging.getLogger(__name__)


Backend = Literal["piper", "kokoro", "pyttsx3"]


class TTSUnavailable(RuntimeError):
    """Raised when no TTS backend can produce audio."""


@dataclass(frozen=True)
class VoicePlan:
    """Resolved per-language voice + speed + pitch assignments for both hosts."""

    backend: Backend
    lang: str                  # lang tag, e.g. "fr-fr"
    host_a_voice: str          # voice id meaningful to `backend`
    host_b_voice: str
    host_a_speed: float
    host_b_speed: float
    host_a_pitch_semitones: float
    host_b_pitch_semitones: float
    single_voice: bool         # both hosts share one voice id

    def voice_for(self, speaker: str) -> str:
        return self.host_a_voice if speaker == "host_a" else self.host_b_voice

    def speed_for(self, speaker: str) -> float:
        return self.host_a_speed if speaker == "host_a" else self.host_b_speed

    def pitch_for(self, speaker: str) -> float:
        return (
            self.host_a_pitch_semitones
            if speaker == "host_a"
            else self.host_b_pitch_semitones
        )


# ---------- backend probing ----------


@lru_cache
def _piper_module() -> object | None:
    """Import piper-tts lazily. Returns None when unavailable on the
    current Python (piper-phonemize wheels lag new CPython releases)."""
    try:
        import piper  # type: ignore[import-not-found]
        return piper
    except Exception as exc:  # pragma: no cover — environment-dependent
        logger.warning("piper-tts unavailable: %s", exc)
        return None


def _piper_voice_paths(voice_id: str) -> tuple[Path, Path]:
    s = get_settings()
    base = Path(s.piper_models_dir)
    return base / f"{voice_id}.onnx", base / f"{voice_id}.onnx.json"


def _piper_voice_present(voice_id: str) -> bool:
    onnx, cfg = _piper_voice_paths(voice_id)
    return onnx.is_file() and cfg.is_file()


def _piper_voice_download_urls(voice_id: str) -> tuple[str, str]:
    """Build HF download URLs from a voice id like 'fr_FR-siwis-medium'.

    Voice ids encode locale + name + quality, e.g. fr_FR-siwis-medium →
    fr/fr_FR/siwis/medium/<voice>.onnx
    """
    s = get_settings()
    parts = voice_id.split("-", 2)
    if len(parts) != 3:
        raise ValueError(f"unexpected piper voice id format: {voice_id!r}")
    locale, name, quality = parts
    lang = locale.split("_", 1)[0]
    base = (
        f"{s.piper_voice_url_base}/{lang}/{locale}/{name}/{quality}/"
        f"{voice_id}"
    )
    return f"{base}.onnx", f"{base}.onnx.json"


def _download_piper_voice(voice_id: str) -> bool:
    """Fetch model + config from HuggingFace. Returns True on success."""
    onnx_path, cfg_path = _piper_voice_paths(voice_id)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx_url, cfg_url = _piper_voice_download_urls(voice_id)
    try:
        for url, dest in ((onnx_url, onnx_path), (cfg_url, cfg_path)):
            if dest.exists():
                continue
            logger.info("downloading piper voice asset %s → %s", url, dest)
            tmp = dest.with_suffix(dest.suffix + ".part")
            with urllib.request.urlopen(url, timeout=120) as resp, tmp.open("wb") as f:
                shutil.copyfileobj(resp, f)
            tmp.replace(dest)
        return True
    except Exception as exc:  # pragma: no cover — network-dependent
        logger.warning("piper voice download failed for %s: %s", voice_id, exc)
        for p in (onnx_path, cfg_path):
            tmp = p.with_suffix(p.suffix + ".part")
            tmp.unlink(missing_ok=True)
        return False


def _ensure_piper_voice(voice_id: str) -> bool:
    if _piper_voice_present(voice_id):
        return True
    if not get_settings().piper_auto_download:
        return False
    return _download_piper_voice(voice_id)


@lru_cache(maxsize=8)
def _try_load_piper_voice(voice_id: str) -> object | None:
    """Load a single Piper voice. Cached so repeated synthesis is cheap."""
    piper = _piper_module()
    if piper is None:
        return None
    if not _ensure_piper_voice(voice_id):
        return None
    onnx_path, cfg_path = _piper_voice_paths(voice_id)
    try:
        return piper.PiperVoice.load(  # type: ignore[attr-defined]
            str(onnx_path), config_path=str(cfg_path)
        )
    except Exception as exc:  # pragma: no cover
        logger.warning("piper voice load failed for %s: %s", voice_id, exc)
        return None


def _piper_supports_language(lang_key: str) -> bool:
    s = get_settings()
    cfg = s.piper_language_voices.get(lang_key)
    if not cfg:
        return False
    if _piper_module() is None:
        return False
    # Allow auto-download to populate later — count the language as supported
    # as long as either the file is present or download is enabled.
    return s.piper_auto_download or all(
        _piper_voice_present(v) for v in (cfg["host_a"], cfg["host_b"])
    )


@lru_cache
def _try_load_kokoro() -> object | None:
    s = get_settings()
    try:
        from kokoro_onnx import Kokoro  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        logger.warning("kokoro-onnx import failed: %s", exc)
        return None

    model_path = Path(s.kokoro_model_path)
    voices_path = Path(s.kokoro_voices_path)
    if not model_path.exists() or not voices_path.exists():
        logger.warning(
            "kokoro model files missing (model=%s exists=%s, voices=%s exists=%s)",
            model_path, model_path.exists(), voices_path, voices_path.exists(),
        )
        return None

    try:
        return Kokoro(str(model_path), str(voices_path))
    except Exception as exc:  # pragma: no cover
        logger.warning("kokoro-onnx load failed: %s", exc)
        return None


@lru_cache
def _try_load_pyttsx3() -> object | None:
    try:
        import pyttsx3  # type: ignore[import-not-found]
        return pyttsx3.init()
    except Exception as exc:  # pragma: no cover
        logger.warning("pyttsx3 init failed: %s", exc)
        return None


# ---------- voice plan resolution ----------


def _kokoro_plan(
    lang_key: str,
    host_a_override: str | None,
    host_b_override: str | None,
) -> VoicePlan:
    s = get_settings()
    cfg = s.language_voices.get(lang_key) or s.language_voices[s.default_language]
    if cfg is s.language_voices.get(s.default_language) and lang_key not in s.language_voices:
        logger.warning(
            "unknown language %r — falling back to %s", lang_key, s.default_language
        )

    host_a_voice = host_a_override or cfg["host_a"]
    host_b_voice = host_b_override or cfg["host_b"]
    single_voice = host_a_voice == host_b_voice

    base = s.tts_speed
    if single_voice:
        host_a_speed = base + s.single_voice_speed_delta
        host_b_speed = base - s.single_voice_speed_delta
        host_a_pitch = s.single_voice_pitch_a_semitones
        host_b_pitch = s.single_voice_pitch_b_semitones
    else:
        host_a_speed = base
        host_b_speed = base
        host_a_pitch = 0.0
        host_b_pitch = 0.0

    return VoicePlan(
        backend="kokoro",
        lang=cfg["lang"],
        host_a_voice=host_a_voice,
        host_b_voice=host_b_voice,
        host_a_speed=host_a_speed,
        host_b_speed=host_b_speed,
        host_a_pitch_semitones=host_a_pitch,
        host_b_pitch_semitones=host_b_pitch,
        single_voice=single_voice,
    )


def _piper_plan(
    lang_key: str,
    host_a_override: str | None,
    host_b_override: str | None,
) -> VoicePlan:
    s = get_settings()
    cfg = s.piper_language_voices[lang_key]
    host_a_voice = host_a_override or cfg["host_a"]
    host_b_voice = host_b_override or cfg["host_b"]
    return VoicePlan(
        backend="piper",
        lang=lang_key,
        host_a_voice=host_a_voice,
        host_b_voice=host_b_voice,
        host_a_speed=s.tts_speed,
        host_b_speed=s.tts_speed,
        host_a_pitch_semitones=0.0,
        host_b_pitch_semitones=0.0,
        single_voice=host_a_voice == host_b_voice,
    )


def resolve_voice_plan(
    *,
    language: str | None = None,
    host_a_override: str | None = None,
    host_b_override: str | None = None,
) -> VoicePlan:
    """Pick voices + backend for a language.

    Probe order: Piper (best multilingual, two distinct neural voices) →
    Kokoro (fallback for langs Piper doesn't cover well) → pyttsx3 (a
    truly minimal offline backstop). Per-call voice_host_a / voice_host_b
    overrides force the matching backend's voice id.
    """
    s = get_settings()
    lang_key = (language or s.default_language).lower()

    # Override hint: Piper voice ids always look like
    # "<lang>_<COUNTRY>-<name>-<quality>" (e.g. fr_FR-siwis-medium) — they
    # have both an underscore (locale) and a hyphen (separators). Kokoro
    # voice ids are like af_heart / am_michael — underscore only, no
    # hyphen. Use that to route an override to the right backend.
    override = host_a_override or host_b_override
    override_is_piper = bool(override and "_" in override and "-" in override)

    if (
        (override_is_piper or _piper_supports_language(lang_key))
        and lang_key in s.piper_language_voices
    ):
        return _piper_plan(lang_key, host_a_override, host_b_override)

    return _kokoro_plan(lang_key, host_a_override, host_b_override)


# ---------- synthesis primitives ----------


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as w:
        frames = w.getnframes()
        rate = w.getframerate() or 1
    return int(round(frames * 1000 / rate))


def _pitch_shift_samples(samples: np.ndarray, semitones: float) -> np.ndarray:
    """Shift pitch by N semitones via linear-interpolated resampling.

    Trade-off: duration changes inversely (pitch up → shorter, pitch down
    → longer). For small shifts (a few semitones) the duration drift is
    acceptable and far less artefact-prone than a phase vocoder. Used
    only when a backend can't supply two distinct voices.
    """
    if not semitones:
        return samples
    factor = 2.0 ** (semitones / 12.0)
    src_len = len(samples)
    if src_len == 0:
        return samples
    new_length = max(1, int(round(src_len / factor)))
    src_idx = np.arange(src_len, dtype=np.float64)
    tgt_idx = np.linspace(0, src_len - 1, new_length)
    return np.interp(tgt_idx, src_idx, samples).astype(np.float32)


def _piper_synthesize_chunks(voice: object, text: str, length_scale: float) -> list:
    """Run Piper synthesis and return its audio chunks.

    piper-tts has shifted its API across versions:
      - 1.3+: synthesize(text, SynthesisConfig(length_scale=...)) → iter[AudioChunk]
      - 1.2.x: synthesize(text, length_scale=...) → iter[AudioChunk]
      - 1.1.x: synthesize(text, wav_file, length_scale=...) → None (legacy)
    We try them in order and ignore the legacy form (it's why our wav file
    came out with no header). An empty result raises so the caller can
    fall back to Kokoro.
    """
    try:
        from piper import SynthesisConfig  # type: ignore[import-not-found]
        return list(voice.synthesize(text, SynthesisConfig(length_scale=length_scale)))  # type: ignore[attr-defined]
    except (ImportError, TypeError, AttributeError):
        pass
    try:
        return list(voice.synthesize(text, length_scale=length_scale))  # type: ignore[attr-defined]
    except TypeError:
        return list(voice.synthesize(text))  # type: ignore[attr-defined]


def _synthesize_piper(
    segment: Segment,
    plan: VoicePlan,
    out_dir: Path,
) -> TTSResult | None:
    cleaned = sanitize_spoken_text(segment.text)
    if not cleaned:
        return None
    voice_id = plan.voice_for(segment.speaker)
    voice = _try_load_piper_voice(voice_id)
    if voice is None:
        raise TTSUnavailable(f"piper voice {voice_id} could not be loaded")

    out_path = out_dir / f"{uuid.uuid4().hex}.wav"
    length_scale = max(0.1, 1.0 / max(plan.speed_for(segment.speaker), 0.1))
    chunks = _piper_synthesize_chunks(voice, cleaned, length_scale)
    if not chunks:
        raise TTSUnavailable("piper produced no audio chunks")

    first = chunks[0]
    sample_rate = getattr(first, "sample_rate", 22050)
    sample_width = getattr(first, "sample_width", 2)
    sample_channels = getattr(first, "sample_channels", 1)
    with wave.open(str(out_path), "wb") as wav_file:
        wav_file.setnchannels(sample_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        for chunk in chunks:
            data = getattr(chunk, "audio_int16_bytes", None)
            if data is None and isinstance(chunk, (bytes, bytearray)):
                data = bytes(chunk)
            if data:
                wav_file.writeframes(data)

    if plan.pitch_for(segment.speaker):
        # All currently-configured Piper languages ship two distinct
        # voices, so this branch is dormant — kept for future single-voice
        # Piper languages.
        with sf.SoundFile(str(out_path), "r") as f:
            data = f.read(dtype="float32")
            sr = f.samplerate
        shifted = _pitch_shift_samples(data, plan.pitch_for(segment.speaker))
        sf.write(str(out_path), shifted, sr, subtype="PCM_16")
    return TTSResult(
        speaker=segment.speaker,
        text=cleaned,
        wav_path=str(out_path),
        duration_ms=_wav_duration_ms(out_path),
    )


def _synthesize_kokoro(
    kokoro: object,
    segment: Segment,
    plan: VoicePlan,
    out_dir: Path,
) -> TTSResult | None:
    cleaned = sanitize_spoken_text(segment.text)
    if not cleaned:
        return None

    samples, sr = kokoro.create(  # type: ignore[attr-defined]
        cleaned,
        voice=plan.voice_for(segment.speaker),
        speed=plan.speed_for(segment.speaker),
        lang=plan.lang,
    )
    arr = np.asarray(samples, dtype=np.float32)
    arr = _pitch_shift_samples(arr, plan.pitch_for(segment.speaker))
    out_path = out_dir / f"{uuid.uuid4().hex}.wav"
    sf.write(str(out_path), arr, sr, subtype="PCM_16")
    return TTSResult(
        speaker=segment.speaker,
        text=cleaned,
        wav_path=str(out_path),
        duration_ms=_wav_duration_ms(out_path),
    )


def _synthesize_pyttsx3(
    engine: object,
    segment: Segment,
    out_dir: Path,
) -> TTSResult | None:
    cleaned = sanitize_spoken_text(segment.text)
    if not cleaned:
        return None
    out_path = out_dir / f"{uuid.uuid4().hex}.wav"
    rate = 175 if segment.speaker == "host_a" else 155
    engine.setProperty("rate", rate)  # type: ignore[attr-defined]
    engine.save_to_file(cleaned, str(out_path))  # type: ignore[attr-defined]
    engine.runAndWait()  # type: ignore[attr-defined]
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise TTSUnavailable("pyttsx3 produced no audio")
    return TTSResult(
        speaker=segment.speaker,
        text=cleaned,
        wav_path=str(out_path),
        duration_ms=_wav_duration_ms(out_path),
    )


# ---------- top-level entry ----------


async def synthesize_script(
    script: PodcastScript,
    out_dir: Path | None = None,
    *,
    plan: VoicePlan | None = None,
) -> tuple[list[TTSResult], bool, VoicePlan]:
    """Synthesize every segment using the plan's backend.

    Returns (results, used_fallback, plan). `used_fallback` is True when
    we couldn't honour Piper/Kokoro and resorted to pyttsx3.
    Raises TTSUnavailable if no backend can produce audio.
    """
    out_dir = out_dir or Path(tempfile.mkdtemp(prefix="podcast_tts_"))
    out_dir.mkdir(parents=True, exist_ok=True)
    plan = plan or resolve_voice_plan()

    if plan.backend == "piper":
        try:
            results: list[TTSResult] = []
            for seg in script.segments:
                res = await asyncio.to_thread(_synthesize_piper, seg, plan, out_dir)
                if res is not None:
                    results.append(res)
            return results, False, plan
        except Exception as exc:  # noqa: BLE001 — piper-tts can raise anything
            logger.warning(
                "piper synthesis failed (%s) — falling back to Kokoro",
                exc,
                exc_info=True,
            )
            plan = _kokoro_plan(plan.lang, None, None)

    if plan.backend == "kokoro":
        kokoro = _try_load_kokoro()
        if kokoro is not None:
            results = []
            for seg in script.segments:
                res = await asyncio.to_thread(
                    _synthesize_kokoro, kokoro, seg, plan, out_dir
                )
                if res is not None:
                    results.append(res)
            return results, False, plan

    engine = _try_load_pyttsx3()
    if engine is not None:
        # Switch the plan's backend label so callers / SSE see the truth.
        fallback_plan = VoicePlan(
            backend="pyttsx3",
            lang=plan.lang,
            host_a_voice="pyttsx3-default",
            host_b_voice="pyttsx3-default",
            host_a_speed=plan.host_a_speed,
            host_b_speed=plan.host_b_speed,
            host_a_pitch_semitones=0.0,
            host_b_pitch_semitones=0.0,
            single_voice=True,
        )
        results = []
        for seg in script.segments:
            res = await asyncio.to_thread(
                _synthesize_pyttsx3, engine, seg, out_dir
            )
            if res is not None:
                results.append(res)
        return results, True, fallback_plan

    raise TTSUnavailable("no TTS backend available (piper, kokoro, pyttsx3 all failed)")
