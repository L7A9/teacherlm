from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from pydub import AudioSegment
from pydub.effects import normalize

from ..config import get_settings
from ..schemas import PodcastScript, TTSResult


logger = logging.getLogger(__name__)


def _silence(ms: int) -> AudioSegment:
    return AudioSegment.silent(duration=ms)


def _format_speaker(speaker: str) -> str:
    return "Host A" if speaker == "host_a" else "Host B"


def build_transcript(script: PodcastScript) -> str:
    """Plain-text transcript with speaker labels — paired with the MP3."""
    lines: list[str] = [script.title.strip(), ""]
    if script.summary.strip():
        lines.append(script.summary.strip())
        lines.append("")
    lines.append("---")
    lines.append("")
    for seg in script.segments:
        lines.append(f"{_format_speaker(seg.speaker)}: {seg.text.strip()}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def compose_audio(results: list[TTSResult]) -> tuple[bytes, int]:
    """Concatenate per-segment wavs with inter-segment silence + normalize.

    Returns (mp3_bytes, total_duration_ms). Requires ffmpeg in PATH
    (the Dockerfile installs it).
    """
    s = get_settings()
    if not results:
        # Edge case: no successful segments. Emit ~1s of silence so the
        # downstream pipeline still has something to upload.
        empty = _silence(1000)
        out_path = Path(tempfile.mktemp(suffix=".mp3"))
        empty.export(out_path, format="mp3", bitrate=s.mp3_bitrate)
        try:
            return out_path.read_bytes(), len(empty)
        finally:
            try:
                out_path.unlink()
            except OSError:
                pass

    track = _silence(s.intro_outro_silence_ms)
    gap = _silence(s.inter_segment_silence_ms)

    for i, res in enumerate(results):
        try:
            seg_audio = AudioSegment.from_wav(res.wav_path)
        except Exception as exc:
            logger.warning("skipping unreadable segment %s: %s", res.wav_path, exc)
            continue
        if i > 0:
            track += gap
        track += seg_audio

    track += _silence(s.intro_outro_silence_ms)
    track = normalize(track)

    out_path = Path(tempfile.mktemp(suffix=".mp3"))
    try:
        track.export(out_path, format="mp3", bitrate=s.mp3_bitrate)
        mp3_bytes = out_path.read_bytes()
    finally:
        try:
            out_path.unlink()
        except OSError:
            pass

    return mp3_bytes, len(track)
