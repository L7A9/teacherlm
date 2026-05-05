from typing import Literal

from pydantic import BaseModel, Field


SpeakerRole = Literal["host_a", "host_b"]
DurationPreset = Literal["short", "medium", "long"]


class NarrativeArc(BaseModel):
    """Output of narrative_extractor — what the script will cover."""

    title: str
    intro: str
    key_points: list[str] = Field(default_factory=list)
    conclusion: str
    sources: list[str] = Field(default_factory=list)  # chunk_ids cited


class Segment(BaseModel):
    """One spoken line in the podcast."""

    speaker: SpeakerRole
    text: str
    # Optional source chunk_id — the teacher cites material; the student doesn't.
    source_chunk_id: str | None = None


class PodcastScript(BaseModel):
    """Top-level script object the LLM emits with ollama format=."""

    title: str
    summary: str
    segments: list[Segment]


class ScriptSection(BaseModel):
    """A single section's segments — used by per-section script generation
    so each LLM call is bounded and the model is unable to satisfy the
    schema with a near-empty response (the failure mode of small models in
    JSON-structured mode for whole-script generation)."""

    segments: list[Segment]


class TTSResult(BaseModel):
    """Per-segment TTS output handed to the audio composer."""

    speaker: SpeakerRole
    text: str
    # Path on local disk to the rendered wav (kept tiny + temporary).
    wav_path: str
    duration_ms: int


class PodcastBundle(BaseModel):
    """Final composed podcast metadata returned in GeneratorOutput.metadata."""

    title: str
    summary: str
    duration_ms: int
    word_count: int
    segment_count: int
    transcript: str
    used_fallback_tts: bool = False
    tts_skipped: bool = False  # true if both kokoro and pyttsx3 unavailable
