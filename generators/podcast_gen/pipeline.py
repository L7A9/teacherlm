from __future__ import annotations

import json
import logging
import shutil
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from teacherlm_core.llm.language import set_current_language
from teacherlm_core.llm.runtime import set_current_llm_options
from teacherlm_core.schemas.generator_io import (
    GeneratorArtifact,
    GeneratorInput,
    GeneratorOutput,
    LearnerUpdates,
)

from .config import get_settings
from .schemas import PodcastBundle, PodcastScript
from .services.artifact_store import get_artifact_store
from .services.audio_composer import build_transcript, compose_audio
from .services.llm_service import get_llm_service
from .services.narrative_extractor import extract_narrative_arc
from .services.script_generator import (
    enforce_language,
    generate_script,
    language_hint,
    script_word_count,
)
from .services.tts_service import (
    TTSUnavailable,
    VoicePlan,
    resolve_voice_plan,
    synthesize_script,
)


logger = logging.getLogger(__name__)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _resolve_duration(options: dict) -> str:
    s = get_settings()
    raw = (options.get("duration") or s.default_duration).lower()
    return raw if raw in s.duration_word_targets else s.default_duration


def _resolve_topic(options: dict, user_message: str) -> str:
    if topic := options.get("topic"):
        return str(topic).strip()
    return (user_message or "").strip()


def _resolve_language(options: dict) -> str:
    s = get_settings()
    raw = (options.get("language") or options.get("lang") or s.default_language).lower()
    supported = set(s.language_voices) | set(s.piper_language_voices)
    return raw if raw in supported else s.default_language


def _resolve_host_names(options: dict) -> tuple[str | None, str | None]:
    """Per-call > settings default > None (use no-name introduction)."""
    s = get_settings()
    a = options.get("host_a_name")
    b = options.get("host_b_name")
    a = a.strip() if isinstance(a, str) and a.strip() else s.host_a_name
    b = b.strip() if isinstance(b, str) and b.strip() else s.host_b_name
    return a, b


def _safe_filename(title: str) -> str:
    cleaned = "".join(
        c if c.isalnum() or c in ("-", "_", " ") else "_" for c in title
    ).strip().replace(" ", "_")
    return (cleaned[:60] or "podcast").lower()


def _intro_message(
    bundle: PodcastBundle,
    duration_choice: str,
    plan: VoicePlan | None,
) -> str:
    if bundle.tts_skipped:
        return (
            f"I drafted a {bundle.word_count}-word two-host script for "
            f"\"{bundle.title}\" — but text-to-speech isn't available in this "
            "environment, so I've attached the transcript only."
        )
    minutes = round(bundle.duration_ms / 60000, 1)
    notes: list[str] = []
    if bundle.used_fallback_tts:
        notes.append(
            "using the offline pyttsx3 fallback voice — install the Piper "
            "or Kokoro models for richer audio"
        )
    if plan and plan.single_voice and not bundle.used_fallback_tts:
        notes.append(
            f"this language ({plan.lang}) only ships one {plan.backend} "
            "voice, so both hosts share it with a pitch and speed shift to "
            "make the two speakers clearly distinguishable"
        )
    note_block = f" ({'; '.join(notes)})" if notes else ""
    return (
        f"Here's your {duration_choice} podcast on \"{bundle.title}\" — "
        f"about {minutes} minutes, two hosts walking through {bundle.segment_count} "
        f"segments together{note_block}. The transcript is attached too if you "
        "want to follow along."
    )


def _empty_response(reason: str) -> str:
    return (
        "I couldn't pull enough material from your sources to script a "
        f"podcast ({reason}). Upload more course files or chat with me about "
        "the topic first, then try again."
    )


async def run(inp: GeneratorInput) -> AsyncIterator[str]:
    settings = get_settings()
    options = dict(inp.options or {})
    set_current_llm_options(options)
    llm = get_llm_service()
    duration = _resolve_duration(options)
    topic = _resolve_topic(options, inp.user_message)
    language = _resolve_language(options)
    set_current_language(language)
    host_a_name, host_b_name = _resolve_host_names(options)
    plan = resolve_voice_plan(
        language=language,
        host_a_override=options.get("voice_host_a"),
        host_b_override=options.get("voice_host_b"),
    )

    yield _sse(
        "progress",
        {
            "stage": "starting",
            "chunks": len(inp.context_chunks),
            "duration": duration,
            "language": language,
            "backend": plan.backend,
            "voices": [plan.host_a_voice, plan.host_b_voice],
            "single_voice": plan.single_voice,
        },
    )

    if not inp.context_chunks:
        msg = _empty_response("no context chunks were retrieved")
        yield _sse("token", {"delta": msg})
        yield _sse(
            "done",
            GeneratorOutput(
                response=msg,
                generator_id=settings.generator_id,
                output_type=settings.output_type,
                sources=[],
                metadata={"reason": "no_context"},
            ).model_dump(),
        )
        return

    yield _sse("progress", {"stage": "extracting_arc"})
    lang_hint = language_hint(language)
    arc = await extract_narrative_arc(
        inp.context_chunks,
        topic_focus=topic,
        language_hint=lang_hint,
        llm=llm,
    )
    yield _sse(
        "progress",
        {
            "stage": "arc_ready",
            "title": arc.title,
            "key_points": len(arc.key_points),
        },
    )

    yield _sse("progress", {"stage": "scripting", "duration": duration})
    script: PodcastScript = await generate_script(
        arc,
        inp.context_chunks,
        duration=duration,
        language=language,
        host_a_name=host_a_name,
        host_b_name=host_b_name,
        llm=llm,
    )
    yield _sse("progress", {"stage": "language_check", "language": language})
    script = await enforce_language(script, language=language, llm=llm)
    word_count = script_word_count(script)
    yield _sse(
        "progress",
        {
            "stage": "scripted",
            "segments": len(script.segments),
            "word_count": word_count,
        },
    )

    transcript = build_transcript(script)
    base_name = _safe_filename(script.title)
    store = get_artifact_store()
    artifacts: list[GeneratorArtifact] = []

    # ---------- TTS + audio composition ----------
    used_fallback = False
    tts_skipped = False
    duration_ms = 0
    work_dir = Path(tempfile.mkdtemp(prefix="podcast_seg_"))
    try:
        try:
            yield _sse("progress", {"stage": "synthesizing"})
            tts_results, used_fallback, plan = await synthesize_script(
                script, work_dir, plan=plan
            )
            yield _sse(
                "progress",
                {"stage": "synthesized", "segments": len(tts_results)},
            )

            yield _sse("progress", {"stage": "composing"})
            mp3_bytes, duration_ms = compose_audio(tts_results)
            mp3_filename = f"{base_name}.mp3"
            mp3_key, mp3_url = await store.save(
                conversation_id=inp.conversation_id,
                filename=mp3_filename,
                payload=mp3_bytes,
                content_type="audio/mpeg",
            )
            artifacts.append(
                GeneratorArtifact(
                    type="audio", url=mp3_url, filename=mp3_filename, key=mp3_key
                )
            )
            yield _sse(
                "progress",
                {"stage": "composed", "duration_ms": duration_ms},
            )
        except TTSUnavailable as exc:
            logger.warning("TTS unavailable, emitting transcript only: %s", exc)
            tts_skipped = True
            yield _sse("progress", {"stage": "tts_skipped", "reason": str(exc)})
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    # ---------- Transcript artifact (always uploaded) ----------
    transcript_filename = f"{base_name}.txt"
    transcript_key, transcript_url = await store.save(
        conversation_id=inp.conversation_id,
        filename=transcript_filename,
        payload=transcript.encode("utf-8"),
        content_type="text/plain; charset=utf-8",
    )
    artifacts.append(
        GeneratorArtifact(
            type="transcript",
            url=transcript_url,
            filename=transcript_filename,
            key=transcript_key,
        )
    )

    bundle = PodcastBundle(
        title=script.title,
        summary=script.summary,
        duration_ms=duration_ms,
        word_count=word_count,
        segment_count=len(script.segments),
        transcript=transcript,
        used_fallback_tts=used_fallback,
        tts_skipped=tts_skipped,
    )

    response_text = _intro_message(bundle, duration, plan)
    yield _sse("token", {"delta": response_text})

    concepts_covered = [p for p in arc.key_points if p]

    output = GeneratorOutput(
        response=response_text,
        generator_id=settings.generator_id,
        output_type=settings.output_type,
        artifacts=artifacts,
        sources=inp.context_chunks,
        learner_updates=LearnerUpdates(concepts_covered=concepts_covered),
        metadata={
            "podcast": bundle.model_dump(),
            "narrative_arc": arc.model_dump(),
            "duration_choice": duration,
            "language": language,
            "voices": {
                "backend": plan.backend,
                "host_a": plan.host_a_voice,
                "host_b": plan.host_b_voice,
                "lang_code": plan.lang,
                "single_voice": plan.single_voice,
                "host_a_pitch_semitones": plan.host_a_pitch_semitones,
                "host_b_pitch_semitones": plan.host_b_pitch_semitones,
            },
            "host_names": {"host_a": host_a_name, "host_b": host_b_name},
        },
    )
    yield _sse("done", output.model_dump())
