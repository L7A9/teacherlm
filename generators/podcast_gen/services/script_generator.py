from __future__ import annotations

import logging

from teacherlm_core.schemas.chunk import Chunk

from ..config import get_settings
from ..schemas import DurationPreset, NarrativeArc, PodcastScript, Segment
from .llm_service import LLMService, build_system_prompt
from .narrative_extractor import format_context_for_speech
from .text_sanitizer import sanitize_spoken_text


logger = logging.getLogger(__name__)


LANGUAGE_HINTS = {
    "en-us": "Write in English (US).",
    "en-gb": "Write in English (UK).",
    "fr-fr": "Écris en français.",
    "es": "Escribe en español.",
    "it": "Scrivi in italiano.",
    "pt-br": "Escreva em português (Brasil).",
    "de": "Schreib auf Deutsch.",
    "ja": "日本語で書いてください。",
    "cmn": "请用普通话书写。",
    "hi": "हिन्दी में लिखें।",
}


# Map our language tags to langdetect's two-letter ISO 639-1 codes so the
# post-pass can spot English filler in a French script (etc).
_LANGDETECT_TARGET = {
    "en-us": "en",
    "en-gb": "en",
    "fr-fr": "fr",
    "es": "es",
    "it": "it",
    "pt-br": "pt",
    "de": "de",
    "ja": "ja",
    "cmn": "zh-cn",
    "hi": "hi",
}


def _format_key_points(points: list[str]) -> str:
    return "\n".join(f"  {i + 1}. {p}" for i, p in enumerate(points))


def resolve_word_target(duration: DurationPreset | str) -> int:
    s = get_settings()
    targets = s.duration_word_targets
    if duration in targets:
        return targets[duration]
    return targets[s.default_duration]


def _word_count(script: PodcastScript) -> int:
    return sum(len(seg.text.split()) for seg in script.segments)


def _language_hint(language: str | None) -> str:
    if not language:
        return ""
    return LANGUAGE_HINTS.get(language.lower(), "")


# Public alias — pipeline.py calls this to share the same hint with the
# narrative arc extractor so both LLM calls agree on the target language.
def language_hint(language: str | None) -> str:
    return _language_hint(language)


def _host_identity_block(
    host_a_name: str | None,
    host_b_name: str | None,
) -> str:
    """Block of identity rules inserted into the script prompt.

    AI hosts can't truthfully claim a human name. We default to no names —
    the script must NOT say "I'm Marie", "[prénom]", or any placeholder.
    If the user supplies names via settings or options, we tell the LLM to
    use them naturally.
    """
    a = (host_a_name or "").strip()
    b = (host_b_name or "").strip()
    if not a and not b:
        return (
            "You are AI hosts. You do NOT have human names. NEVER introduce\n"
            "yourselves by name. NEVER write placeholders like [prénom],\n"
            "[name], [your name], [host name], or any bracketed slot for a\n"
            "name. Open with a name-less greeting — for example: 'Welcome to\n"
            "today's episode', 'Hello and welcome', 'Bonjour et bienvenue à\n"
            "cet épisode' — and refer to each other by role only ('my\n"
            "co-host', 'our teacher today', or no reference at all)."
        )
    parts = ["The hosts have these names — use them naturally in the intro,"
            " but never as bracketed placeholders:"]
    if a:
        parts.append(f"  - host_a is named: {a}")
    if b:
        parts.append(f"  - host_b is named: {b}")
    parts.append(
        "Do NOT make up additional names. Do NOT use placeholders like\n"
        "[prénom] or [name] under any circumstance."
    )
    return "\n".join(parts)


async def generate_script(
    arc: NarrativeArc,
    chunks: list[Chunk],
    *,
    duration: DurationPreset | str,
    language: str | None,
    host_a_name: str | None,
    host_b_name: str | None,
    llm: LLMService,
) -> PodcastScript:
    """Generate a two-host script from the narrative arc."""
    target = resolve_word_target(duration)
    min_words = int(target * 0.75)
    max_words = int(target * 1.25)

    system = build_system_prompt(
        "script_educational.txt",
        title=arc.title,
        intro=arc.intro,
        key_points_block=_format_key_points(arc.key_points),
        conclusion=arc.conclusion,
        context_block=format_context_for_speech(chunks),
        target_words=target,
        min_words=min_words,
        max_words=max_words,
        language_hint=_language_hint(language),
        host_identity_block=_host_identity_block(host_a_name, host_b_name),
    )
    script = await llm.generate_structured(
        system=system,
        user_message=(
            f"Write the full podcast script now, ~{target} words total. "
            "host_a opens, host_b carries the explanations, alternating "
            "naturally. Return JSON only, matching the PodcastScript schema."
        ),
        schema=PodcastScript,
    )

    if not script.segments:
        script.segments = [
            Segment(
                speaker="host_a",
                text=arc.intro or "Welcome to today's episode.",
            ),
            Segment(
                speaker="host_b",
                text=" ".join(arc.key_points) or "Let's dig in.",
            ),
        ]
    speakers = {seg.speaker for seg in script.segments}
    if len(speakers) == 1 and len(script.segments) >= 2:
        only = next(iter(speakers))
        other: str = "host_b" if only == "host_a" else "host_a"
        script.segments[1].speaker = other  # type: ignore[assignment]

    # Strip any technical labels the LLM leaked into spoken text and split
    # very long monologues into paced ~80-word turns.
    safe_segments: list[Segment] = []
    for seg in script.segments:
        cleaned = sanitize_spoken_text(seg.text)
        if not cleaned:
            continue
        # The model may set source_chunk_id to a fake "Source N" label —
        # keep it only if it actually matches a real chunk_id from the input.
        valid_chunk_ids = {c.chunk_id for c in chunks}
        chunk_ref = (
            seg.source_chunk_id
            if seg.source_chunk_id in valid_chunk_ids
            else None
        )
        words = cleaned.split()
        if len(words) <= 120:
            safe_segments.append(
                Segment(speaker=seg.speaker, text=cleaned, source_chunk_id=chunk_ref)
            )
            continue
        for i in range(0, len(words), 80):
            piece = " ".join(words[i : i + 80]).strip()
            if piece:
                safe_segments.append(
                    Segment(
                        speaker=seg.speaker,
                        text=piece,
                        source_chunk_id=chunk_ref if i == 0 else None,
                    )
                )
    script.segments = safe_segments

    return script


def script_word_count(script: PodcastScript) -> int:
    return _word_count(script)


# ---------- language enforcement ----------


def _detect_language(text: str) -> str | None:
    """Return ISO 639-1 code, or None when langdetect can't decide."""
    try:
        from langdetect import DetectorFactory, detect  # type: ignore[import-not-found]
    except Exception:
        return None
    DetectorFactory.seed = 0  # deterministic across runs
    try:
        return detect(text)
    except Exception:
        return None


async def _retranslate_segment(
    segment: Segment,
    *,
    target_language: str,
    target_hint: str,
    llm: LLMService,
) -> Segment:
    """Ask the LLM to rewrite one segment entirely in the target language."""
    system = (
        "You are a translator that rewrites podcast lines so every word is "
        f"in the target language: {target_language}. {target_hint} "
        "Keep the speaker's tone and meaning. Preserve proper nouns and "
        "technical terms only when they have no native equivalent. Output "
        "ONLY the rewritten line — no explanations, no quotes, no labels."
    )
    rewritten = await llm.reply(system=system, user_message=segment.text)
    cleaned = sanitize_spoken_text(rewritten)
    return Segment(
        speaker=segment.speaker,
        text=cleaned or segment.text,
        source_chunk_id=segment.source_chunk_id,
    )


async def enforce_language(
    script: PodcastScript,
    *,
    language: str | None,
    llm: LLMService,
) -> PodcastScript:
    """Detect any segment whose language drifts from the user's selection
    and rewrite it. No-op when langdetect can't load (degrades to prompt-
    only enforcement) or the target language is unknown."""
    if not language:
        return script
    target = _LANGDETECT_TARGET.get(language.lower())
    if target is None:
        return script
    target_hint = _language_hint(language) or f"Write in {language}."
    fixed: list[Segment] = []
    for seg in script.segments:
        # Very short turns ("oui", "ok") confuse langdetect — don't touch
        # them.
        if len(seg.text.split()) < 4:
            fixed.append(seg)
            continue
        detected = _detect_language(seg.text)
        if detected and detected != target:
            logger.info(
                "language drift: segment detected as %r, expected %r — retranslating",
                detected, target,
            )
            try:
                fixed.append(
                    await _retranslate_segment(
                        seg,
                        target_language=language,
                        target_hint=target_hint,
                        llm=llm,
                    )
                )
                continue
            except Exception:
                logger.exception("retranslation failed; keeping original")
        fixed.append(seg)
    script.segments = fixed
    return script
