from __future__ import annotations

import logging

from teacherlm_core.schemas.chunk import Chunk

from ..config import get_settings
from ..schemas import (
    DurationPreset,
    NarrativeArc,
    PodcastScript,
    ScriptSection,
    Segment,
)
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


def _section_directive(
    section_type: str,
    *,
    arc: NarrativeArc,
    key_point: str | None,
    point_idx: int | None,
    total_points: int,
) -> str:
    if section_type == "intro":
        return (
            f"Write the OPENING of the podcast. host_a welcomes the listener "
            f"warmly and previews the topic ('{arc.title}'); host_b adds brief "
            f"framing of what they'll cover. Keep it natural and conversational. "
            f"Do NOT start covering the key points — that's for later sections."
        )
    if section_type == "outro":
        return (
            f"Write the CLOSING of the podcast. host_a reflects in one or two "
            f"sentences on what they learned across the episode. host_b gives "
            f"a warm, encouraging sign-off. Do NOT introduce new content."
        )
    # keypoint
    return (
        f"Cover ONLY this key point in this section: \"{key_point}\". "
        f"This is point {point_idx} of {total_points} in the podcast. "
        f"host_a opens with a curious, student-style question or observation "
        f"about this point; host_b answers with a clear explanation grounded "
        f"in the source excerpts, including at least one concrete example or "
        f"analogy; then host_a follows up with a clarifying question or "
        f"reaction; host_b deepens with another angle. Aim for 4-6 "
        f"back-and-forth exchanges (8-12 segments) on this point alone. "
        f"Do NOT introduce other key points and do NOT recap the intro."
    )


async def _generate_section(
    *,
    section_type: str,
    arc: NarrativeArc,
    chunks: list[Chunk],
    key_point: str | None,
    point_idx: int | None,
    total_points: int,
    target_words: int,
    duration: DurationPreset | str,
    language: str | None,
    host_a_name: str | None,
    host_b_name: str | None,
    llm: LLMService,
) -> list[Segment]:
    """Generate the segments for one section of the podcast.

    Per-section generation is reliable on small models because each call
    targets a bounded word count (150-400 words typically) that the model
    can actually produce — unlike a 1400-word whole-script call which the
    model compresses into a near-empty response in JSON mode.
    """
    settings = get_settings()
    min_words = max(50, int(target_words * 0.8))
    max_words = int(target_words * 1.3)

    system = build_system_prompt(
        "script_section.txt",
        title=arc.title,
        intro=arc.intro,
        key_points_block=_format_key_points(arc.key_points),
        conclusion=arc.conclusion,
        context_block=format_context_for_speech(chunks),
        target_words=target_words,
        min_words=min_words,
        max_words=max_words,
        language_hint=_language_hint(language),
        host_identity_block=_host_identity_block(host_a_name, host_b_name),
        section_directive=_section_directive(
            section_type,
            arc=arc,
            key_point=key_point,
            point_idx=point_idx,
            total_points=total_points,
        ),
    )

    options = {
        # 3x word target with a 1024 floor handles JSON overhead and non-Latin
        # scripts safely. Section calls are small so this is cheap.
        "num_predict": max(1024, target_words * 3),
        "temperature": settings.generation_temperature,
    }

    user_msg = (
        f"Generate ONLY this section now. The combined word count across "
        f"all segments in this section MUST be between {min_words} and "
        f"{max_words} words (target ~{target_words}). Return JSON only, "
        f"matching the schema with a 'segments' array."
    )

    try:
        section = await llm.generate_structured(
            system=system,
            user_message=user_msg,
            schema=ScriptSection,
            options=options,
        )
        if section.segments:
            return section.segments
    except Exception:
        logger.exception(
            "section generation failed (type=%s, idx=%s) — using fallback",
            section_type,
            point_idx,
        )

    # Fallback so the podcast still has SOMETHING for this section if the
    # LLM call fails or returns empty. Better a thin section than missing one.
    if section_type == "intro":
        return [
            Segment(speaker="host_a", text=f"Welcome — today we're talking about {arc.title}."),
            Segment(speaker="host_b", text=arc.intro or "Let's dig in."),
        ]
    if section_type == "outro":
        return [
            Segment(speaker="host_a", text="That was a lot — thanks for walking through it with me."),
            Segment(speaker="host_b", text=arc.conclusion or "Thanks for listening."),
        ]
    point_text = key_point or "this topic"
    return [
        Segment(speaker="host_a", text=f"What about {point_text}?"),
        Segment(speaker="host_b", text=f"{point_text} — the materials cover this; let's go through it."),
    ]


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
    """Generate a two-host script section-by-section.

    Earlier versions asked the LLM for the WHOLE script in one structured-
    output call. With small models (8B/Q4) this collapses to a near-empty
    response that satisfies the JSON schema, producing 2-3 minute audio
    when the user picked 9 minutes. Per-section generation gives each LLM
    call a bounded target it can actually deliver, then we concatenate.
    """
    target = resolve_word_target(duration)
    n_points = max(1, len(arc.key_points))

    # Word budget: ~10% intro, ~8% outro, the rest split across key points.
    intro_target = max(80, target // 10)
    outro_target = max(70, target // 12)
    keypoints_total = max(target - intro_target - outro_target, n_points * 100)
    per_point_target = max(150, keypoints_total // n_points)

    all_segments: list[Segment] = []

    # --- intro ---
    intro_segs = await _generate_section(
        section_type="intro",
        arc=arc,
        chunks=chunks,
        key_point=None,
        point_idx=None,
        total_points=n_points,
        target_words=intro_target,
        duration=duration,
        language=language,
        host_a_name=host_a_name,
        host_b_name=host_b_name,
        llm=llm,
    )
    all_segments.extend(intro_segs)

    # --- one section per key point ---
    for i, point in enumerate(arc.key_points, start=1):
        point_segs = await _generate_section(
            section_type="keypoint",
            arc=arc,
            chunks=chunks,
            key_point=point,
            point_idx=i,
            total_points=n_points,
            target_words=per_point_target,
            duration=duration,
            language=language,
            host_a_name=host_a_name,
            host_b_name=host_b_name,
            llm=llm,
        )
        all_segments.extend(point_segs)

    # --- outro ---
    outro_segs = await _generate_section(
        section_type="outro",
        arc=arc,
        chunks=chunks,
        key_point=None,
        point_idx=None,
        total_points=n_points,
        target_words=outro_target,
        duration=duration,
        language=language,
        host_a_name=host_a_name,
        host_b_name=host_b_name,
        llm=llm,
    )
    all_segments.extend(outro_segs)

    script = PodcastScript(
        title=arc.title,
        summary=arc.intro or arc.title,
        segments=all_segments,
    )
    logger.info(
        "podcast script assembled: %d segments, %d words (target %d, duration=%s)",
        len(script.segments),
        _word_count(script),
        target,
        duration,
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
