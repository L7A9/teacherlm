"""Shared language-enforcement directive used by every generator.

The frontend's Settings page can force a language across all generators.
That language flows in through GeneratorInput.options['language'] — each
generator's pipeline calls set_current_language() at the start of run()
and the OllamaClient transparently appends language_directive() to the
system message of every LLM call. This keeps the wiring to one line per
generator instead of edits at every prompt site.
"""

from __future__ import annotations

import contextvars


_LANGUAGE_NAMES: dict[str, str] = {
    "en-us": "English (US)",
    "en-gb": "English (UK)",
    "fr-fr": "French",
    "es": "Spanish",
    "it": "Italian",
    "pt-br": "Portuguese (Brazil)",
    "de": "German",
    "ja": "Japanese",
    "cmn": "Mandarin Chinese",
    "hi": "Hindi",
}


def language_name(code: str | None) -> str | None:
    if not code:
        return None
    return _LANGUAGE_NAMES.get(code.lower())


_current_language: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "teacherlm_current_language", default=None
)


def set_current_language(code: str | None) -> None:
    """Set the forced language for every LLM call inside the current
    async context (typically a single FastAPI request). Pass None to
    clear. Each generator's pipeline calls this at the start of run()."""
    _current_language.set(code or None)


def get_current_language() -> str | None:
    return _current_language.get()


def inject_language_directive(messages: list[dict]) -> list[dict]:
    """If a language is set in the current context, append the directive
    to the first system message (or insert a system message if none).
    Otherwise return messages unchanged."""
    code = get_current_language()
    if not code:
        return messages
    directive = language_directive(code)
    if not directive:
        return messages
    out: list[dict] = []
    injected = False
    for msg in messages:
        if not injected and msg.get("role") == "system":
            out.append({**msg, "content": (msg.get("content") or "") + directive})
            injected = True
        else:
            out.append(msg)
    if not injected:
        out.insert(0, {"role": "system", "content": directive.strip()})
    return out


def language_directive(code: str | None) -> str:
    """Return a strict directive to append to a system prompt, or '' if
    the user hasn't forced a language.

    The directive is intentionally absolute: 'translate from source if
    needed', no English filler, etc. Generators that already had soft
    'match the source language' rules in their prompts should still
    append this — it overrides them.
    """
    name = language_name(code)
    if not name:
        return ""
    return (
        f"\n\n---\nLANGUAGE — ABSOLUTE REQUIREMENT\n"
        f"Write your entire response in {name}. Every word — including "
        f"headings, labels, code comments, and any explanatory text — "
        f"must be in {name}. If the source material is in another "
        f"language, translate the ideas as you go; do not leave anything "
        f"in the source language. Do not mix languages. Proper nouns and "
        f"technical terms with no native equivalent may stay in the "
        f"source language; everything connecting them must be in "
        f"{name}. This rule overrides any earlier 'match the source "
        f"language' instructions.\n"
    )
