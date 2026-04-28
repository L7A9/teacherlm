"""Strip machine-readable metadata that the LLM may accidentally include in
spoken text — chunk_ids, source labels, bracketed refs, etc. We do this so
the listener never hears "according to chunk_id abc123" or "Source 3 says".

This is a safety net: the prompt also instructs the LLM to use natural
phrasing only, but LLMs slip and a regex pass is cheap.
"""

from __future__ import annotations

import re


# `[chunk_id=abc def=ghi]`, `[source=foo.pdf]`, `[score: 0.91]` — anything
# that looks like a key=val pair inside square brackets, single or comma-
# separated. Generous matching since the LLM may invent variants.
_BRACKET_META = re.compile(
    r"\[\s*(?:chunk_?id|source(?:_chunk_?id)?|score|metadata|id|ref|reference)"
    r"\s*[:=][^\]]*\]",
    re.IGNORECASE,
)

# `[Source 3]`, `[Reference 2]`, `[Doc 1]`, `(Source 4)` — numbered citations
# that read awkwardly aloud.
_NUMBERED_REF = re.compile(
    r"[\[\(]\s*(?:source|reference|ref|doc(?:ument)?|excerpt|chunk)\s+\d+\s*[\]\)]",
    re.IGNORECASE,
)

# `[prénom]`, `[name]`, `[your name]`, `[host name]`, `[insert ...]`,
# `[nom]`, `[nombre]`, `[名前]` — placeholder names the LLM drops in for the
# AI hosts to "fill in". An AI doesn't have a name; we strip these entirely
# and the surrounding sentence has to stand on its own.
_NAME_PLACEHOLDER = re.compile(
    r"\[\s*(?:"
    r"pr[ée]nom|name|nom(?:bre)?|your\s+name|host\s+name|"
    r"insert\s*[^\]]*|name\s*here|votre\s*nom|名前"
    r")\s*\]",
    re.IGNORECASE,
)

# Inline `chunk_id=abc123`, `chunk_id: abc123`, `source_chunk_id abc123` —
# without brackets. We require the key be followed by `=`, `:`, or whitespace
# + an identifier-looking token so we don't eat ordinary prose.
_INLINE_META = re.compile(
    r"\b(?:chunk_?id|source_chunk_?id)\s*[:=]?\s*[\w\-./]+",
    re.IGNORECASE,
)

# `according to source 3`, `as source 2 says`, `from source number 4` —
# a soft pattern; the LLM tends to fall back to these even when told not to.
_PROSE_REF = re.compile(
    r"\b(?:according to|as|per|from|in|see)\s+(?:source|chunk|reference|excerpt|doc(?:ument)?)\s+(?:number\s+)?\d+\b",
    re.IGNORECASE,
)

# `according to chunk_id=abc123`, `from source_chunk_id xyz` — the inline-
# metadata variant of _PROSE_REF. We swap the whole prepositional phrase
# for "the material" so the surrounding sentence still flows.
_PROSE_REF_META = re.compile(
    r"\b(?:according to|as|per|from|in|see)\s+(?:chunk_?id|source_chunk_?id)\s*[:=]?\s*[\w\-./]+",
    re.IGNORECASE,
)

# Collapse runs of whitespace and stray punctuation left over after stripping.
_DANGLING_PUNCT = re.compile(r"\s+([,.;:!?])")
_MULTI_SPACE = re.compile(r"\s{2,}")
_EMPTY_BRACKETS = re.compile(r"\(\s*\)|\[\s*\]")


def sanitize_spoken_text(text: str) -> str:
    """Remove technical labels that should never be read aloud.

    Idempotent. Safe to call on already-clean text.
    """
    if not text:
        return text
    out = text
    out = _BRACKET_META.sub("", out)
    out = _NUMBERED_REF.sub("", out)
    out = _NAME_PLACEHOLDER.sub("", out)
    # Swap "according to chunk_id=foo" / "according to source 3" for a
    # natural phrase BEFORE stripping bare inline metadata, so we don't
    # leave dangling "according to," fragments.
    out = _PROSE_REF_META.sub("the material", out)
    out = _PROSE_REF.sub("the material", out)
    out = _INLINE_META.sub("", out)
    out = _EMPTY_BRACKETS.sub("", out)
    out = _DANGLING_PUNCT.sub(r"\1", out)
    out = _MULTI_SPACE.sub(" ", out)
    return out.strip()
