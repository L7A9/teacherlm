from __future__ import annotations

import re

from ..schemas import ClozeCard, PrioritizedConcept


def _escape_for_regex(term: str) -> str:
    # Allow matching the term with flexible internal whitespace, case-insensitively.
    parts = [re.escape(p) for p in term.split() if p]
    return r"\s+".join(parts) if parts else re.escape(term)


def _build_cloze(sentence: str, term: str) -> tuple[str, str] | None:
    """Wrap the first occurrence of `term` in the sentence with Anki cloze
    markers. Returns (cloze_text, answer_text) or None if `term` isn't found.
    """
    pattern = re.compile(_escape_for_regex(term), re.IGNORECASE)
    match = pattern.search(sentence)
    if not match:
        return None
    matched = sentence[match.start():match.end()]
    cloze = f"{sentence[:match.start()]}{{{{c1::{matched}}}}}{sentence[match.end():]}"
    return cloze, matched


def generate_cloze_cards(
    prioritized: list[PrioritizedConcept],
    *,
    limit: int | None = None,
) -> list[ClozeCard]:
    """Generate one cloze card per prioritized concept (no LLM).

    Skips concepts whose context sentence doesn't contain the term (rare — the
    miner attaches the sentence that surfaced it — but possible after
    whitespace/punctuation normalization).
    """
    out: list[ClozeCard] = []
    for prio in prioritized:
        result = _build_cloze(prio.concept.context_sentence, prio.concept.name)
        if result is None:
            continue
        cloze_text, answer = result
        out.append(
            ClozeCard(
                text=cloze_text,
                answer=answer,
                concept=prio.concept.name,
                source_chunk_id=prio.concept.source_chunk_id,
            )
        )
        if limit is not None and len(out) >= limit:
            break
    return out
