from __future__ import annotations

import re

_FORMULA_RE = re.compile(r"([=+\-*/^]|\\frac|\\sum|\\int|[A-Za-z]\([^)]+\))")


def should_skip_hyde(query: str) -> bool:
    """Formula-heavy queries usually need exact symbols rather than invented prose."""

    normalized = query.strip()
    if not normalized:
        return True
    symbol_hits = len(_FORMULA_RE.findall(normalized))
    return symbol_hits >= 2 or (symbol_hits == 1 and len(normalized) < 80)


def build_hyde_prompt(query: str, max_chars: int = 900) -> str:
    return (
        "Write a short hypothetical course excerpt that would help retrieve "
        "the right uploaded lecture notes for this student question. Do not "
        "answer as final truth; this text is only for retrieval.\n\n"
        f"Student question:\n{query.strip()}\n\n"
        f"Keep it under {max_chars} characters."
    )

