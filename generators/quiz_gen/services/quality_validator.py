from __future__ import annotations

import re

from teacherlm_core.schemas.chunk import Chunk

from ..schemas import FillBlank, MCQ, Question, TrueFalse


_WS_RE = re.compile(r"\s+")
_GENERIC_SOURCE_RE = re.compile(
    r"\b("
    r"which concept|which source|source section|course material|retrieved section|provided chunk|"
    r"uploaded course|uploaded material|the source includes|the source .*material|"
    r"according to (?:the )?(?:chunk|source|course material)"
    r")\b",
    re.IGNORECASE,
)
_GIVEAWAY_OPTION_RE = re.compile(
    r"\b("
    r"all of the above|none of the above|not enough information|cannot be determined|"
    r"course material|uploaded material|provided chunk|retrieved section"
    r")\b",
    re.IGNORECASE,
)


def _is_valid_mcq(q: MCQ, chunks_by_id: dict[str, Chunk] | None = None) -> bool:
    if len(q.options) < 4:
        return False
    if not (0 <= q.correct_index < len(q.options)):
        return False
    if any(not opt.strip() for opt in q.options):
        return False
    # Distinct options (case-insensitive, whitespace-trimmed).
    seen = {opt.strip().lower() for opt in q.options}
    if len(seen) != len(q.options):
        return False
    if not q.question.strip() or not q.explanation.strip():
        return False
    if _looks_generic_or_source_aware(q.question):
        return False
    if any(_GIVEAWAY_OPTION_RE.search(option) for option in q.options):
        return False
    correct = q.options[q.correct_index]
    if _answer_is_shown_in_question(q.question, correct):
        return False
    if chunks_by_id and _is_ambiguous_list_mcq(q, chunks_by_id):
        return False
    return True


def _is_ambiguous_list_mcq(q: MCQ, chunks_by_id: dict[str, Chunk]) -> bool:
    if not re.search(r"\b(which|which of the following|listed|shown|mentioned)\b", q.question, re.IGNORECASE):
        return False
    chunk = chunks_by_id.get(q.source_chunk_id)
    if chunk is None:
        return False
    source_text = chunk.text.casefold()
    mentioned_options = [
        opt
        for opt in q.options
        if len(opt.strip()) >= 4 and opt.strip().casefold() in source_text
    ]
    return len({opt.strip().casefold() for opt in mentioned_options}) > 1


def _is_valid_true_false(q: TrueFalse) -> bool:
    if not q.question.strip() or not q.explanation.strip():
        return False
    if _looks_generic_or_source_aware(q.question):
        return False
    return isinstance(q.answer, bool)


def _looks_generic_or_source_aware(question: str) -> bool:
    return bool(_GENERIC_SOURCE_RE.search(question))


def _answer_is_shown_in_question(question: str, answer: str) -> bool:
    question_key = _surface_key(question)
    answer_key = _surface_key(answer)
    if len(answer_key) < 4:
        return False
    if answer_key in question_key:
        return True
    answer_tokens = answer_key.split()
    if len(answer_tokens) < 2:
        return False
    question_tokens = set(question_key.split())
    matched = sum(1 for token in answer_tokens if token in question_tokens and len(token) > 3)
    return matched >= max(2, len(answer_tokens) - 1)


def _surface_key(value: str) -> str:
    return _WS_RE.sub(" ", re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold())).strip()


def _is_valid_fill_blank(q: FillBlank) -> bool:
    if "____" not in q.question:
        return False
    # Exactly one blank.
    if q.question.count("____") != 1:
        return False
    if not q.answer.strip() or not q.explanation.strip():
        return False
    # Don't accept a blank that the answer itself fills with whitespace.
    if q.answer.strip().lower() == "____":
        return False
    return True


def is_valid(question: Question, chunks_by_id: dict[str, Chunk] | None = None) -> bool:
    if isinstance(question, MCQ):
        return _is_valid_mcq(question, chunks_by_id)
    if isinstance(question, TrueFalse):
        return _is_valid_true_false(question)
    if isinstance(question, FillBlank):
        return _is_valid_fill_blank(question)
    return False


def validate_questions(
    questions: list[Question],
    chunks: list[Chunk] | None = None,
) -> tuple[list[Question], list[dict]]:
    """Drop invalid questions and return them alongside per-question rejection notes."""
    chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks or []}
    kept: list[Question] = []
    dropped: list[dict] = []
    for q in questions:
        if is_valid(q, chunks_by_id):
            kept.append(q)
        else:
            dropped.append(
                {
                    "type": q.type,
                    "concept": getattr(q, "concept", ""),
                    "reason": "failed_quality_validation",
                }
            )
    return kept, dropped


def _question_signature(q: Question) -> str:
    """Normalized form used to detect near-duplicate questions.

    Lowercases, collapses whitespace, strips trailing punctuation. Good enough
    to catch the common failure mode (same concept + same chunk → same wording).
    """
    text = q.question.strip().lower().rstrip("?.!")
    return _WS_RE.sub(" ", text)


def deduplicate_questions(
    questions: list[Question],
) -> tuple[list[Question], list[dict]]:
    """Drop questions that share a normalized wording with an earlier one."""
    kept: list[Question] = []
    dropped: list[dict] = []
    seen: set[str] = set()
    for q in questions:
        sig = _question_signature(q)
        if sig in seen:
            dropped.append(
                {
                    "type": q.type,
                    "concept": getattr(q, "concept", ""),
                    "reason": "duplicate_question",
                }
            )
            continue
        seen.add(sig)
        kept.append(q)
    return kept, dropped


def bloom_distribution(questions: list[Question]) -> dict[str, int]:
    counts: dict[str, int] = {"remember": 0, "understand": 0, "apply": 0, "analyze": 0}
    for q in questions:
        counts[q.bloom_level] = counts.get(q.bloom_level, 0) + 1
    return counts
