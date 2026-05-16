from __future__ import annotations

import re

from teacherlm_core.schemas.chunk import Chunk

from ..schemas import FillBlank, MCQ, Question, TrueFalse


_WS_RE = re.compile(r"\s+")


def _is_valid_mcq(q: MCQ, chunks_by_id: dict[str, Chunk] | None = None) -> bool:
    if len(q.options) < 2:
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
    return isinstance(q.answer, bool)


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
