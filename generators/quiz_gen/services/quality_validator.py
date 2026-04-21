from __future__ import annotations

from ..schemas import FillBlank, MCQ, Question, TrueFalse


def _is_valid_mcq(q: MCQ) -> bool:
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
    return True


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


def is_valid(question: Question) -> bool:
    if isinstance(question, MCQ):
        return _is_valid_mcq(question)
    if isinstance(question, TrueFalse):
        return _is_valid_true_false(question)
    if isinstance(question, FillBlank):
        return _is_valid_fill_blank(question)
    return False


def validate_questions(questions: list[Question]) -> tuple[list[Question], list[dict]]:
    """Drop invalid questions and return them alongside per-question rejection notes."""
    kept: list[Question] = []
    dropped: list[dict] = []
    for q in questions:
        if is_valid(q):
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


def bloom_distribution(questions: list[Question]) -> dict[str, int]:
    counts: dict[str, int] = {"remember": 0, "understand": 0, "apply": 0, "analyze": 0}
    for q in questions:
        counts[q.bloom_level] = counts.get(q.bloom_level, 0) + 1
    return counts
