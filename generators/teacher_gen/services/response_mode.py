from teacherlm_core.schemas.learner_state import LearnerState

from ..config import get_settings
from ..schemas import QueryAnalysis, ResponseMode


def _student_seems_correct(analysis: QueryAnalysis) -> bool:
    return analysis.confusion_level < 0.3


def select_mode(
    analysis: QueryAnalysis,
    learner_state: LearnerState,
) -> ResponseMode:
    s = get_settings()

    if (
        learner_state.turns_since_progress > s.stuck_turns_threshold
        and analysis.confusion_level > s.confusion_guide_threshold
    ):
        return "explain"

    if analysis.confusion_level > s.confusion_guide_threshold:
        return "guide"

    if analysis.intent == "confusion":
        return "guide"

    if analysis.intent == "confirmation" and _student_seems_correct(analysis):
        return "affirm"

    if analysis.intent == "confirmation":
        return "quiz_back"

    if analysis.intent == "new_question":
        return "explain"

    if analysis.requires_direct_answer:
        return "explain"

    return "explain"
