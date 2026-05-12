import json
import re
from typing import Any

from ..schemas import QueryAnalysis
from .llm_service import LLMService, build_system_prompt


def _format_history(chat_history: list[dict], max_turns: int = 6) -> str:
    if not chat_history:
        return "(no prior turns)"
    tail = chat_history[-max_turns:]
    return "\n".join(f"{m['role']}: {m['content']}" for m in tail)


def _format_learner_state(learner_state: dict[str, Any] | None) -> str:
    if not learner_state:
        return "(empty)"
    return json.dumps(
        {
            "understood_concepts": learner_state.get("understood_concepts", []),
            "struggling_concepts": learner_state.get("struggling_concepts", []),
            "session_turns": learner_state.get("session_turns", 0),
        },
        indent=2,
    )


async def analyze(
    user_message: str,
    chat_history: list[dict],
    learner_state: dict[str, Any] | None,
    llm: LLMService,
) -> QueryAnalysis:
    system = build_system_prompt(
        "query_analysis.txt",
        user_message=user_message,
        chat_history=_format_history(chat_history),
        learner_state=_format_learner_state(learner_state),
    )
    try:
        return await llm.analyze_structured(
            system=system,
            user_message=user_message,
            schema=QueryAnalysis,
        )
    except Exception:
        return _fallback_analysis(user_message, chat_history)


def _fallback_analysis(user_message: str, chat_history: list[dict]) -> QueryAnalysis:
    text = user_message.strip()
    lowered = text.lower()

    confusion_markers = (
        "i don't understand",
        "i dont understand",
        "i don't get",
        "i dont get",
        "i'm lost",
        "im lost",
        "confused",
        "huh",
        "what do you mean",
        "not clear",
        "doesn't make sense",
        "doesnt make sense",
    )
    confirmation_markers = (
        "right?",
        "is that correct",
        "am i right",
        "so ",
        "does that mean",
        "can i say",
    )
    clarification_markers = (
        "explain again",
        "rephrase",
        "another way",
        "more detail",
        "clarify",
        "example",
    )

    if any(marker in lowered for marker in confusion_markers):
        intent = "confusion"
        confusion_level = 0.85
    elif any(marker in lowered for marker in clarification_markers) and chat_history:
        intent = "clarification"
        confusion_level = 0.45
    elif any(marker in lowered for marker in confirmation_markers):
        intent = "confirmation"
        confusion_level = 0.25
    elif chat_history and len(text.split()) < 12:
        intent = "follow_up"
        confusion_level = 0.2
    else:
        intent = "new_question"
        confusion_level = 0.15

    requires_direct_answer = bool(re.search(r"\?|what|why|how|define|explain", lowered))
    return QueryAnalysis(
        intent=intent,
        confusion_level=confusion_level,
        targets_concept=_guess_target_concept(text),
        requires_direct_answer=requires_direct_answer,
    )


def _guess_target_concept(text: str) -> str | None:
    words = re.findall(r"[\wÀ-ÿ'-]+", text)
    stop = {
        "what",
        "why",
        "how",
        "is",
        "are",
        "the",
        "a",
        "an",
        "can",
        "you",
        "explain",
        "me",
        "please",
        "does",
        "mean",
    }
    kept = [word for word in words if word.lower() not in stop]
    if not kept:
        return None
    return " ".join(kept[:6])
