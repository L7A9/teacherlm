import json
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
    return await llm.analyze_structured(
        system=system,
        user_message=user_message,
        schema=QueryAnalysis,
    )
