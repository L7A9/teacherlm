from ..schemas import ConceptExtraction
from .llm_service import LLMService, build_system_prompt


async def extract_learner_updates(
    user_message: str,
    assistant_response: str,
    llm: LLMService,
) -> ConceptExtraction:
    system = build_system_prompt(
        "learner_update_extraction.txt",
        user_message=user_message,
        assistant_response=assistant_response,
    )
    analyst_input = (
        f"Student message:\n{user_message}\n\n"
        f"Teacher response:\n{assistant_response}"
    )
    try:
        return await llm.extract_structured(
            system=system,
            user_message=analyst_input,
            schema=ConceptExtraction,
        )
    except Exception:
        return ConceptExtraction()
