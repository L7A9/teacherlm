from ..schemas import ConceptExtraction
from .llm_service import LLMService, build_system_prompt


async def extract_learner_updates(
    user_message: str,
    assistant_response: str,
    llm: LLMService,
    known_concepts: list[dict] | None = None,
) -> ConceptExtraction:
    concept_lines = []
    for item in known_concepts or []:
        name = str(item.get("name") or "").strip()
        aliases = item.get("aliases") or []
        alias_text = ", ".join(str(alias) for alias in aliases if str(alias).strip())
        if name:
            concept_lines.append(f"- {name}" + (f" (aliases: {alias_text})" if alias_text else ""))
    system = build_system_prompt(
        "learner_update_extraction.txt",
        user_message=user_message,
        assistant_response=assistant_response,
        known_concepts="\n".join(concept_lines) or "(no canonical concept inventory available)",
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
