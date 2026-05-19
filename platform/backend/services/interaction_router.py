from __future__ import annotations

import logging
import uuid
from functools import lru_cache
from typing import Any, Literal

from pydantic import BaseModel, Field

from teacherlm_core.llm.ollama_client import OllamaClient
from teacherlm_core.llm.runtime import build_llm_client_kwargs

from config import Settings, get_settings
from db.session import session_scope
from services.course_content_store import get_course_content_store


logger = logging.getLogger(__name__)

RouteAction = Literal["conversational_reply", "retrieve", "outside_files"]


class InteractionDecision(BaseModel):
    action: RouteAction
    response: str = ""
    retrieval_query: str = ""
    reasoning: str = Field(default="", exclude=True)


class InteractionRouter:
    """LLM router that decides whether a chat turn needs course retrieval."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def route(
        self,
        *,
        conversation_id: uuid.UUID,
        user_message: str,
        chat_history: list[dict[str, str]],
        learner_state: dict[str, Any] | None,
        options: dict[str, Any],
    ) -> InteractionDecision:
        course_summary = await build_course_summary(conversation_id)
        client = self._client(options)
        try:
            return await client.chat_structured(
                messages=[
                    {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
                    {
                        "role": "user",
                        "content": _format_router_input(
                            user_message=user_message,
                            course_summary=course_summary,
                            chat_history=chat_history,
                            learner_state=learner_state,
                        ),
                    },
                ],
                schema=InteractionDecision,
                options={"temperature": 0.1, "num_predict": 500, "max_tokens": 500},
            )
        except Exception:
            logger.exception("interaction routing failed; falling back to retrieval")
            return InteractionDecision(
                action="retrieve",
                retrieval_query=user_message,
                reasoning="router_failed",
            )

    def _client(self, options: dict[str, Any]) -> OllamaClient:
        raw_llm = options.get("llm") if isinstance(options, dict) else None
        cfg = build_llm_client_kwargs(
            default_base_url=self._settings.ollama_host,
            default_model=self._settings.ollama_chat_model,
            options=raw_llm if isinstance(raw_llm, dict) else None,
        )
        return OllamaClient(
            str(cfg["base_url"]),
            str(cfg["model"]),
            provider=str(cfg["provider"]),
            api_key=cfg["api_key"],
        )


async def build_course_summary(conversation_id: uuid.UUID) -> str:
    store = get_course_content_store()
    async with session_scope() as session:
        documents = await store.get_documents(session, conversation_id)
        sections = await store.get_sections(session, conversation_id)

    if not documents and not sections:
        return "No uploaded course files are indexed for this conversation yet."

    doc_labels = [
        _compact_label(str(doc.title or doc.source_filename))
        for doc in documents[:6]
        if str(doc.title or doc.source_filename).strip()
    ]

    concepts: list[str] = []
    headings: list[str] = []
    for section in sections:
        headings.extend(str(item) for item in (section.heading_path or [section.title])[-1:])
        concepts.extend(str(item) for item in (section.key_concepts or []))

    concept_labels = _dedupe_labels([*concepts, *headings], limit=18)
    pieces: list[str] = []
    if doc_labels:
        pieces.append("Uploaded files: " + ", ".join(_dedupe_labels(doc_labels, limit=6)) + ".")
    if concept_labels:
        pieces.append("Main visible topics: " + ", ".join(concept_labels) + ".")
    if not pieces:
        pieces.append("Uploaded course files are present, but no reliable summary terms were extracted.")

    summary = " ".join(pieces)
    if len(summary) > 1200:
        summary = summary[:1200].rsplit(" ", 1)[0].strip() + "."
    return summary


def _format_router_input(
    *,
    user_message: str,
    course_summary: str,
    chat_history: list[dict[str, str]],
    learner_state: dict[str, Any] | None,
) -> str:
    history = "\n".join(
        f"{item.get('role', 'user')}: {item.get('content', '')}"
        for item in chat_history[-6:]
    ) or "(no prior turns)"
    learner = learner_state or {}
    struggling = ", ".join(str(item) for item in learner.get("struggling_concepts", [])[:5])
    understood = ", ".join(str(item) for item in learner.get("understood_concepts", [])[:5])
    return (
        f"Course summary:\n{course_summary}\n\n"
        f"Recent chat history:\n{history}\n\n"
        f"Learner state: understood=[{understood or 'none'}], "
        f"struggling=[{struggling or 'none'}]\n\n"
        f"Student message:\n{user_message}"
    )


def _compact_label(value: str) -> str:
    return " ".join(value.replace("_", " ").split()).strip(" -")


def _dedupe_labels(values: list[str], *, limit: int) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        label = _compact_label(value)
        if not 2 <= len(label) <= 90:
            continue
        key = label.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(label)
        if len(out) >= limit:
            break
    return out


_ROUTER_SYSTEM_PROMPT = """You are TeacherLM's interaction router.

You receive:
- a compact summary of the uploaded course files,
- recent chat history,
- learner state,
- the latest student message.

Decide exactly one action:

1. "conversational_reply"
Use this for normal human interaction that does not require factual course
content: greetings, thanks, encouragement, meta questions about how to study,
emotional support, or brief classroom back-and-forth. Write a warm teacher
reply in response. Do not invent course facts.

2. "retrieve"
Use this when the student asks for explanation, teaching, examples, formulas,
comparison, summary, quiz-like help, study order, or any answer that should be
grounded in the uploaded files. The message must be plausibly about the course
summary, a previous course topic, or the course as a whole. Set retrieval_query
to the best compact search query.

3. "outside_files"
Use this when the student asks for factual/substantive information that is not
related to the uploaded course summary or recent course context. Reply briefly
that it appears outside the uploaded files and invite them to ask about the
course material.

Rules:
- If answering requires course knowledge, do not answer directly; choose
  "retrieve".
- If unsure whether a substantive question is course-related, choose
  "retrieve" so the retrieval system can verify.
- For "conversational_reply" and "outside_files", response must be non-empty.
- For "retrieve", response may be empty.
- Return only JSON matching the schema."""


@lru_cache(maxsize=1)
def get_interaction_router() -> InteractionRouter:
    return InteractionRouter()
