import json
from collections.abc import AsyncIterator
from functools import lru_cache

from teacherlm_core.llm.language import set_current_language
from teacherlm_core.llm.runtime import set_current_llm_options
from teacherlm_core.retrieval.reranker import CrossEncoderReranker
from teacherlm_core.schemas.chunk import Chunk
from teacherlm_core.schemas.generator_io import GeneratorInput, LearnerUpdates

from .config import get_settings
from .schemas import ResponseMode
from .services.confidence_scorer import compute as compute_confidence
from .services.hyde_generator import rerank_with_hyde
from .services.learner_analyzer import extract_learner_updates
from .services.llm_service import build_chat_system_prompt, get_llm_service
from .services.query_analyzer import analyze as analyze_query
from .services.response_mode import select_mode

_MODE_PROMPT_FILE: dict[ResponseMode, str] = {
    "explain": "mode_explain.txt",
    "guide": "mode_guide.txt",
    "quiz_back": "mode_quiz_back.txt",
    "affirm": "mode_affirm.txt",
}


@lru_cache
def get_reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker()


def _format_chunks(chunks: list[Chunk]) -> str:
    if not chunks:
        return "(no context chunks available)"
    return "\n\n".join(
        f"[{i + 1}] source={c.source} score={c.score:.3f}\n{c.text}"
        for i, c in enumerate(chunks)
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


async def run(inp: GeneratorInput) -> AsyncIterator[str]:
    settings = get_settings()
    options = dict(inp.options or {})
    set_current_llm_options(options)
    llm = get_llm_service()

    set_current_language(options.get("language"))

    history = inp.chat_history or []
    learner = inp.learner_state

    analysis = await analyze_query(
        user_message=inp.user_message,
        chat_history=history,
        learner_state=learner.model_dump(),
        llm=llm,
    )
    mode = select_mode(analysis, learner)

    yield _sse(
        "analysis",
        {
            "intent": analysis.intent,
            "confusion_level": analysis.confusion_level,
            "targets_concept": analysis.targets_concept,
            "mode": mode,
        },
    )

    if inp.context_chunks:
        ranked_chunks = await rerank_with_hyde(
            user_message=inp.user_message,
            chunks=inp.context_chunks,
            top_k=settings.rerank_top_k,
            reranker=get_reranker(),
            llm=llm,
            enabled=settings.hyde_enabled,
        )
    else:
        ranked_chunks = []

    yield _sse(
        "sources",
        {
            "sources": [
                {
                    "text": c.text,
                    "source": c.source,
                    "score": c.score,
                    "chunk_id": c.chunk_id,
                }
                for c in ranked_chunks
            ]
        },
    )

    top_score = ranked_chunks[0].score if ranked_chunks else float("-inf")
    if top_score < settings.min_relevance_score:
        refusal = (
            "That question doesn't appear to be covered in the course materials you've "
            "uploaded, so I can't answer it from your sources. I stay grounded in your "
            "uploaded files — want to explore a topic from them instead?"
        )
        yield _sse("token", {"delta": refusal})
        yield _sse(
            "done",
            {
                "response": refusal,
                "generator_id": settings.generator_id,
                "output_type": settings.output_type,
                "artifacts": [],
                "sources": [],
                "learner_updates": LearnerUpdates().model_dump(),
                "metadata": {
                    "mode": "refuse",
                    "analysis": analysis.model_dump(),
                    "confidence": 1.0,
                    "hyde_enabled": settings.hyde_enabled,
                    "refused_reason": "off_topic",
                    "top_score": top_score,
                },
            },
        )
        return

    system = build_chat_system_prompt(
        _MODE_PROMPT_FILE[mode],
        context=_format_chunks(ranked_chunks),
        understood_concepts=", ".join(learner.understood_concepts) or "(none yet)",
        struggling_concepts=", ".join(learner.struggling_concepts) or "(none)",
        user_message=inp.user_message,
    )

    response_parts: list[str] = []
    async for delta in llm.stream_reply(
        system=system,
        chat_history=history,
        user_message=inp.user_message,
    ):
        response_parts.append(delta)
        yield _sse("token", {"delta": delta})

    full_response = "".join(response_parts)

    confidence = await compute_confidence(
        response=full_response,
        chunks=ranked_chunks,
        query=inp.user_message,
    )

    extraction = await extract_learner_updates(
        user_message=inp.user_message,
        assistant_response=full_response,
        llm=llm,
    )
    learner_updates = LearnerUpdates(
        concepts_covered=extraction.covered,
        concepts_demonstrated=extraction.demonstrated_understanding,
        concepts_struggled=extraction.showed_confusion,
    )

    yield _sse(
        "done",
        {
            "response": full_response,
            "generator_id": settings.generator_id,
            "output_type": settings.output_type,
            "artifacts": [],
            "sources": [c.model_dump() for c in ranked_chunks],
            "learner_updates": learner_updates.model_dump(),
            "metadata": {
                "mode": mode,
                "analysis": analysis.model_dump(),
                "confidence": confidence,
                "hyde_enabled": settings.hyde_enabled,
            },
        },
    )
