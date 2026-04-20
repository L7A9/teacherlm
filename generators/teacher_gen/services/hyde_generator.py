from teacherlm_core.retrieval.reranker import CrossEncoderReranker
from teacherlm_core.schemas.chunk import Chunk

from .llm_service import LLMService

HYDE_SYSTEM = (
    "You are drafting a short, plausible textbook-style answer to a student's "
    "question. You do NOT have the source material — write what the answer "
    "would likely look like in a textbook on this topic. Keep it under 120 "
    "words. Dense with relevant terminology. Do not hedge, do not say you "
    "don't know, do not add disclaimers. Just write the hypothetical answer."
)


async def generate_hypothetical(user_message: str, llm: LLMService) -> str:
    response = await llm.analysis.chat(
        messages=[
            {"role": "system", "content": HYDE_SYSTEM},
            {"role": "user", "content": user_message},
        ],
        options={"temperature": 0.3},
    )
    return response["message"]["content"].strip()


async def rerank_with_hyde(
    user_message: str,
    chunks: list[Chunk],
    top_k: int,
    reranker: CrossEncoderReranker,
    llm: LLMService,
    enabled: bool = True,
) -> list[Chunk]:
    if not chunks:
        return []
    if not enabled:
        return await reranker.rerank(query=user_message, chunks=chunks, top_k=top_k)

    hyde_doc = await generate_hypothetical(user_message, llm)
    query = f"{user_message}\n\n{hyde_doc}"
    return await reranker.rerank(query=query, chunks=chunks, top_k=top_k)
