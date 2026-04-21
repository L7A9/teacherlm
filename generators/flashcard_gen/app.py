from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from teacherlm_core.schemas.generator_io import GeneratorInput

from .config import get_settings
from .pipeline import run

settings = get_settings()

app = FastAPI(title="TeacherLM — flashcard_gen", version=settings.version)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "generator_id": settings.generator_id}


@app.get("/info")
async def info() -> dict:
    return {
        "generator_id": settings.generator_id,
        "output_type": settings.output_type,
        "version": settings.version,
        "retrieval_mode": "coverage_broad",
        "streams": True,
        "capabilities": {
            "card_types": ["basic", "cloze"],
            "exports": ["json", "csv", "apkg"],
            "scheduling": "sm2",
            "adapts_to_learner_state": True,
        },
        "models": {
            "chat": settings.chat_model,
            "generation": settings.generation_model,
        },
        "spacy_model": settings.spacy_model,
        "embedding_model": settings.embedding_model,
    }


@app.post("/run")
async def run_endpoint(payload: GeneratorInput) -> StreamingResponse:
    return StreamingResponse(
        run(payload),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "flashcard_gen.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
