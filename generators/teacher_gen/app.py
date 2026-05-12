from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from teacherlm_core.llm.streaming import safe_sse_stream
from teacherlm_core.schemas.generator_io import GeneratorInput

from .config import get_settings
from .pipeline import run

settings = get_settings()

app = FastAPI(title="TeacherLM — teacher_gen", version=settings.version)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "generator_id": settings.generator_id}


@app.get("/info")
async def info() -> dict:
    return {
        "generator_id": settings.generator_id,
        "output_type": settings.output_type,
        "version": settings.version,
        "retrieval_mode": "semantic_topk",
        "streams": True,
        "capabilities": {
            "modes": ["explain", "guide", "quiz_back", "affirm"],
            "hyde_enabled": settings.hyde_enabled,
            "reports_learner_updates": True,
            "returns_confidence": True,
        },
        "models": {
            "chat": settings.chat_model,
            "analysis": settings.analysis_model,
            "extraction": settings.extraction_model,
        },
    }


@app.post("/run")
async def run_endpoint(payload: GeneratorInput) -> StreamingResponse:
    return StreamingResponse(
        safe_sse_stream(run(payload)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "teacher_gen.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
