from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from teacherlm_core.llm.streaming import safe_sse_stream
from teacherlm_core.schemas.generator_io import GeneratorInput

from .config import get_settings
from .pipeline import run

settings = get_settings()

app = FastAPI(title="TeacherLM — podcast_gen", version=settings.version)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "generator_id": settings.generator_id}


@app.get("/info")
async def info() -> dict:
    languages = sorted(
        set(settings.piper_language_voices) | set(settings.language_voices)
    )
    voices_per_language = {}
    for lang in languages:
        piper = settings.piper_language_voices.get(lang)
        kokoro = settings.language_voices.get(lang)
        voices_per_language[lang] = {
            "piper": piper,
            "kokoro": (
                {"host_a": kokoro["host_a"], "host_b": kokoro["host_b"]}
                if kokoro
                else None
            ),
            "preferred_backend": "piper" if piper else "kokoro",
        }
    return {
        "generator_id": settings.generator_id,
        "output_type": settings.output_type,
        "version": settings.version,
        "retrieval_mode": "narrative_arc",
        "streams": True,
        "capabilities": {
            "durations": list(settings.duration_word_targets.keys()),
            "default_duration": settings.default_duration,
            "exports": ["mp3", "txt"],
            "languages": languages,
            "default_language": settings.default_language,
            "voices_per_language": voices_per_language,
            "tts_backends": ["piper", "kokoro", "pyttsx3"],
            "adapts_to_learner_state": False,
        },
        "models": {
            "chat": settings.chat_model,
            "extraction": settings.extraction_model,
            "generation": settings.generation_model,
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
        "podcast_gen.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
