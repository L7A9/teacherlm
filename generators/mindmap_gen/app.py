from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from teacherlm_core.llm.streaming import safe_sse_stream
from teacherlm_core.schemas.generator_io import GeneratorInput

from .config import settings
from .pipeline import run as pipeline_run

app = FastAPI(title="TeacherLM — mindmap_gen")

# Frontend on a different origin (e.g. http://localhost:3000) fetches the
# generated chart JSON directly from /artifacts/ — needs CORS.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["*"],
)

_artifacts_dir = Path(settings.ARTIFACTS_DIR)
_artifacts_dir.mkdir(parents=True, exist_ok=True)
app.mount(
    "/artifacts",
    StaticFiles(directory=str(_artifacts_dir)),
    name="artifacts",
)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "generator_id": "mindmap_gen"}


@app.get("/info")
async def info() -> dict:
    return {
        "generator_id": "mindmap_gen",
        "output_type": "mindmap",
        "description": "Generates hierarchical mind maps from course materials",
        "supported_sizes": ["concise", "standard", "comprehensive"],
        "language_support": "auto (matches source content)",
        "retrieval_mode": "topic_clusters",
        "streams": True,
        "models": {
            "generation": settings.MODEL_NAME,
        },
    }


@app.post("/run")
async def run_endpoint(payload: GeneratorInput) -> StreamingResponse:
    return StreamingResponse(
        safe_sse_stream(pipeline_run(payload)),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mindmap_gen.app:app", host="0.0.0.0", port=8008, reload=False)
