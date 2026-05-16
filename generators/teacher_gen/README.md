# teacher_gen

Chat Q&A generator with teacher personality and adaptive guidance. This is the main chat experience of TeacherLM — it answers questions grounded in uploaded course files, switches to Socratic mode when a student is stuck, affirms when a student gets it, and tracks learner progress across the session.

- **Port:** 8001
- **Python:** 3.14+
- **Output type:** `text`
- **Retrieval mode requested:** `semantic_topk`

## What it does

For each turn, the pipeline:

1. **Analyzes the query** — classifies intent (`new_question` / `clarification` / `confusion` / `confirmation` / `follow_up`) and scores `confusion_level`.
2. **Picks a response mode**:
   - `guide` — Socratic, one question, no answer (student is confused)
   - `quiz_back` — affirm partial + probe the gap (student is half-right)
   - `affirm` — celebrate + suggest next concept (student got it)
   - `explain` — direct answer with analogies and citations (default)
   - Stuck-safety: if the student has been struggling for more than `stuck_turns_threshold` turns, the pipeline falls back to `explain` instead of another Socratic question.
3. **Uses backend-prepared context chunks** from the shared RAG pipeline.
4. **Streams the response** in the shared teacher voice (prompts prepend `teacherlm_core/prompts/teacher_voice.txt`).
5. **Scores confidence** (groundedness + coverage) and **extracts learner updates** (concepts covered / demonstrated / struggled).

## Endpoints

| Method | Path      | Description |
|--------|-----------|-------------|
| POST   | `/run`    | SSE stream. Body: `GeneratorInput`. Events: `analysis`, `sources`, `token` (many), `done`. |
| GET    | `/health` | Liveness. |
| GET    | `/info`   | Capabilities, models, version. |

The `done` event payload matches the platform-wide `GeneratorOutput` contract (`response`, `generator_id`, `output_type`, `artifacts`, `sources`, `learner_updates`, `metadata`).

## Running locally

```bash
cd teacherlm/generators/teacher_gen
pip install -r requirements.txt
python -m teacher_gen.app
# or
uvicorn teacher_gen.app:app --host 0.0.0.0 --port 8001
```

`ollama` must be reachable at `TEACHER_GEN_OLLAMA_HOST` (default `http://localhost:11434`) with the configured models pulled.

## Running in Docker

Build from the **repo root** so the build context includes `packages/teacherlm_core`:

```bash
docker build -f generators/teacher_gen/Dockerfile -t teacherlm/teacher_gen:latest .
docker run --rm -p 8001:8001 \
  -e TEACHER_GEN_OLLAMA_HOST=http://host.docker.internal:11434 \
  teacherlm/teacher_gen:latest
```

## Configuration

All settings are env-prefixed with `TEACHER_GEN_`. Highlights:

| Env var                               | Default                          |
|---------------------------------------|----------------------------------|
| `TEACHER_GEN_PORT`                    | `8001`                           |
| `TEACHER_GEN_OLLAMA_HOST`             | `http://localhost:11434`         |
| `TEACHER_GEN_CHAT_MODEL`              | `llama3.1:8b-instruct`           |
| `TEACHER_GEN_ANALYSIS_MODEL`          | `llama3.1:8b-instruct`           |
| `TEACHER_GEN_EXTRACTION_MODEL`        | `llama3.1:8b-instruct`           |
| `TEACHER_GEN_CONFUSION_GUIDE_THRESHOLD` | `0.7`                          |
| `TEACHER_GEN_STUCK_TURNS_THRESHOLD`   | `4`                              |

## Enabling in `generators_registry.json`

Add an entry at the repo root `generators_registry.json`:

```json
{
  "generators": [
    {
      "id": "teacher_gen",
      "name": "Teacher",
      "description": "Chat Q&A with teacher personality and adaptive guidance.",
      "output_type": "text",
      "endpoint": "http://teacher_gen:8001/run",
      "info_endpoint": "http://teacher_gen:8001/info",
      "health_endpoint": "http://teacher_gen:8001/health",
      "retrieval_mode": "semantic_topk",
      "streams": true,
      "default": true,
      "enabled": true
    }
  ]
}
```

The `streams: true` flag tells the platform to proxy SSE events to the frontend. The `default: true` flag marks this as the generator the platform routes to when the user hasn't explicitly chosen a different output type.

## Voice and prompts

All mode prompts are prepended with the shared `teacherlm_core/prompts/teacher_voice.txt`. To tune the personality globally (across every generator), edit that file — do **not** duplicate voice instructions inside each mode prompt. Mode prompts should describe only what is specific to that mode (explain / guide / quiz_back / affirm).
