# quiz_gen

Adaptive quiz generator for TeacherLM. Port **8002**, Python **3.14+**.

Generates MCQ / true-false / fill-in-the-blank questions grounded in a
student's uploaded course materials. Adapts difficulty to `learner_state` and
returns the quiz as both inline `metadata.quiz_data` and an artifact JSON in
MinIO.

## Pipeline

1. Extract concepts from the retrieved chunks, grouped by Bloom's level
   (`remember | understand | apply | analyze`) — via `ollama format=` with
   the `ExtractedConcepts` schema.
2. Plan the question mix from `learner_state` (60% struggling / 30% coverage
   / 10% stretch) — see `services/difficulty_adapter.py`.
3. Generate one question at a time, each via
   `ollama format=Schema.model_json_schema()` for reliability.
4. Enhance MCQ distractors with `fastembed` cosine similarity in the
   `[0.4, 0.7]` band (hard negatives).
5. Validate quality (option distinctness, single blank, etc.).
6. Build a teacher-voice intro message that names struggling concepts.
7. Save the quiz JSON to MinIO at
   `conversations/{conversation_id}/artifacts/{uuid}_quiz.json` and return a
   presigned URL.
8. Return a `GeneratorOutput` with the intro as `response`, the artifact, and
   `learner_updates.concepts_covered` populated.

## Endpoints

- `GET  /health` — liveness probe.
- `GET  /info`   — capabilities, models, retrieval mode (`coverage_broad`).
- `POST /run`    — SSE stream. Body: `GeneratorInput`.

## Configuration

All settings prefixed `QUIZ_GEN_` (env or `.env`). Notable knobs:

| Setting | Default | Purpose |
|---|---|---|
| `QUIZ_GEN_PORT` | `8002` | HTTP port |
| `QUIZ_GEN_OLLAMA_HOST` | `http://localhost:11434` | Ollama base URL |
| `QUIZ_GEN_GENERATION_MODEL` | `llama3.1:8b-instruct-q4_K_M` | Used for question generation |
| `QUIZ_GEN_DEFAULT_QUESTION_COUNT` | `8` | Default `n_questions` |
| `QUIZ_GEN_MIX_STRUGGLING` / `MIX_COVERAGE` / `MIX_STRETCH` | `0.6 / 0.3 / 0.1` | Slot mix |
| `QUIZ_GEN_DISTRACTOR_SIM_MIN` / `_MAX` | `0.4 / 0.7` | Hard-negative band |
| `QUIZ_GEN_MINIO_ENDPOINT` | `localhost:9000` | Artifact storage |
| `QUIZ_GEN_MINIO_BUCKET` | `teacherlm` | Shared with platform |

`options.n_questions` (or `count`) in `GeneratorInput.options` overrides the
default per request, clamped to `[min_question_count, max_question_count]`.

## Local run

```bash
pip install -r requirements.txt
QUIZ_GEN_OLLAMA_HOST=http://localhost:11434 \
QUIZ_GEN_MINIO_ENDPOINT=localhost:9000 \
uvicorn quiz_gen.app:app --host 0.0.0.0 --port 8002
```

(Run from `generators/` so `quiz_gen` resolves as a package, or use the
Dockerfile which sets `PYTHONPATH=/app/generators`.)

## Docker

```bash
docker build -f generators/quiz_gen/Dockerfile -t teacherlm/quiz_gen:latest .
```

To enable in the platform: flip `enabled: true` for `quiz_gen` in
`generators_registry.json` and add the service to `platform/docker-compose.yml`
(mirror the `teacher_gen` block, change ports/env to `QUIZ_GEN_*`, and set
`QUIZ_GEN_MINIO_ENDPOINT=minio:9000`).
