# teacher_gen

`teacher_gen` is the default chat generator for TeacherLM. It answers student questions in a warm teacher voice while staying grounded in retrieved course chunks.

## Service

Default port: `8001`

Endpoints:

- `GET /health`
- `GET /info`
- `POST /run`

`/run` streams server-sent events.

Common events:

- `analysis`: query analysis and selected response mode.
- `sources`: source chunks used by the response.
- `token`: streamed markdown response text.
- `done`: final generator output.
- `error`: failure details.

## Generator Info

| Field | Value |
| --- | --- |
| Generator id | `teacher_gen` |
| Output type | `text` |
| Retrieval mode | `semantic_topk` |
| Max context chunks consumed | 16 |

The backend retrieves and filters context first, including selected source-file filters from the frontend. `teacher_gen` then uses at most the first 16 provided context chunks.

## Behavior

The generator can:

- Explain course concepts.
- Guide the student through reasoning.
- Ask quick check-back questions.
- Affirm correct understanding.
- Refuse off-topic answers when the retrieved evidence is too weak.
- Produce formula-focused answers when the context contains formula cards.
- Produce course-overview answers when the user asks for broad course structure.
- Return learner updates for covered, demonstrated, and struggled concepts.
- Include confidence metadata based on groundedness and coverage.

Responses must use uploaded course evidence only. If the course context does not support an answer, the generator should say so and guide the student back to available material.

## Input Options

The platform may pass:

- Runtime LLM provider overrides.
- Forced language preferences.
- Source-file-filtered context chunks.
- Chat history.
- Learner state.

The frontend sends selected file ids to the backend. The backend applies the filter before calling this generator.

## Environment

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `TEACHER_GEN_OLLAMA_URL` | Ollama base URL override |
| `TEACHER_GEN_CHAT_MODEL` | model for final chat responses |
| `TEACHER_GEN_ANALYSIS_MODEL` | model for query analysis |
| `TEACHER_GEN_EXTRACTION_MODEL` | model for learner-update extraction |
| `TEACHER_GEN_MAX_CONTEXT_CHUNKS` | maximum chunks consumed by the generator |
| `OLLAMA_URL` | shared Ollama fallback URL |
| `OLLAMA_CHAT_MODEL` | shared chat model fallback |
| `OLLAMA_ANALYSIS_MODEL` | shared analysis model fallback |

## Local Run

From this directory:

```bash
pip install -e ../../packages/teacherlm_core
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001
```

Through Docker Compose:

```bash
cd ../../platform
docker compose up -d teacher_gen
```

## Tests

From the repository root:

```bash
pytest generators/teacher_gen/tests
```
