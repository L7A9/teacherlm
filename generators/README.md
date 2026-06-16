# TeacherLM Generators

This folder contains standalone output generator services. Each implemented generator is an independent FastAPI app with the same platform contract:

- `GET /health`
- `GET /info`
- `POST /run`

The platform owns retrieval and dispatch. Generators receive `GeneratorInput`, stream SSE events, return `GeneratorOutput`, and stay grounded in the provided `context_chunks`.

## Current Services

| Folder | Generator id | Output type | Port | Retrieval mode | Status |
| --- | --- | --- | ---: | --- | --- |
| `teacher_gen/` | `teacher_gen` | `text` | 8001 | `semantic_topk` | enabled |
| `quiz_gen/` | `quiz_gen` | `quiz` | 8002 | `coverage_broad` | enabled |
| `podcast_gen/` | `podcast_gen` | `podcast` | 8007 | `narrative_arc` | enabled |
| `mindmap_gen/` | `mindmap_gen` | `mindmap` | 8008 | `topic_clusters` | enabled |

Disabled registry entries currently have no enabled service folder:

- `report_gen`,
- `presentation_gen`,
- `chart_gen`.

The backend still reserves retrieval policies for those output types.

## Shared Contract

Input:

```json
{
  "conversation_id": "string",
  "user_message": "string",
  "context_chunks": [
    {
      "text": "course evidence",
      "source": "lecture.pdf",
      "score": 0.0,
      "chunk_id": "stable-id",
      "metadata": {}
    }
  ],
  "learner_state": {},
  "chat_history": [],
  "options": {}
}
```

Output:

```json
{
  "response": "markdown",
  "generator_id": "quiz_gen",
  "output_type": "quiz",
  "artifacts": [],
  "sources": [],
  "learner_updates": {
    "concepts_covered": [],
    "concepts_demonstrated": [],
    "concepts_struggled": []
  },
  "metadata": {}
}
```

The models are defined in `packages/teacherlm_core/teacherlm_core/schemas`.

## How The Platform Calls A Generator

1. `generators_registry.json` declares the generator endpoint and enabled state.
2. The backend resolves a generator by default chat generator or requested `output_type`.
3. The backend retrieves course context according to output type.
4. The backend applies source-file selection before the generator sees any chunks.
5. The backend loads learner state and chat history.
6. The backend resolves runtime provider/language options.
7. `ApiAdapter` posts the stable input to `POST /run`.
8. The generator streams events.
9. The backend persists the final output and merges learner updates.

Generators do not:

- parse uploaded files,
- query Qdrant,
- build BM25 indexes for platform retrieval,
- apply source-file filters,
- mutate Postgres learner state,
- decide whether disabled output types are available.

## Common Technologies

All implemented generators use:

- Python 3.14+,
- FastAPI,
- Pydantic V2,
- Ollama Python client,
- native Ollama `format=` for structured local output,
- `teacherlm_core`,
- SSE streaming,
- runtime LLM provider overrides,
- forced language support through shared `ContextVar` helpers.

They intentionally avoid:

- LangChain,
- LangGraph,
- llama-parse,
- llama-cloud-services,
- Pydantic V1.

## Events

Common stream events:

- `progress`: long-running stage update,
- `token`: response text delta,
- `chunk`: alternate text delta name supported by the platform/frontend,
- `sources`: source chunks used,
- `artifact`: artifact metadata as soon as available,
- `done`: final `GeneratorOutput`,
- `error`: failure message from `safe_sse_stream()`.

`teacher_gen` also emits:

- `analysis`: query intent, confusion score, target concept, and selected teacher mode.

## Artifact Patterns

| Generator | Artifact behavior |
| --- | --- |
| `teacher_gen` | No artifact; text response only. |
| `quiz_gen` | Uploads `quiz.json` to MinIO when storage is reachable. |
| `podcast_gen` | Uploads transcript text and MP3 audio when TTS succeeds. Transcript generation remains useful even if audio is skipped. |
| `mindmap_gen` | Writes JSON and standalone HTML to its artifact directory and serves them under `/artifacts`. |

Artifact records include `type`, `url`, `filename`, and optionally `key`. MinIO-backed artifacts include `key` so the platform can re-sign expiring URLs later.

## Runtime Options

Common options:

| Option | Purpose |
| --- | --- |
| `language` | Forces the output language. |
| `llm.enabled` | Enables runtime provider override. |
| `llm.provider` | `ollama`, `openai`, `anthropic`, or `openai_compatible`. |
| `llm.model` | Provider model. |
| `llm.base_url` / `llm.api_base_url` | Provider endpoint. |
| `llm.api_key` | Provider key resolved by backend settings. |

Generator-specific options are documented in each service README.

## Local Development

From a generator directory:

```bash
pip install -e ../../packages/teacherlm_core
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port <port>
```

Through Docker Compose:

```bash
cd ../platform
docker compose up -d teacher_gen quiz_gen podcast_gen mindmap_gen
```

## Per-Generator Docs

- `teacher_gen/README.md`
- `quiz_gen/README.md`
- `podcast_gen/README.md`
- `mindmap_gen/README.md`
