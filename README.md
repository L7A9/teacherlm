# TeacherLM

TeacherLM is an AI study workspace for course materials. A student uploads files, waits for them to be parsed and indexed, then works with a grounded teacher chat and several generators that only use the uploaded material.

The current app can:

- Chat with a warm tutor grounded in the selected course files.
- Generate quizzes, podcasts, and mind maps from uploaded files.
- Build a generated course view from all course materials.
- Track learner progress, concept coverage, struggles, review windows, and mastery signals.
- Let the student choose which ready source files are used for chat and generators when more than one file is available.

Course Builder is the exception to source-file selection: it intentionally uses every course material in the conversation so the generated course remains complete.

## Current Status

Enabled generators in `generators_registry.json`:

| Generator | Output type | Port | Retrieval mode |
| --- | --- | ---: | --- |
| `teacher_gen` | `text` | 8001 | `semantic_topk` |
| `quiz_gen` | `quiz` | 8002 | `coverage_broad` |
| `podcast_gen` | `podcast` | 8007 | `narrative_arc` |
| `mindmap_gen` | `mindmap` | 8008 | `topic_clusters` |

Registered but disabled generators:

- `report_gen`
- `presentation_gen`
- `chart_gen`

## Architecture

TeacherLM is split into a platform layer, shared core package, and standalone generator services.

```text
teacherlm/
  generators_registry.json
  packages/
    teacherlm_core/
  platform/
    backend/
    frontend/
    docker-compose.yml
  generators/
    teacher_gen/
    quiz_gen/
    podcast_gen/
    mindmap_gen/
```

Main services:

- Frontend: Next.js app on port `3000`.
- Backend: FastAPI app on port `8000`.
- Worker: ARQ background worker for ingestion and course generation.
- Postgres: relational records, course content, learner state.
- Redis: ARQ queue and transient coordination.
- Qdrant: vector search.
- MinIO: uploaded files, parsed markdown, artifacts.
- LlamaCloud: document parsing through `llama-cloud`.
- Ollama or configured provider: generator LLM calls.

## Compatibility Rules

These project rules are enforced by the root `AGENTS.md`:

- Python `3.14+`.
- Do not use LangChain or LangGraph.
- Do not use `llama-parse` or `llama-cloud-services`.
- Use `llama-cloud >= 1.0`.
- Use Pydantic V2 only.
- Use FastAPI `>= 0.135`.
- Prefer `fastembed`.
- Use the Ollama Python library native `format=` argument for Ollama structured output.

The shared LLM wrapper also supports OpenAI, Anthropic, and OpenAI-compatible providers through runtime settings.

## Quick Start

From the repository root:

```bash
cp platform/.env.example platform/.env
# Edit platform/.env and set your LlamaCloud key and model settings.
./run.sh build
cd platform
./scripts/init.sh
```

Useful URLs:

- Frontend: `http://localhost:3000`
- Backend health: `http://localhost:8000/api/health`
- Backend readiness: `http://localhost:8000/api/health/ready`
- MinIO console: `http://localhost:9001`
- Teacher generator health: `http://localhost:8001/health`
- Quiz generator health: `http://localhost:8002/health`
- Podcast generator health: `http://localhost:8007/health`
- Mind map generator health: `http://localhost:8008/health`

Common commands:

```bash
./run.sh up
./run.sh logs
./run.sh ps
./run.sh stop
```

The reset script is destructive because it removes local service data:

```bash
cd platform
./scripts/reset.sh
```

## Ingestion Flow

1. The frontend uploads files to the backend.
2. Files are stored in MinIO and recorded in Postgres.
3. The ARQ worker parses files with LlamaCloud.
4. Parsed markdown is normalized into course documents, sections, concepts, and search chunks.
5. Dense vectors are written to Qdrant and sparse/BM25 data is kept for hybrid retrieval.
6. When all files are ready, TeacherLM rebuilds concepts, the learning map, the knowledge graph, and the generated course.

Uploaded files move through statuses such as `uploaded`, `queued`, `parsing`, `chunking`, `embedding`, `ready`, and `failed`.

## Source-File Selection

The frontend shows ready course files in the Sources sidebar.

- If there is one ready file, it is forced on.
- If there are multiple ready files, the student can check or uncheck which files should ground chat and generators.
- At least one ready file remains selected.
- Chat and generator requests send `source_file_ids`.
- Backend chat and generate routes validate the selection and filter retrieval by those file ids.
- Course Builder ignores this selection and uses all course materials.

## Retrieval Defaults

Current platform defaults:

| Setting | Default |
| --- | ---: |
| `RETRIEVAL_TOP_K` | 16 |
| `RETRIEVAL_RERANK_TOP_K` | 16 |
| `RETRIEVAL_DENSE_CANDIDATE_K` | 80 |
| `RETRIEVAL_SPARSE_CANDIDATE_K` | 80 |
| `RETRIEVAL_RERANK_CANDIDATE_K` | 50 |

Mode mapping:

| Output | Retrieval mode |
| --- | --- |
| Chat / teacher text | `semantic_topk` |
| Quiz | `coverage_broad` |
| Podcast | `narrative_arc` |
| Mind map | `topic_clusters` |
| Report / presentation | `topic_clusters` |
| Chart / diagram | `relationship_dense` |

The backend can expand retrieved context with neighboring chunks, section summaries, equations, tables, timelines, concept maps, and topic clusters depending on the requested output type.

## Frontend Notes

The active workspace is built around three surfaces:

- Sources sidebar for uploads, ingestion status, and source-file selection.
- Generated course pane for CourseBuilder output.
- Chat pane for teacher chat and generator launch buttons.

On mobile and tablet layouts, the top-right toggle switches between the generated course interface and the chat interface instead of opening chat as an overlay. This keeps generator dialogs usable on small screens.

The generated course interface uses chapter accordions. A chapter can be opened and closed by clicking it. Subchapters are shown directly under the chapter summary, and opening one subchapter closes the other open subchapter in that chapter.

## API Surfaces

Main backend routers:

- `/api/health`
- `/api/conversations`
- `/api/conversations/{conversation_id}/files`
- `/api/conversations/{conversation_id}/chat`
- `/api/conversations/{conversation_id}/generate`
- `/api/generators`
- `/api/conversations/{conversation_id}/coursebuilder`
- `/api/conversations/{conversation_id}/course-player`
- `/api/conversations/{conversation_id}/knowledge-checks`
- `/api/conversations/{conversation_id}/knowledge-graph`
- `/api/conversations/{conversation_id}/review-tests`
- `/api/runtime-settings`

Generator services expose:

- `GET /health`
- `GET /info`
- `POST /run`

Chat and generation use server-sent events. Common event names include `token`, `chunk`, `sources`, `artifact`, `progress`, `done`, and `error`.

## Tests And Checks

Run the full configured test set from the repository root:

```bash
pytest
```

Configured test paths are:

- `packages/teacherlm_core/tests`
- `generators/teacher_gen/tests`
- `generators/mindmap_gen/tests`
- `platform/backend/tests`

Useful backend evaluation scripts are documented in `platform/backend/evals/README.md`.

## Documentation Map

- `platform/README.md`: platform services, environment, Docker, API, operations.
- `packages/teacherlm_core/README.md`: shared schemas, LLM wrappers, retrieval, confidence utilities.
- `generators/teacher_gen/README.md`: teacher chat generator.
- `generators/quiz_gen/README.md`: quiz generator.
- `generators/podcast_gen/README.md`: podcast generator.
- `generators/mindmap_gen/README.md`: mind map generator.
- `platform/backend/evals/README.md`: retrieval and course-context evaluation scripts.
