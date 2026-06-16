# TeacherLM

TeacherLM is an AI study workspace for uploaded course materials. A student uploads files, TeacherLM parses and indexes them, then the student gets a personal tutor plus grounded output generators for quizzes, podcasts, and mind maps.

The product rule is simple: every teaching answer and generated artifact must be grounded in uploaded course files.

## Non-Negotiable Compatibility

TeacherLM is intentionally built without agent/orchestration frameworks that conflict with the target runtime.

- Python `3.14+`.
- Pydantic V2 only.
- FastAPI `>= 0.135`.
- `llama-cloud >= 1.0` for parsing.
- No LangChain.
- No LangGraph.
- No llama-parse.
- No llama-cloud-services.
- Prefer fastembed over sentence-transformers.
- Use the Ollama Python library native `format=` argument for local structured output.

## What The App Does

- Chats with a warm teacher voice grounded in selected source files.
- Generates quizzes, podcasts, and mind maps from uploaded course material.
- Builds a generated course view from all materials in the conversation.
- Tracks concepts covered, understood concepts, struggling concepts, mastery signals, review windows, and learner progress.
- Lets the student select which ready source files should ground chat and generators when more than one file is ready.

Course Builder is the exception to source-file selection. It intentionally uses every course file in the conversation so the generated course remains complete.

## Current Generators

Enabled generators in `generators_registry.json`:

| Generator | Output type | Port | Retrieval mode | Role |
| --- | --- | ---: | --- | --- |
| `teacher_gen` | `text` | 8001 | `semantic_topk` | Default teacher chat, explanations, guidance, confidence, learner updates |
| `quiz_gen` | `quiz` | 8002 | `coverage_broad` | Grounded quiz planning, question generation, validation, quiz artifact |
| `podcast_gen` | `podcast` | 8007 | `narrative_arc` | Two-host educational transcript and optional MP3 audio |
| `mindmap_gen` | `mindmap` | 8008 | `topic_clusters` | Interactive course mind map JSON and HTML artifacts |

Registered but disabled generators:

- `report_gen`
- `presentation_gen`
- `chart_gen`

They stay visible as registry entries only; this repository currently documents the four implemented generators above.

## Implemented Vs Reserved

Implemented and running in Docker Compose:

- `teacher_gen`
- `quiz_gen`
- `podcast_gen`
- `mindmap_gen`

Implemented platform support but not standalone enabled generator services:

- output-type policies for `report`, `presentation`, and `chart`,
- frontend renderers for chart/diagram artifacts,
- registry entries for disabled report/presentation/chart generators.

The disabled entries are useful for future work because the backend already knows which retrieval mode each output type should use. They are not currently user-runnable generators.

## Architecture

TeacherLM is split into a platform layer, a shared core package, and standalone generator services.

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

Main runtime services:

| Service | Technology | Purpose |
| --- | --- | --- |
| Frontend | Next.js, React, React Query, Zustand | Workspace UI, uploads, chat, generator dialogs, artifact rendering |
| Backend | FastAPI, Pydantic V2, SQLAlchemy async | API, routing, retrieval orchestration, learner state, generator dispatch |
| Worker | ARQ, Redis | Background ingestion, parsing, indexing, concept/map/graph/course rebuilds |
| Postgres | PostgreSQL | Conversations, messages, files, course documents, sections, chunks, learner state, graph records |
| Redis | Redis | ARQ queue and background job coordination |
| Qdrant | Qdrant | Dense vector index per conversation |
| MinIO | S3-compatible object storage | Uploaded files, parsed markdown, cleaned text, generator artifacts |
| Parser | `llama-cloud >= 1.0` | Converts uploaded files to markdown |
| LLM runtime | Ollama or configured provider | Structured extraction, teacher answers, generator planning |
| Shared core | `teacherlm_core` | Stable schemas, retrieval primitives, LLM wrappers, prompts, confidence scoring |

## End-To-End Flow

1. The frontend creates a conversation and uploads one or more course files.
2. The backend stores the original file in MinIO and creates a Postgres `UploadedFile` record.
3. The ARQ worker parses the file with `llama-cloud`, stores parsed markdown, cleans parser/slide noise, and extracts course structure.
4. Structured documents, sections, and deterministic search chunks are stored in Postgres.
5. Search chunks are embedded with fastembed and written to a Qdrant collection for that conversation.
6. Chunk metadata also feeds BM25 keyword search, generated question metadata, formulas, tables, timelines, concepts, and section summaries.
7. Once files are ready, the worker rebuilds the concept inventory, learning map, knowledge graph, and generated course.
8. Chat or generator requests load learner state, apply selected source-file filters, retrieve course context, and dispatch a `GeneratorInput` to the selected generator.
9. Generator services stream SSE events back through the backend. The backend persists the assistant message, artifacts, sources, and learner updates.

## How Generators Connect To The Platform

The connection is registry-driven and intentionally simple.

1. `generators_registry.json` declares each generator id, output type, endpoint, enabled flag, icon, and default-chat status.
2. The backend loads that registry on startup and exposes enabled generators at `/api/generators`.
3. The frontend shows output-type buttons from the enabled registry entries and sends requests to the backend only.
4. Chat calls `/api/conversations/{conversation_id}/chat`, which resolves the default text generator.
5. Output generation calls `/api/conversations/{conversation_id}/generate`, which resolves a generator by `output_type`.
6. The backend loads chat history, learner state, runtime LLM options, and selected source files.
7. Retrieval happens in the backend through `RetrievalOrchestrator`.
8. The backend builds the immutable `GeneratorInput` contract from `teacherlm_core`.
9. The HTTP `ApiAdapter` posts that JSON to the generator `POST /run` endpoint inside the Compose network.
10. The generator streams SSE events back to the backend.
11. The backend proxies events to the frontend and persists the final `GeneratorOutput` on `done`.
12. Learner updates are merged by the backend, not by the generator.

This keeps generators stateless and replaceable. They receive evidence, produce output, report sources/artifacts/learner updates, and never directly touch upload parsing, Qdrant, Postgres learner records, or source-file filtering.

## RAG In One Page

TeacherLM uses retrieval-augmented generation at the platform layer. Generators do not query Qdrant directly; they receive already-filtered `context_chunks` from the backend.

The retrieval stack combines several techniques:

- Structured chunking: chunks are created from extracted course sections rather than raw text. Metadata keeps headings, section IDs, neighbor IDs, summaries, key concepts, formulas, tables, and source file IDs.
- Semantic search: fastembed embeds the query and Qdrant returns dense vector candidates by meaning.
- Keyword search: `rank-bm25` scores exact term matches over chunk text plus searchable metadata such as headings, key concepts, and generated questions.
- RRF fusion: dense and BM25 rankings are merged with reciprocal rank fusion so meaning and exact terminology reinforce each other.
- Reranking: a fastembed cross-encoder reranker refines candidate order for all retrieval modes by default.
- Graph search: knowledge graph nodes and edges add chunks connected to matching concepts, objectives, formulas, skills, examples, and prerequisites.
- Context expansion: final chunks can be expanded with section summaries, local neighbor chunks, graph neighbors, equations, tables, or course outlines depending on output type.

Retrieval modes:

| Mode | Used by | Why |
| --- | --- | --- |
| `semantic_topk` | Teacher chat | Focuses tightly on a student's question while still using BM25, RRF, reranking, graph candidates, and neighbor expansion |
| `coverage_broad` | Quiz | Samples across the selected course material so quizzes test breadth, not only the top matching paragraph |
| `narrative_arc` | Podcast | Adds introductory, middle, and concluding context so the script has a teachable flow |
| `topic_clusters` | Mind map, disabled report/presentation entries | Groups course topics so overview outputs cover the subject structure |
| `relationship_dense` | Disabled chart/diagram entry | Prefers chunks with entities, relations, formulas, processes, comparisons, and numeric facts |

The full reference is in `platform/backend/docs/rag_course_content.md`.

## Compatibility Rules

These rules are non-negotiable for the project:

- Python `3.14+`.
- Do not use LangChain or LangGraph.
- Do not use deprecated LlamaParse packages.
- Use `llama-cloud >= 1.0`.
- Use Pydantic V2 only, currently `>= 2.12`.
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
- Runtime settings: `http://localhost:8000/api/settings/runtime`
- MinIO console: `http://localhost:9001`
- Teacher generator health: `http://localhost:8001/health`
- Quiz generator health: `http://localhost:8002/health`
- Podcast generator health: `http://localhost:8007/health`
- Mind map generator health: `http://localhost:8008/health`

Common commands:

```bash
./run.sh up
./run.sh build
./run.sh rebuild
./run.sh logs
./run.sh ps
./run.sh shell backend
./run.sh stop
./run.sh down
./run.sh report-charts
```

`run.sh` auto-detects a LAN host, configures browser-facing API/MinIO/mindmap URLs for Compose, and prints a phone-friendly URL for testing on another device on the same Wi-Fi.

The reset script removes local platform data:

```bash
cd platform
./scripts/reset.sh
```

## Frontend Workflow

The workspace has three main surfaces:

- Sources sidebar: upload files, watch ingestion status, retry failed files, and choose which ready files ground chat/generators.
- Course pane: generated course chapters, lessons, subchapters, chapter quizzes, and progress.
- Chat pane: teacher chat plus output-type buttons for implemented generators.

The frontend streams chat/generation with manual POST-SSE parsing because native `EventSource` only supports GET. It renders artifacts inline when possible:

- quiz JSON with an interactive quiz renderer,
- mind map JSON with Markmap,
- podcast audio with transcript support,
- generic files with download controls.

Current UI behavior by generator:

| Output | UI behavior |
| --- | --- |
| Teacher chat | Chat input sends the student's exact message to `/chat`. |
| Quiz | Dialog exposes question count, difficulty, and one quiz type. It hides topic so the quiz covers selected ready source files broadly. |
| Podcast | Dialog currently hides podcast fields, so UI calls use service defaults plus selected source files and forced language. |
| Mind map | Dialog uses standard size, `max_nodes: 110`, `llm_refine: true`, and a fresh generation id. Topic is hidden for broad overview maps. |

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
- `/api/settings/runtime`

Generator services expose:

- `GET /health`
- `GET /info`
- `POST /run`

Common SSE event names are preserved across the platform:

- `token`
- `chunk`
- `sources`
- `artifact`
- `progress`
- `done`
- `error`

## Tests And Checks

Run the configured test set from the repository root:

```bash
pytest
```

Configured test paths include:

- `packages/teacherlm_core/tests`
- `generators/teacher_gen/tests`
- `generators/mindmap_gen/tests`
- `platform/backend/tests`

Useful retrieval and context evaluation scripts are documented in `platform/backend/evals/README.md`.

## Documentation Map

- `platform/README.md`: platform services, environment, ingestion, retrieval orchestration, APIs, operations.
- `platform/backend/README.md`: backend routers, ingestion worker, retrieval, generator dispatch, learner state, CourseBuilder.
- `platform/frontend/README.md`: frontend API layer, POST-SSE streaming, Zustand stores, workspace layout, source selection, artifact rendering.
- `platform/backend/docs/rag_course_content.md`: deep RAG architecture and technique reference.
- `platform/backend/evals/README.md`: retrieval and course-context evaluation scripts.
- `packages/teacherlm_core/README.md`: shared schemas, LLM wrappers, retrieval primitives, confidence utilities, prompts.
- `generators/README.md`: generator-service contract, registry connection, shared events, artifact patterns.
- `generators/teacher_gen/README.md`: teacher chat generator.
- `generators/quiz_gen/README.md`: quiz generator.
- `generators/podcast_gen/README.md`: podcast generator.
- `generators/mindmap_gen/README.md`: mind map generator.
