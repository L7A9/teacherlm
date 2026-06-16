# TeacherLM Platform

The platform contains the FastAPI backend, Next.js frontend, ARQ worker, Docker Compose stack, and operational scripts that turn uploaded course files into grounded teacher interactions and generator outputs.

The platform owns retrieval. Generator services receive prepared `context_chunks`; they do not parse files or query Qdrant directly.

## Services

`docker-compose.yml` starts:

| Service | Purpose | Port |
| --- | --- | ---: |
| `frontend` | Next.js workspace UI | 3000 |
| `backend` | FastAPI API server, retrieval orchestration, generator dispatch | 8000 |
| `arq_worker` | file ingestion, indexing, concept/map/graph/course rebuild jobs | internal |
| `postgres` | relational records, course content, learner state, graph records | 5432 |
| `redis` | ARQ queue and transient job coordination | 6379 |
| `qdrant` | dense vector search | 6333 |
| `minio` | uploaded files, parsed markdown, cleaned text, artifacts | 9000 / 9001 |
| `teacher_gen` | teacher chat generator | 8001 |
| `quiz_gen` | quiz generator | 8002 |
| `podcast_gen` | podcast generator | 8007 |
| `mindmap_gen` | mind map generator | 8008 |

The backend and generator containers use the host Ollama endpoint by default:

```text
http://host.docker.internal:11434
```

## Technology Inventory

Backend stack:

| Technology | Why it is used |
| --- | --- |
| Python 3.14+ | Project compatibility target. |
| FastAPI, Uvicorn, sse-starlette | Async HTTP API and SSE streams. |
| Pydantic V2, pydantic-settings | Request/response schemas and environment configuration. |
| SQLAlchemy 2 async, asyncpg, Alembic | PostgreSQL models, async sessions, migrations. |
| PostgreSQL | Authoritative app/course/learner data. |
| Redis, ARQ | Background ingestion and CourseBuilder jobs. |
| MinIO, aiofiles | Uploaded files, parsed markdown, cleaned text, generated artifacts. |
| llama-cloud >= 1.0 | Supported document parser. Deprecated llama-parse packages are not used. |
| Qdrant, fastembed | Dense vector search and local embeddings. |
| rank-bm25 via `teacherlm_core` | Exact-term lexical retrieval. |
| fastembed cross-encoder reranker | Final relevance ordering after hybrid candidate retrieval. |
| httpx, httpx-sse | Generator API calls and SSE proxying. |
| cryptography | Runtime provider API-key encryption. |

Frontend stack:

| Technology | Why it is used |
| --- | --- |
| Next.js 14, React 18, TypeScript | Typed workspace app and standalone production build. |
| React Query | Backend data, artifact JSON loading, cache invalidation. |
| Zustand | Streaming state, UI state, selected source files, progress mirror, persisted theme/language. |
| Tailwind CSS, Radix UI, lucide-react | Local UI system, accessible dialogs/tooltips, icons. |
| react-dropzone | File upload UI. |
| react-markdown, remark-gfm, remark-math, rehype-katex, Katex | Markdown, tables, and math rendering. |
| markmap-lib, markmap-view | Interactive mind map artifacts. |
| mermaid, svg-pan-zoom | Diagram rendering and pan/zoom controls. |
| react-pdf, pdfjs-dist | PDF viewing support. |
| sonner | Toast notifications. |

Generator stack:

| Generator | Additional technologies |
| --- | --- |
| `teacher_gen` | teacher prompts, structured analysis, confidence scoring, LLM fallback. |
| `quiz_gen` | Bloom-level planning, quality validation, optional fastembed distractors, MinIO quiz JSON. |
| `podcast_gen` | Piper, Kokoro, pyttsx3, langdetect, pydub, ffmpeg, espeak-ng, NLTK, MinIO artifacts. |
| `mindmap_gen` | recursive Pydantic hierarchy, Jinja2 standalone HTML, Markmap markdown artifacts. |

## Setup

From the repository root:

```bash
cp platform/.env.example platform/.env
# Edit platform/.env before starting the stack.
./run.sh build
cd platform
./scripts/init.sh
```

`scripts/init.sh` pulls default Ollama models, creates the MinIO bucket, and runs Alembic migrations.

Useful runtime commands from the repository root:

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

`run.sh up` and `run.sh build` auto-detect a LAN host, set browser-facing API/MinIO/mindmap URLs, rebuild the frontend when needed, start Compose, and print both localhost and phone-friendly URLs.

The reset script removes local platform data:

```bash
cd platform
./scripts/reset.sh
```

## Environment Groups

Important configuration groups:

| Group | Variables |
| --- | --- |
| Database | `DATABASE_URL` |
| Queue | `REDIS_URL` |
| Vector store | `QDRANT_URL`, `QDRANT_API_KEY` |
| Object storage | `MINIO_ENDPOINT`, `MINIO_PUBLIC_ENDPOINT`, `MINIO_*` |
| Parser | `LLAMA_CLOUD_API_KEY`, `LLAMA_CLOUD_BASE_URL`, parser timeout/poll settings |
| Runtime settings | `SETTINGS_ENCRYPTION_KEY` |
| Embeddings | `EMBEDDING_MODEL`, `EMBEDDING_DIM`, `EMBEDDING_BATCH_SIZE` |
| Chunking | `CHUNK_MAX_TOKENS`, `CHUNK_OVERLAP_TOKENS`, chunk-question settings |
| Retrieval | `RETRIEVAL_*`, `COURSE_CONTEXT_*` |
| Worker limits | `INGESTION_*` |
| Ollama defaults | `OLLAMA_HOST`, `OLLAMA_CHAT_MODEL` |

Runtime model settings can also be configured through:

```text
GET/PATCH /api/settings/runtime
```

Supported provider types:

- `ollama`
- `openai`
- `anthropic`
- `openai_compatible`

Provider API keys stored through runtime settings are encrypted with `SETTINGS_ENCRYPTION_KEY`.

## Backend App

The backend lives in `platform/backend`.

On startup it:

1. loads `generators_registry.json` and fails fast if it is malformed,
2. ensures the MinIO bucket exists,
3. starts reranker warmup in the background when enabled,
4. opens the ARQ Redis pool,
5. registers API routers.

Included routers:

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

## File Ingestion

Uploaded files are ingested by the ARQ worker. The worker is deliberately separate from the API server so parsing, embedding, and course rebuilds do not block chat requests.

Pipeline:

1. Store the original upload in MinIO and create a Postgres file record.
2. Parse file bytes to markdown with `llama-cloud >= 1.0`.
3. Store raw parsed markdown in MinIO.
4. Clean parser and slide boilerplate locally.
5. Store cleaned text in MinIO.
6. Extract a structured course document: document title, sections, heading paths, summaries, key concepts, equations, tables, and timeline events.
7. Build deterministic search chunks from structured sections.
8. Optionally annotate chunks with generated search questions to improve BM25 keyword recall.
9. Replace Postgres course document, section, and search chunk records for the file.
10. Delete old vectors for that file and upsert new fastembed vectors to Qdrant.
11. Mark the file `ready`.
12. When all files are ready, rebuild the concept inventory, learning map, knowledge graph, and CourseBuilder course.

The parser intentionally uses `llama-cloud`. Deprecated LlamaParse packages are not used.

## Course Content Model

Postgres is the source of truth for course structure:

- `course_documents`: one parsed upload, source filename, object keys, title, text hash, metadata.
- `course_sections`: ordered section hierarchy, text, summary, key concepts, equations, tables, timeline events.
- `search_chunks`: deterministic searchable units linked to documents and sections, with token counts, heading paths, source file IDs, and neighbor IDs.

Qdrant stores dense vectors and payloads for search, but Postgres keeps the authoritative structure and metadata needed for context expansion, citations, filtering, and rebuilds.

## Retrieval Orchestration

Retrieval is split into two layers:

- `teacherlm_core` provides reusable primitives: BM25, hybrid dense+sparse retrieval, reciprocal rank fusion, reranking, retrieval modes, confidence utilities.
- `platform/backend/services/retrieval_orchestrator.py` applies product policy: output-type mode selection, source-file filters, course-overview detection, graph candidates, comparison balancing, section summaries, equations/tables, reranking, and neighbor expansion.

Default retrieval values:

| Setting | Default |
| --- | ---: |
| `RETRIEVAL_TOP_K` | 16 |
| `RETRIEVAL_RERANK_TOP_K` | 16 |
| `RETRIEVAL_DENSE_CANDIDATE_K` | 80 |
| `RETRIEVAL_SPARSE_CANDIDATE_K` | 80 |
| `RETRIEVAL_RERANK_CANDIDATE_K` | 50 |
| `RETRIEVAL_RERANKER_MODEL` | `BAAI/bge-reranker-base` |

Output type to retrieval mode mapping:

| Output type | Mode | Context policy |
| --- | --- | --- |
| `text` / chat | `semantic_topk` | focused hybrid hits, graph candidates, reranking, neighbor expansion |
| `quiz` | `coverage_broad` | broad section/course coverage; topic context only when a topic is supplied |
| `podcast` | `narrative_arc` | course outline plus representative flow, or focused topic sections |
| `mindmap` | `topic_clusters` | course outline and module packs for whole-course overview |
| `report` | `topic_clusters` | disabled registry entry; policy reserved for future implementation |
| `presentation` | `topic_clusters` | disabled registry entry; adds equations/tables when implemented |
| `chart` / `diagram` | `relationship_dense` | disabled registry entry; relation-heavy facts when implemented |

For the complete RAG reference, see `backend/docs/rag_course_content.md`.

## Chat And Generation

There are two user-facing generation entry points:

```text
POST /api/conversations/{conversation_id}/chat
POST /api/conversations/{conversation_id}/generate
```

The chat endpoint always routes to the default `text` generator. The generate endpoint resolves an enabled generator by `output_type`.

Request flow:

1. Validate the conversation and requested generator.
2. Load recent chat history.
3. Load current learner state from Postgres.
4. Resolve runtime model/provider options.
5. Normalize and validate selected `source_file_ids`.
6. Retrieve platform-owned course context through the retrieval orchestrator.
7. Build a stable `GeneratorInput`.
8. Dispatch to the generator service using the registry endpoint.
9. Proxy generator SSE events to the frontend.
10. On `done`, persist the assistant message, sources, artifacts, and learner updates.

Common SSE event names:

- `token`
- `chunk`
- `sources`
- `artifact`
- `progress`
- `done`
- `error`

The backend supports both one-shot and streaming generator dispatch, but chat and UI-triggered generation use streaming.

## How Generators Connect To The Platform

The active connection is registry-driven HTTP SSE:

1. `generators_registry.json` defines every generator id, display name, output type, enabled flag, endpoint, icon, and whether it is the chat default.
2. The backend loads the registry on startup and exposes enabled entries through `/api/generators`.
3. The frontend uses the generator list to show output-type controls.
4. Chat always resolves the enabled `is_chat_default` generator, currently `teacher_gen`.
5. `/generate` resolves an enabled generator by `output_type`.
6. The backend resolves runtime LLM/parser options, learner state, history, and source-file filters.
7. `RetrievalOrchestrator` chooses the output-specific context policy and returns `context_chunks`.
8. The backend builds the stable `GeneratorInput` from `teacherlm_core`.
9. `ApiAdapter` posts that input to the registry endpoint, such as `http://quiz_gen:8002/run`.
10. The generator streams SSE events. The backend proxies them to the browser unchanged.
11. On `done`, the backend persists the final `GeneratorOutput`, artifact records, source chunks, and learner updates.

Enabled generators are stateless HTTP services. They can call an LLM and store artifacts, but they do not parse uploads, mutate learner records directly, or query Qdrant. The `McpAdapter` exists as a future transport placeholder and is not wired today.

## Source-File Selection

The frontend shows ready course files in the Sources sidebar.

- If no ready files exist, chat/generation has no source context.
- If one ready file exists, it is forced on.
- If multiple ready files exist, the student can check or uncheck files.
- At least one ready source remains selected.
- Chat and generator requests send `source_file_ids`.
- Backend routes reject an explicit empty source-file selection.
- Retrieval filters documents, sections, chunks, graph candidates, and context expansion by selected source files.
- CourseBuilder intentionally ignores this filter and uses all course files.

This lets students focus a chat or generator on a subset of files without corrupting the full generated course.

## Learner State

Before dispatching any generator, the backend loads `LearnerState` and includes it in `GeneratorInput`.

Generators return `learner_updates`:

- `concepts_covered`: concepts the output taught or tested,
- `concepts_demonstrated`: concepts the student showed understanding of,
- `concepts_struggled`: concepts where confusion appeared.

`learner_tracker.py` merges these into canonical progress records. Current chat/generation persistence applies encounter-style updates and keeps mastery updates conservative; review and assessment flows handle stronger mastery signals.

Learner state is used by:

- teacher chat response mode selection,
- quiz planning,
- review-test scheduling,
- knowledge graph remediation,
- progress panels in the frontend.

## CourseBuilder

CourseBuilder is the generated course interface used by the course pane.

It:

- waits for course files to finish ingestion,
- uses all course materials in the conversation,
- extracts or reconstructs chapter and subchapter structure,
- generates grounded lessons with citations,
- generates chapter quizzes,
- records progress events and validation status.

The frontend displays chapters as accordions, subchapters under the chapter summary, and chapter quizzes inside the generated course flow.

## Frontend App

The frontend lives in `platform/frontend`.

Key technologies:

- Next.js, React, and TypeScript for the app shell.
- React Query for server data and artifact JSON loading.
- Zustand for local workspace state, selected source files, streaming state, persisted theme/language, and progress state.
- Tailwind CSS plus local UI components, Radix primitives, and lucide icons.
- Manual POST-SSE parsing because native `EventSource` cannot POST.
- Markmap, Mermaid, Katex, react-markdown, and artifact renderers for generated outputs.
- react-dropzone for uploads and sonner for toast feedback.

Main workspace surfaces:

- Sources sidebar: upload, ingestion status, retry, source-file selection.
- Course pane: generated course content and course progress.
- Chat pane: teacher chat and generator launch buttons.
- Generated items sidebar: generated quiz, podcast, mind map, and downloadable artifacts.

On mobile and tablet, a top-right toggle switches between course and chat views while side panels become drawers.

See `frontend/README.md` for the detailed UI, state, streaming, and artifact-rendering reference.

## Artifact Handling

Artifacts are stored by generators or the platform and surfaced through message records.

- Quiz and podcast artifacts are uploaded to MinIO when storage configuration is available.
- Mind map artifacts are written to the generator artifact directory and served by `mindmap_gen` under `/artifacts`.
- Artifact records include `type`, `url`, `filename`, and optionally `key`.
- MinIO URLs may expire, so stored object keys allow the platform to re-sign URLs when serving history.

## Scripts

Backend scripts:

- `scripts/evaluate_retrieval.py`
- `scripts/eval_retrieval.py`
- `scripts/eval_course_context.py`
- `scripts/benchmark_embeddings.py`
- `scripts/compare_retrieval_variants.py`
- `scripts/reindex_from_parsed.py`

Platform scripts:

- `scripts/init.sh`
- `scripts/reset.sh`

Root script:

- `../run.sh`: stack start/build/rebuild/logs/shell/LAN URL/report chart helper.

See `backend/evals/README.md` for retrieval and course-context evaluation usage.

## Detailed Part READMEs

- `backend/README.md`: backend architecture, routers, ingestion, retrieval, generator dispatch, learner state, CourseBuilder.
- `frontend/README.md`: frontend API layer, POST-SSE streaming, stores, workspace layout, source selection, artifact renderers.
- `backend/evals/README.md`: retrieval evaluation files, commands, metrics, variant comparisons, report charts.

## Tests

From the repository root:

```bash
pytest platform/backend/tests
```

The root `pytest.ini` also includes shared-core and generator test directories.
