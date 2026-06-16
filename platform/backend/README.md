# TeacherLM Backend

The backend is the FastAPI API server for TeacherLM. It owns ingestion orchestration, retrieval policy, learner state, generator dispatch, CourseBuilder, knowledge checks, review tests, and runtime settings.

The most important boundary is this: generators do not own RAG. The backend prepares `context_chunks`, builds `GeneratorInput`, dispatches to an enabled generator service, streams the result back to the frontend, and persists the final assistant turn.

## Runtime Role

Default port: `8000`

Main responsibilities:

- conversation, message, and file APIs,
- upload storage in MinIO,
- ARQ background ingestion jobs,
- LlamaCloud parsing through `llama-cloud >= 1.0`,
- cleaned course-document extraction,
- deterministic section-based chunking,
- optional generated search questions for chunk metadata,
- dense vector indexing in Qdrant with fastembed,
- BM25, hybrid retrieval, RRF, reranking, graph search, and context expansion,
- source-file filtering for chat and generator requests,
- learner-state loading and update merging,
- CourseBuilder course generation,
- knowledge checks, review tests, and remediation paths,
- runtime LLM/parser settings with encrypted provider API keys,
- generator registry loading and HTTP SSE proxying.

## Technologies And Why

| Technology | Why |
| --- | --- |
| Python 3.14+ | Project compatibility target; avoids libraries that still depend on older Pydantic behavior. |
| FastAPI >= 0.135 | Async API server, dependency injection, Pydantic V2 request/response models. |
| Uvicorn | ASGI server for local and Docker runtime. |
| sse-starlette | Streams chat/generator events to the frontend as `text/event-stream`. |
| Pydantic V2 / pydantic-settings | Strict schemas and environment-driven settings. |
| SQLAlchemy 2 async + asyncpg | Async PostgreSQL access for conversations, messages, files, course content, learner state, graph, and generated course records. |
| Alembic | Database migrations under `db/migrations`. |
| PostgreSQL | Source of truth for structured app and course data. |
| Redis + ARQ | Background ingestion and CourseBuilder jobs without blocking the API server. |
| MinIO | S3-compatible storage for uploads, parsed markdown, cleaned text, and artifacts. |
| Qdrant | Dense vector search over per-conversation chunk collections. |
| fastembed | Local embeddings and cross-encoder reranking without sentence-transformers. |
| rank-bm25 through `teacherlm_core` | Exact-term lexical retrieval for acronyms, formulas, headings, and generated questions. |
| llama-cloud >= 1.0 | Supported parser API; deprecated llama-parse packages are intentionally not used. |
| httpx / httpx-sse | HTTP generator dispatch and SSE client support. |
| cryptography | Encrypts runtime provider API keys stored in Postgres. |
| `teacherlm_core` | Shared schemas, LLM client, retrieval primitives, prompts, and confidence utilities. |

Non-goals:

- no LangChain,
- no LangGraph,
- no llama-parse,
- no direct generator access to Qdrant or upload parsing.

## App Startup

`main.py` creates the FastAPI app and registers CORS plus routers.

During lifespan startup it:

1. loads `generators_registry.json` and fails fast if malformed,
2. tries to ensure the MinIO bucket exists,
3. starts retrieval reranker warmup in the background when enabled,
4. opens the ARQ Redis pool,
5. registers shutdown cleanup for Redis, Qdrant/vector service, and SQLAlchemy.

If storage or Redis warmup fails, the backend logs the error and keeps read-only or non-upload routes available where possible.

## Routers

| Router | Prefix | Purpose |
| --- | --- | --- |
| `health.py` | `/api/health` | liveness and readiness checks. |
| `conversations.py` | `/api/conversations` | CRUD, messages, learner state. |
| `files.py` | `/api/conversations/{conversation_id}/files` | upload, list, retry, delete uploaded files. |
| `chat.py` | `/api/conversations/{conversation_id}/chat` | default teacher chat via SSE. |
| `generate.py` | `/api/conversations/{conversation_id}/generate` | non-chat output generators via SSE. |
| `generators.py` | `/api/generators` | registry-backed generator catalog. |
| `coursebuilder.py` | `/api/conversations/{conversation_id}/coursebuilder` | generated course, progress events, chapter quizzes. |
| `course_player.py` | `/api/conversations/{conversation_id}/course-player` | older/generated course-player flow. |
| `knowledge_checks.py` | `/api/conversations/{conversation_id}/knowledge-checks` | adaptive checks and quiz attempt grading. |
| `knowledge_graph.py` | `/api/conversations/{conversation_id}/knowledge-graph` | graph read/rebuild/remediation endpoints. |
| `review_tests.py` | `/api/conversations/{conversation_id}/review-tests` | spaced review windows. |
| `runtime_settings.py` | `/api/settings/runtime` | provider and parser settings. |

## Generator Connection Flow

The backend connects generators to the platform through the registry, router, and transport adapter.

1. `dispatcher.registry.GeneratorRegistry` loads `generators_registry.json`.
2. Chat uses `chat_default()`; generate uses `for_output_type(output_type)`.
3. Disabled registry entries are rejected before dispatch.
4. The backend loads chat history and `LearnerState`.
5. Runtime settings are resolved into `options`, including optional provider overrides under `options["llm"]`.
6. `source_file_ids` are normalized; an explicit empty list is rejected.
7. `RetrievalOrchestrator.retrieve_for()` chooses the retrieval/context policy for the output type.
8. The backend builds `teacherlm_core.schemas.GeneratorInput`.
9. `GeneratorRouter` chooses an adapter from the registry entry type.
10. Enabled generators currently use `ApiAdapter`, which posts JSON to the generator `POST /run` endpoint and reads SSE with `httpx-sse`.
11. Generator events are proxied to the frontend.
12. On `done`, the backend persists the assistant message, artifacts, sources, and learner updates.
13. `LearnerTracker.apply_updates()` merges generator updates into canonical learner state.

The `McpAdapter` exists as a placeholder for future plugin-style generators. It mirrors the API adapter surface, but it intentionally raises `NotImplementedError` today.

## SSE Contract

Generator streams may emit:

- `analysis`: teacher-specific query analysis metadata,
- `token`: streamed text deltas,
- `chunk`: alternate streamed text event name,
- `sources`: source chunk list,
- `artifact`: one artifact record as soon as it is available,
- `progress`: long-running stage updates,
- `done`: final `GeneratorOutput`,
- `error`: recoverable stream failure.

The backend preserves event names and JSON-serializes event data for the frontend.

## Ingestion Pipeline

`workers/ingestion_worker.py` owns the upload processing job.

For each uploaded file:

1. mark the file `parsing`,
2. read upload bytes from MinIO,
3. resolve parser API key from runtime settings,
4. parse to markdown with `llama-cloud`,
5. store raw parsed markdown,
6. clean parser/slide noise with `DocumentCleaningService`,
7. normalize course intake with `CourseIntakeNormalizer`,
8. store cleaned text,
9. extract structured course documents and sections,
10. chunk sections with `ChunkingService`,
11. optionally annotate chunks with generated search questions,
12. replace document/section/chunk records in Postgres,
13. delete old vectors for the source file,
14. embed and upsert chunks to Qdrant,
15. mark the file `ready`,
16. when every file in the conversation is ready, rebuild concept inventory, learning map, knowledge graph, and CourseBuilder.

Worker startup also recovers files left in interrupted statuses and requeues interrupted CourseBuilder jobs.

## Course Data Model

Important SQLAlchemy records in `db/models.py` include:

- `Conversation`, `Message`,
- `UploadedFile`,
- `CourseDocumentRecord`, `CourseSectionRecord`, `SearchChunkRecord`,
- `CourseConceptRecord`, `CourseLearningPhaseRecord`, `CourseLearningObjectiveRecord`,
- `CourseKnowledgeNodeRecord`, `CourseKnowledgeEdgeRecord`, `CourseGraphRebuildRecord`,
- Course player records,
- CourseBuilder course/chapter/lesson/block/quiz/progress records,
- knowledge check and attempt records,
- review window and answered question records,
- `LearnerStateRecord`,
- `AppRuntimeSettingsRecord`.

Postgres is authoritative. Qdrant stores vector copies for fast retrieval, and MinIO stores binary/text objects.

## Retrieval Techniques

The backend uses `RetrievalOrchestrator` plus `CourseContextService`.

Core techniques:

- low-information chunk filtering,
- source-file filtered corpus loading,
- dense semantic search in Qdrant,
- BM25 over text plus heading path, section title, key concepts, and generated questions,
- reciprocal rank fusion with `RRF_K = 60`,
- formula/equation intent detection and formula hit boosts,
- comparison query detection and balanced per-term retrieval,
- graph candidate retrieval from matching knowledge graph nodes,
- fastembed cross-encoder reranking,
- comparison-aware group reranking,
- neighbor expansion,
- section summary expansion,
- graph neighbor expansion,
- output-specific context policies.

Mode mapping:

| Output type | Mode |
| --- | --- |
| `text` / `chat` | `semantic_topk` |
| `quiz` | `coverage_broad` |
| `podcast` | `narrative_arc` |
| `mindmap` | `topic_clusters` |
| disabled `report` | `topic_clusters` |
| disabled `presentation` | `topic_clusters` |
| disabled `chart` / `diagram` | `relationship_dense` |

See `docs/rag_course_content.md` for the complete RAG reference.

## Source-File Selection

The frontend sends `source_file_ids` for chat and generator requests.

The backend applies that filter to:

- documents,
- sections,
- chunks,
- graph candidates,
- graph neighbors,
- equations,
- tables,
- mind map module packs,
- representative course context.

`None` means no explicit filter. An explicit empty list is invalid. CourseBuilder ignores this filter and always uses all files in a conversation.

## Learner State

Before generator dispatch the backend loads `LearnerState`, including:

- simple lists: understood and struggling concepts,
- mastery scores,
- known concepts,
- concept progress,
- learning phases,
- objective progress,
- phase progress,
- remediation paths.

Generators report:

- `concepts_covered`,
- `concepts_demonstrated`,
- `concepts_struggled`.

Chat/generation persistence applies conservative encounter-style updates. Knowledge checks, course quizzes, and review tests can apply stronger mastery changes.

## CourseBuilder

CourseBuilder turns all ready conversation files into an ordered generated course.

Pipeline stages:

- queued,
- analyzing,
- generating outline,
- generating chapters,
- generating lessons,
- generating quizzes,
- validating,
- ready or failed.

CourseBuilder uses:

- parsed markdown planning context,
- source structure from intake metadata, markdown headings, table-of-contents patterns, sections, and chunk headings,
- concept inventory and learning map,
- grounded chunk retrieval per chapter, lesson, and block,
- lesson block validation,
- fallback lesson blocks when LLM output is thin,
- chapter quizzes with source citations,
- progress events for frontend polling.

`LOCK_COURSEBUILDER_CHAPTERS` is currently `False`, so chapters are open by default.

## Runtime Settings

`RuntimeSettingsService` supports:

- local Ollama,
- OpenAI,
- Anthropic,
- OpenAI-compatible APIs,
- parser API key storage.

Provider API keys are stored encrypted when `SETTINGS_ENCRYPTION_KEY` is configured. Resolved settings flow to generators through `GeneratorInput.options`.

## Local Development

From the repo root:

```bash
pip install -e packages/teacherlm_core
pip install -r platform/backend/requirements.txt
cd platform/backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

Run backend tests:

```bash
pytest platform/backend/tests
```

Useful scripts:

```bash
python platform/backend/scripts/evaluate_retrieval.py run platform/backend/evals/example_retrieval_eval.json
python platform/backend/scripts/eval_course_context.py <conversation-id>
python platform/backend/scripts/compare_retrieval_variants.py platform/backend/evals/example_retrieval_eval.json
python platform/backend/scripts/reindex_from_parsed.py <uploaded-file-id>
```
