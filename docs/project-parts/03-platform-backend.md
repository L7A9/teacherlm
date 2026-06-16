# Platform Backend

`platform/backend` is the central application service. It owns HTTP APIs, database state, file ingestion orchestration, retrieval, generator dispatch, learner state, knowledge checks, review tests, and generated course surfaces.

The backend is built with FastAPI, Pydantic V2, SQLAlchemy async, PostgreSQL, Redis/ARQ, Qdrant, MinIO, llama-cloud, fastembed, and shared utilities from `teacherlm_core`.

## Backend Top-Level Files

| Path | Purpose |
| --- | --- |
| `platform/backend/main.py` | Creates the FastAPI app, configures CORS, starts and stops shared resources, reloads the generator registry, ensures MinIO bucket existence, warms retrieval, opens ARQ Redis, and includes all routers. |
| `platform/backend/config.py` | Backend Pydantic settings. Defines app metadata, CORS origins, database URL, Redis URL, Qdrant URL, MinIO credentials, llama-cloud settings, encryption key, embedding and reranker models, chunking settings, retrieval settings, course context settings, ingestion worker settings, registry path, and Ollama defaults. |
| `platform/backend/requirements.txt` | Python dependencies for API and worker. |
| `platform/backend/Dockerfile` | Python 3.14 backend image. Installs core package, backend requirements, optional model prefetches, and runs Uvicorn by default. |
| `platform/backend/alembic.ini` | Alembic configuration. |
| `platform/backend/CLAUDE.md` | Local assistant guidance for backend work. |
| `platform/backend/docs/rag_course_content.md` | Existing deep reference for course content and RAG architecture. |

## Application Lifecycle

`main.py` defines a FastAPI lifespan hook.

Startup:

1. Logs configuration and startup state.
2. Reloads the generator registry from `generators_registry.json`.
3. Ensures the MinIO bucket exists.
4. Starts retrieval warmup in the background.
5. Opens an ARQ Redis pool and stores it on app state.
6. Includes routers under the API prefix.

Shutdown:

1. Closes the ARQ Redis pool.
2. Closes vector service resources.
3. Disposes the SQLAlchemy async engine.

This is why API startup depends on object storage, registry readability, and Redis availability.

## Database Layer

### `db/session.py`

Defines async database access:

- Async SQLAlchemy engine.
- Async session factory.
- `get_db()` dependency for FastAPI routes.
- `session_scope()` helper for worker/service code.
- Automatic commit on successful dependency exit.
- Rollback on exceptions.
- `dispose_engine()` for clean shutdown.

### `db/models.py`

Defines the relational model. Major groups:

Conversations and messages:

- `Conversation`
- `Message`
- `UploadedFile`
- `LearnerStateRecord`

Runtime settings:

- `AppRuntimeSettingsRecord`

Course content:

- `CourseDocumentRecord`
- `CourseSectionRecord`
- `SearchChunkRecord`
- `CourseConceptRecord`
- `CourseLearningPhaseRecord`
- `CourseLearningObjectiveRecord`

Knowledge graph:

- `CourseKnowledgeNodeRecord`
- `CourseKnowledgeEdgeRecord`
- `CourseGraphRebuildRecord`

Older/adaptive course player:

- `CourseChapterRecord`
- `CourseLessonRecord`
- `CourseLessonBlockRecord`
- `ChapterQuizRecord`
- `ChapterAttemptRecord`

Course builder surface:

- `CourseBuilderCourseRecord`
- `CourseBuilderChapterRecord`
- `CourseBuilderLessonRecord`
- `CourseBuilderLessonBlockRecord`
- `CourseBuilderQuizRecord`
- `CourseBuilderQuizQuestionRecord`
- `CourseBuilderChapterAttemptRecord`
- `CourseBuilderProgressEventRecord`

Knowledge checks and review:

- `KnowledgeCheckRecord`
- `KnowledgeAttemptRecord`
- `AnsweredCourseQuestionRecord`
- `LearningReviewWindowRecord`

Common model patterns:

- UUID primary keys.
- Conversation-scoped cascade deletes.
- JSONB metadata fields.
- Status strings for long-running pipelines.
- Created/updated timestamps.
- Source chunk ID arrays where generated learning objects need traceability.

## Migrations

Migration files live under `platform/backend/db/migrations/`.

| Migration | Adds |
| --- | --- |
| `0001_initial.py` | Initial conversations, messages, files, and base tables. |
| `0002_course_content.py` | Parsed course documents, sections, and search chunks. |
| `0003_add_uploaded_file_summary.py` | Uploaded file summary metadata. |
| `0004_course_concepts.py` | Course concept inventory. |
| `0005_knowledge_checks.py` | Knowledge check records and attempts. |
| `0006_learning_map.py` | Learning phases and objectives. |
| `0007_discussion_reviews.py` | Review windows and answered course question tracking. |
| `0008_course_player.py` | Adaptive course player tables. |
| `0009_course_knowledge_graph.py` | Knowledge graph nodes, edges, and rebuild tracking. |
| `0010_coursebuilder.py` | Course builder course/chapter/lesson/block/quiz/progress tables. |
| `0011_app_runtime_settings.py` | Stored runtime settings for LLM and parser configuration. |

The worker startup also calls schema-ensuring helpers in some services, which is a defensive layer around migrations during development.

## API Schemas

Pydantic schemas live under `platform/backend/schemas/`.

| File | Covers |
| --- | --- |
| `conversation.py` | Conversation create/read/update models and learner-state read shapes. |
| `message.py` | Message roles, output types, source shapes, artifacts, chat request, generation request, and source-file selection. |
| `file.py` | Uploaded file status, read models, upload metadata, retry/delete responses. |
| `coursebuilder.py` | Course builder status, generated course, chapters, lessons, blocks, quizzes, quiz submission, and progress event shapes. |
| `course_player.py` | Older course player course/chapter/lesson/block/quiz/unlock/submit shapes. |
| `knowledge_check.py` | Knowledge check start/submit/read and quiz attempt schemas. |
| `knowledge_graph.py` | Knowledge graph node, edge, graph, rebuild, and remediation path shapes. |
| `review_test.py` | Review test status, start, submit, snooze, dismiss, and action shapes. |
| `runtime_settings.py` | Runtime LLM/parser settings read/update schemas and masked secret handling. |

These schemas are mirrored in `platform/frontend/lib/types.ts`.

## Routers

Routers live under `platform/backend/routers/`.

### `health.py`

Endpoints:

- `/api/health`
- `/api/ready`

Responsibilities:

- Basic liveness.
- Readiness checks for database, MinIO, and Qdrant.

### `conversations.py`

Responsibilities:

- Create conversations.
- List conversations.
- Read one conversation.
- Update title.
- Delete conversation.
- Read messages.
- Read learner state.
- Refresh artifact URLs by signing stored MinIO keys.
- Delete a conversation's Qdrant collection during cleanup.

### `files.py`

Responsibilities:

- Upload multipart course files.
- Store originals in MinIO.
- Create `UploadedFile` rows.
- Enqueue ingestion jobs.
- List and read files.
- Retry failed files.
- Delete files.
- Clean up vectors, stored parsed text, cleaned text, document rows, and course surfaces as needed.
- Rebuild concept inventory, learning map, graph, and course builder state after file deletion when appropriate.

The upload and retry paths can carry `llm_options_json`, allowing parser and LLM settings from the UI to reach ingestion.

### `chat.py`

Endpoint:

- `POST /api/conversations/{conversation_id}/chat`

Responsibilities:

- Validate chat request.
- Resolve runtime settings.
- Set request-scoped LLM/language options.
- Run an interaction router before retrieval for direct actions, off-topic messages, or outside-files cases.
- Retrieve context through the retrieval orchestrator.
- Dispatch to the default chat generator, normally `teacher_gen`.
- Stream POST-SSE events back to the frontend.
- Persist user and assistant messages.
- Persist sources, artifacts, metadata, and learner updates.
- Record answered course questions for review scheduling.

Important helpers:

- `_stream_direct_chat`
- `_stream_chat`
- `_persist_assistant_turn`
- `_load_history`
- `_source_chunk_ids`
- `_looks_like_learning_question`

### `generate.py`

Endpoint:

- `POST /api/conversations/{conversation_id}/generate`

Responsibilities:

- Resolve output type to an enabled generator.
- Synthesize a generator prompt/topic from request options.
- Retrieve context using the generator's preferred retrieval mode.
- Dispatch to the generator.
- Stream tokens, progress, artifacts, sources, done, and errors.
- Persist the generated output as an assistant message.

This powers non-chat outputs such as quizzes, podcasts, and mind maps.

### `generators.py`

Responsibilities:

- List available generator registry entries.
- Optionally include disabled entries.
- Read one generator by ID.

The frontend uses this to render output type buttons and dialogs.

### `coursebuilder.py`

Responsibilities:

- Read generated course builder state.
- Queue generation.
- Rebuild course builder output.
- Stream course builder progress events through SSE.
- Submit chapter quiz attempts.

It enqueues `build_coursebuilder_course` in the worker and reads `CourseBuilderProgressEventRecord` rows for progress.

### `course_player.py`

Responsibilities:

- Read the older/adaptive course player state.
- Rebuild it.
- Unlock chapters.
- Submit chapter quizzes.

If uploaded files are still pending, it returns waiting status rather than attempting to build.

### `knowledge_checks.py`

Responsibilities:

- Start a knowledge check.
- Submit a knowledge check.
- Submit quiz attempts from artifact quizzes.
- Apply resulting learner-state updates.

It sets current language from runtime options before calling assessment services.

### `knowledge_graph.py`

Responsibilities:

- Get the course knowledge graph.
- Rebuild the graph.
- Get remediation paths for a concept.

### `review_tests.py`

Responsibilities:

- Report due review status.
- Start review tests.
- Submit answers.
- Snooze review windows.
- Dismiss review windows.

### `runtime_settings.py`

Responsibilities:

- Read global runtime settings.
- Patch LLM and parser runtime settings.
- Mask saved secrets on reads.
- Delegate encryption and validation to `runtime_settings_service.py`.

## Dispatcher

Dispatcher code lives under `platform/backend/dispatcher/`.

### `registry.py`

Defines:

- `GeneratorEntry`
- `GeneratorRegistry`

Responsibilities:

- Load `generators_registry.json`.
- Validate entries with Pydantic.
- Find a generator by ID.
- Find the enabled generator for an output type.
- Find the default chat generator.
- Filter enabled entries for API responses.

### `router.py`

Defines `GeneratorRouter`.

Responsibilities:

- Route a generator request to the correct adapter.
- Reject disabled generators.
- Reject unsupported adapters.
- Provide a single dispatch interface to chat/generate routers.

### `adapters/api_adapter.py`

Responsibilities:

- POST `GeneratorInput` JSON to a generator `/run` endpoint.
- Parse streamed SSE blocks from the generator.
- Convert generator events into backend `GeneratorEvent` objects.
- Surface dispatch errors.

### `adapters/mcp_adapter.py`

Stub for a future MCP dispatch adapter. The active registry entries use the API adapter.

## Worker

The ARQ worker lives in `platform/backend/workers/ingestion_worker.py`.

Important functions:

- `ingest_file`: full upload ingestion pipeline.
- `build_coursebuilder_course`: background course builder job.
- `startup`: initializes worker dependencies and recovers interrupted jobs.
- `shutdown`: closes shared resources.
- `_set_status`: updates file status and error fields.
- `_all_conversation_files_ready`: checks whether all uploaded files for a conversation are ready.
- `_rebuild_learning_course_if_ready`: triggers concept/map/graph/course rebuilds when all files are ready.
- `_recover_interrupted_uploads`: moves stuck files back to retryable/processing state.
- `_recover_coursebuilder_jobs`: recovers stuck coursebuilder jobs.
- `WorkerSettings`: ARQ worker configuration.

`ingest_file` pipeline:

1. Load `UploadedFile`.
2. Fetch original bytes from MinIO.
3. Resolve parser and LLM settings.
4. Parse through llama-cloud.
5. Store raw parsed markdown.
6. Clean parsed markdown.
7. Store cleaned text.
8. Extract structured course document.
9. Chunk the document.
10. Optionally generate student-style questions for chunks.
11. Replace course content rows in PostgreSQL.
12. Delete old vectors for the file.
13. Embed and upsert new chunks in Qdrant.
14. Mark file ready.
15. If all files are ready, rebuild learning/course surfaces.

## Services

Services live under `platform/backend/services/`. They are grouped here by responsibility.

### Storage and Parsing

| File | Purpose |
| --- | --- |
| `storage_service.py` | MinIO wrapper. Ensures bucket existence, uploads originals and artifacts, computes parsed/cleaned/artifact keys, reads bytes/text, deletes objects, and creates presigned GET URLs. |
| `parsing_service.py` | llama-cloud parser wrapper. Sends uploaded bytes to parser, waits for completion, returns markdown and parser job metadata. |
| `document_cleaning_service.py` | Cleans parser markdown by removing parser noise, repeated boilerplate, footer-like lines, presentation noise, and malformed artifacts. Returns cleaned text and stats. |

### Course Structure and Chunking

| File | Purpose |
| --- | --- |
| `course_structure_service.py` | Extracts structured document metadata, headings, sections, summaries, key concepts, equations, tables, timeline events, and intake metadata from cleaned markdown. |
| `course_intake_normalizer.py` | Normalizes course intake and source metadata used by course builder and course context services. |
| `chunking_service.py` | Converts structured sections into stable search chunks with token budgets, overlaps, heading paths, metadata, and previous/next links. |
| `chunk_question_generator.py` | Uses LLM structured output to attach generated student questions to chunks for better lexical retrieval. |
| `course_content_store.py` | Persists and loads course documents, sections, and chunks. Converts database rows into core `Chunk` objects. |

### Vector and Retrieval

| File | Purpose |
| --- | --- |
| `vector_service.py` | Owns Qdrant collection naming, embedding model loading, passage/query embeddings, collection creation/deletion, payload indexes, chunk upsert, search, scroll, and file-specific vector deletion. |
| `retrieval_orchestrator.py` | Main retrieval brain for chat and generation. Combines dense search, lexical/BM25 signals, reranking, comparison balancing, context expansion, source filtering, formulas, course overview handling, and graph neighbors. |
| `course_context_service.py` | Higher-level context policies for generators and course surfaces. Provides representative chunks, topic contexts, mindmap contexts, equations/tables/timelines, full outline, sections, and generator-specific context packs. |
| `coursebuilder_rag.py` | Course builder retrieval helpers. Loads and filters chunks, retrieves lesson chunks, and builds citations. |
| `coursebuilder_validation.py` | Normalizes block types, validates supported content blocks, checks source support, and validates chart specs. |

### Learning Model

| File | Purpose |
| --- | --- |
| `concept_inventory_service.py` | Builds and loads canonical course concepts from chunks. Uses LLM candidates with deterministic fallback, filters noisy names, merges concepts, and persists source references. |
| `learning_map_service.py` | Builds phases and objectives. Uses LLM or fallback candidates, stable IDs, source references, and compacted objective data. |
| `knowledge_graph_service.py` | Builds and reads concept graph nodes and edges, remediation paths, prerequisite hints, related chunks, and fallback graph structure. |
| `knowledge_assessment_service.py` | Starts checks, grades answers, generates or falls back to check questions, grades quiz attempts, and applies assessment results to learner state. |
| `learner_tracker.py` | Loads and mutates learner state. Applies generator updates, canonical concept updates, and assessment results. Keeps simple and canonical fields in sync. |
| `review_test_service.py` | Tracks answered course questions, schedules review windows, starts review tests, grades submissions, and handles snooze/dismiss actions. |

### Course Surfaces

| File | Purpose |
| --- | --- |
| `course_player_service.py` | Older/adaptive course player. Rebuilds chapter/lesson/block/quiz content, unlocks chapters, grades quizzes, and provides remediation hints. |
| `coursebuilder_service.py` | Main generated course service. Builds source-grounded chapters, lessons, blocks, quizzes, progress events, localized titles, fallback content, validation, persistence, and quiz grading. |
| `coursebuilder_jobs.py` | Job helpers for course builder queueing and state transitions. |

### Interaction and Settings

| File | Purpose |
| --- | --- |
| `interaction_router.py` | Pre-retrieval chat classifier. Handles direct UI-like requests, greetings, off-topic prompts, outside-files cases, and course summary style questions before expensive retrieval/generation. |
| `runtime_settings_service.py` | Stores and resolves runtime LLM/parser settings. Encrypts API keys, masks secrets for reads, sanitizes client options, and combines database settings with environment fallback. |

## Backend Tests

Backend tests live under `platform/backend/tests/`.

They cover:

- Importing app modules without optional services.
- CORS config.
- File schema and retry behavior.
- Document cleaning.
- Chunking.
- Chunk question generation.
- Course structure extraction.
- Course intake normalization.
- Retrieval orchestrator config and comparison terms.
- Course context policy and mindmap context policy.
- Concept inventory.
- Learning map.
- Knowledge graph.
- Knowledge assessment.
- Learner tracker canonical behavior.
- Review test service.
- Course player service.
- Course builder service and router.
- Practical eval catalog.
- Runtime settings service.

The tests are useful as examples of expected service boundaries because many are focused unit tests around deterministic fallback behavior.

## Backend Responsibility Boundary

The backend should:

- Persist everything important.
- Own upload state and ingestion jobs.
- Own retrieval and context selection.
- Own source-file filtering.
- Own learner-state updates.
- Own knowledge graph and course structure rebuilds.
- Dispatch to generators.
- Stream events to the frontend.

The backend should not:

- Encode generator-specific long-form prompts when the generator owns them.
- Depend on frontend stores.
- Invent course facts outside stored chunks.
- Expose disabled generators as usable output types.
