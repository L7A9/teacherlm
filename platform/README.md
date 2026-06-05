# TeacherLM Platform

The platform contains the FastAPI backend, Next.js frontend, background worker, and Docker Compose stack used by TeacherLM.

## Services

`docker-compose.yml` starts:

| Service | Purpose | Port |
| --- | --- | ---: |
| `frontend` | Next.js workspace UI | 3000 |
| `backend` | FastAPI API server | 8000 |
| `arq_worker` | ingestion and course-generation jobs | internal |
| `postgres` | relational data | 5432 |
| `redis` | ARQ queue | 6379 |
| `qdrant` | vector search | 6333 |
| `minio` | files, parsed markdown, artifacts | 9000 / 9001 |
| `teacher_gen` | teacher chat generator | 8001 |
| `quiz_gen` | quiz generator | 8002 |
| `podcast_gen` | podcast generator | 8007 |
| `mindmap_gen` | mind map generator | 8008 |

The backend and generator containers use the host Ollama endpoint by default:

```text
http://host.docker.internal:11434
```

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
./run.sh logs
./run.sh ps
./run.sh stop
./run.sh down
```

The reset script removes local platform data:

```bash
cd platform
./scripts/reset.sh
```

## Environment

Important environment groups:

- Database: `DATABASE_URL`
- Queue: `REDIS_URL`
- Vector store: `QDRANT_URL`, `QDRANT_COLLECTION`
- Object storage: `MINIO_*`
- Parser: `LLAMA_CLOUD_API_KEY`
- Runtime settings encryption: `SETTINGS_ENCRYPTION_KEY`
- Ollama defaults: `OLLAMA_URL`, `OLLAMA_CHAT_MODEL`, `OLLAMA_ANALYSIS_MODEL`, `OLLAMA_EMBED_MODEL`
- Retrieval: `RETRIEVAL_*`
- Course context: `COURSE_CONTEXT_*`
- Ingestion worker limits: `INGESTION_*`

Do not commit real API keys in `.env` or `.env.example`.

Runtime model settings can also be configured through `/api/runtime-settings`. Supported LLM provider types are:

- `ollama`
- `openai`
- `anthropic`
- `openai_compatible`

Provider API keys stored through runtime settings are encrypted with `SETTINGS_ENCRYPTION_KEY`.

## Backend App

The FastAPI app lives in `platform/backend`.

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
- `/api/runtime-settings`

On startup the backend loads the generator registry, ensures the MinIO bucket exists, warms the reranker when enabled, and creates the ARQ Redis pool.

## File Ingestion

Uploaded files are stored in MinIO and ingested by the ARQ worker.

Pipeline:

1. Parse file bytes with `llama-cloud`.
2. Store parsed markdown in MinIO.
3. Clean and normalize the markdown.
4. Extract course documents, sections, concepts, and structure.
5. Chunk sections and annotate search chunks.
6. Store structured records in Postgres.
7. Upsert vectors to Qdrant.
8. Rebuild concepts, learning map, knowledge graph, and CourseBuilder output when all files are ready.

The parser intentionally uses `llama-cloud >= 1.0`; deprecated `llama-parse` and `llama-cloud-services` are not used.

## Chat And Generation

Chat endpoint:

```text
POST /api/conversations/{conversation_id}/chat
```

Generate endpoint:

```text
POST /api/conversations/{conversation_id}/generate
```

Both stream server-sent events and accept `source_file_ids`.

Source-file behavior:

- If one ready file exists, the frontend forces it on.
- If multiple ready files exist, the student selects files from the Sources sidebar.
- Backend routes reject an explicit empty file selection.
- Retrieval filters by selected ready file ids.
- CourseBuilder does not use this filter; it uses all course files.

Common SSE events:

- `token` or `chunk`
- `sources`
- `artifact`
- `progress`
- `done`
- `error`

## Retrieval

Default retrieval values:

| Setting | Default |
| --- | ---: |
| `RETRIEVAL_TOP_K` | 16 |
| `RETRIEVAL_RERANK_TOP_K` | 16 |
| `RETRIEVAL_DENSE_CANDIDATE_K` | 80 |
| `RETRIEVAL_SPARSE_CANDIDATE_K` | 80 |
| `RETRIEVAL_RERANK_CANDIDATE_K` | 50 |

Output type to retrieval mode mapping:

| Output type | Mode |
| --- | --- |
| `text` | `semantic_topk` |
| `quiz` | `coverage_broad` |
| `podcast` | `narrative_arc` |
| `mindmap` | `topic_clusters` |
| `report` | `topic_clusters` |
| `presentation` | `topic_clusters` |
| `chart` / `diagram` | `relationship_dense` |

The backend uses hybrid dense plus sparse retrieval, optional reranking, context expansion, section-aware context, course overview handling, and output-specific context policies.

## CourseBuilder

CourseBuilder is the generated course interface used by the frontend course pane.

It:

- Waits for course files to finish ingestion.
- Uses all course materials in the conversation.
- Extracts or reconstructs chapter and subchapter structure.
- Generates grounded lessons with citations.
- Generates chapter quizzes.
- Records progress events and validation status.

The active frontend displays chapters as open/close accordions. Subchapters are shown directly under the chapter summary rather than inside an extra lesson holder. Opening one subchapter closes the previous subchapter in that chapter.

## Frontend App

The frontend lives in `platform/frontend`.

Main workspace surfaces:

- Sources sidebar: upload files, see ingestion status, select source files.
- Course pane: generated CourseBuilder content.
- Chat pane: teacher chat and generator buttons.
- Generated items sidebar: generated quiz, podcast, mind map, and downloadable artifacts.

On mobile and tablet, a top-right toggle switches between course and chat views. This avoids nested popups when generator dialogs are open.

## Scripts

Backend scripts:

- `scripts/evaluate_retrieval.py`
- `scripts/eval_retrieval.py`
- `scripts/eval_course_context.py`
- `scripts/benchmark_embeddings.py`
- `scripts/reindex_from_parsed.py`

Platform scripts:

- `scripts/init.sh`
- `scripts/reset.sh`

See `backend/evals/README.md` for retrieval and course-context evaluation usage.

## Tests

From the repository root:

```bash
pytest platform/backend/tests
```

The root `pytest.ini` also includes shared-core and generator test directories.
