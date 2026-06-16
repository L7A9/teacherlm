# Root And Operations

This file documents the repository root, operational entry points, Docker Compose stack, registry, generated root artifacts, and support directories.

## Root Files

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Project instructions for coding agents and maintainers. It defines the product, compatibility rules, naming rules, shared core package expectations, generator I/O contract, retrieval modes, and subdirectory reading rules. |
| `README.md` | Main human-facing project README. It explains the implemented architecture, active generators, stack, API surfaces, RAG pipeline, and development commands. |
| `generators_registry.json` | Runtime registry that tells the backend which generator services exist, which output type they serve, whether they are enabled, their endpoints, and their preferred retrieval modes. |
| `pytest.ini` | Root pytest configuration. It currently points pytest at core tests, teacher generator tests, mindmap generator tests, and backend tests. |
| `run.sh` | Convenience shell wrapper around Docker Compose. It starts, stops, rebuilds, logs, shells into services, and can generate retrieval/report charts. |
| `.dockerignore` | Build context exclusion rules for Docker. |
| `.gitignore` | Git ignore rules. |
| `CLAUDE.md` | Local assistant guidance file. It overlaps with project instructions but is not the root contract for this documentation. |
| `.vscode/settings.json` | Workspace editor settings. |
| `.claude/settings.local.json` | Local Claude tooling settings. |
| `.claude/worktrees/` | Local worktree metadata used by assistant tooling. |
| `artifacts/` | Generated presentation artifacts and deck comparison images. These are not runtime source files. |

## Top-Level Generated Report Files

The root contains several `.tex` files that appear to be generated or manually authored report and presentation deliverables:

- `final_report.tex`
- `presentation.tex`
- `presentation5.tex`
- `presentation_4.tex`
- `repport.tex`
- `teacherlm_report.tex`
- `teacherlm_report_merge.tex`
- `teacherlm_report_part1.tex`
- `teacherlm_report_part2.tex`
- `teacherlm_report_plan.tex`

These files are documentation/report artifacts rather than the running application. They should be treated separately from the service code.

## `generators_registry.json`

The registry is loaded by `platform/backend/dispatcher/registry.py`. It is the backend's source of truth for generator routing.

Implemented and enabled entries:

| Generator ID | Output type | Endpoint | Retrieval mode | Role |
| --- | --- | --- | --- | --- |
| `teacher_gen` | `text` | `http://teacher_gen:8001/run` | `semantic_topk` | Default chat tutor. |
| `quiz_gen` | `quiz` | `http://quiz_gen:8002/run` | `coverage_broad` | Quiz artifact generator. |
| `podcast_gen` | `podcast` | `http://podcast_gen:8007/run` | `narrative_arc` | Audio and transcript study podcast generator. |
| `mindmap_gen` | `mindmap` | `http://mindmap_gen:8008/run` | `topic_clusters` | Visual study map generator. |

Registered but disabled entries:

| Generator ID | Output type | Retrieval mode | Status in this checkout |
| --- | --- | --- | --- |
| `report_gen` | `report` | `narrative_arc` | Disabled in registry and no `generators/report_gen` directory is present. |
| `presentation_gen` | `presentation` | `topic_clusters` | Disabled in registry and no `generators/presentation_gen` directory is present. |
| `chart_gen` | `chart` | `relationship_dense` | Disabled in registry and no `generators/chart_gen` directory is present. |

Important registry fields:

- `id`: stable generator identifier.
- `name`: display name used by API/frontend.
- `output_type`: canonical output type routed by `/api/conversations/{id}/generate`.
- `description`: user-facing capability description.
- `enabled`: whether the backend should expose and dispatch this generator.
- `endpoint`: HTTP run URL for `adapter: "api"`.
- `adapter`: dispatch implementation, currently API for implemented services.
- `retrieval_mode`: backend retrieval mode to use before dispatch.
- `icon`: display icon metadata for frontend use.

## Docker Compose Stack

`platform/docker-compose.yml` defines the local stack.

| Service | Image/build | Main role |
| --- | --- | --- |
| `postgres` | `postgres:16-alpine` | Relational database for conversations, messages, files, course structure, learner state, and generated course data. |
| `redis` | `redis:7-alpine` | Queue backend for ARQ jobs. |
| `qdrant` | `qdrant/qdrant` | Vector database for embedded search chunks. |
| `minio` | `minio/minio` | Object storage for originals, parsed markdown, cleaned text, and generated artifacts. |
| `backend` | `platform/backend/Dockerfile` | FastAPI API server on port 8000. |
| `arq_worker` | Same backend image | Runs `workers.ingestion_worker.WorkerSettings`. |
| `teacher_gen` | `generators/teacher_gen/Dockerfile` | Chat generator on port 8001. |
| `quiz_gen` | `generators/quiz_gen/Dockerfile` | Quiz generator on port 8002. |
| `mindmap_gen` | `generators/mindmap_gen/Dockerfile` | Mindmap generator on port 8008. |
| `podcast_gen` | `generators/podcast_gen/Dockerfile` | Podcast generator on port 8007. |
| `frontend` | `platform/frontend/Dockerfile` | Next.js frontend on port 3000. |

### Compose Volumes

| Volume | Used by | Stores |
| --- | --- | --- |
| `postgres_data` | `postgres` | Database files. |
| `qdrant_data` | `qdrant` | Vector collections and payload indexes. |
| `minio_data` | `minio` | Object storage bucket data. |
| `mindmap_artifacts` | `mindmap_gen` | Mindmap static artifact files. |
| `podcast_artifacts` | `podcast_gen` | Podcast artifact output directory. |

Podcast also bind-mounts a host models directory into `/app/models` so Piper/Kokoro assets can persist outside the container image.

### Compose Networking

Services talk over the default Compose network by service name:

- Backend to Postgres: `postgres:5432`
- Backend and worker to Redis: `redis:6379`
- Backend to Qdrant: `qdrant:6333`
- Backend and generators to MinIO: `minio:9000`
- Backend to generators: `teacher_gen:8001`, `quiz_gen:8002`, `podcast_gen:8007`, `mindmap_gen:8008`
- Containers to host Ollama by default: `host.docker.internal:11434`

The frontend receives `NEXT_PUBLIC_API_BASE_URL` as a build argument and environment value.

## `run.sh`

`run.sh` is a convenience script. It wraps `docker compose` and exports local network settings before starting the stack.

Commands:

| Command | Behavior |
| --- | --- |
| `./run.sh up` | Starts the stack. |
| `./run.sh build` | Builds images. |
| `./run.sh rebuild` | Rebuilds images and starts services. |
| `./run.sh stop` | Stops services without removing volumes. |
| `./run.sh down` | Stops services and removes Compose resources according to Compose defaults. |
| `./run.sh logs [service]` | Follows logs for all services or one service. |
| `./run.sh ps` | Shows Compose service status. |
| `./run.sh shell [service]` | Opens a shell in a service container. Defaults to backend. |
| `./run.sh report-charts` | Runs retrieval comparisons and chart notebook generation inside the backend container, then copies outputs back. |

Details:

- It detects a LAN host address so the frontend and generated artifact URLs can be reachable from other devices on the same network.
- It sets `NEXT_PUBLIC_API_BASE_URL`.
- It sets `MINIO_PUBLIC_ENDPOINT`.
- It sets `MINDMAP_GEN_PUBLIC_URL`.
- `report-charts` starts backend dependencies, copies eval inputs into the container, runs comparison scripts, and copies result artifacts back to `platform/backend/evals/`.

## Platform Scripts

### `platform/scripts/init.sh`

This script prepares a local stack after Compose is up.

It does three main things:

1. Loads `.env` if present.
2. Pulls Ollama models:
   - `llama3.1:8b`
   - `nomic-embed-text`
3. Creates the MinIO bucket with `minio/mc`.
4. Runs Alembic migrations inside the backend container.

The Alembic configuration file is `platform/backend/alembic.ini`; migration scripts live under `platform/backend/db/migrations/`.

### `platform/scripts/reset.sh`

This script resets local Docker state.

Behavior:

- Prompts for confirmation unless called with `--yes`.
- Runs Docker Compose cleanup.
- Removes named volumes.
- Is destructive to local database, vector database, object storage, and artifact volumes.

## Backend Docker Image

`platform/backend/Dockerfile`:

- Uses Python 3.14 slim.
- Installs build tools, curl, PostgreSQL libraries, Rust, and Cargo.
- Installs `packages/teacherlm_core` as a local editable dependency.
- Installs backend requirements.
- Can prefetch embedding and reranker models through build arguments.
- Copies backend code and the root registry into the image.
- Runs Uvicorn on port 8000 for the API image.

The ARQ worker reuses this image with a different command.

## Frontend Docker Image

`platform/frontend/Dockerfile`:

- Uses Node 20 Alpine.
- Installs dependencies with `npm install`.
- Builds a Next.js standalone output.
- Copies standalone server files and static assets.
- Runs as a non-root `nextjs` user.
- Starts the Next.js server on port 3000.

The comments mention a Windows lockfile optional dependency issue, which is why the image uses `npm install` rather than relying only on `npm ci`.

## Generator Docker Images

| Generator | Dockerfile behavior |
| --- | --- |
| `teacher_gen` | Python 3.14 slim, installs core and generator requirements, optionally pre-downloads a reranker model, exposes port 8001. |
| `quiz_gen` | Python 3.14 slim, installs fastembed and MinIO dependencies, pre-downloads a BAAI embedding model for distractor support, exposes port 8002. |
| `podcast_gen` | Python 3.14 slim, installs audio system libraries, ffmpeg, espeak, TTS-related Python packages, NLTK punkt data, creates model/artifact dirs, exposes port 8007. |
| `mindmap_gen` | Python 3.14 slim, installs light dependencies, installs `teacherlm_core` without heavy dependencies, creates static artifact directory, exposes port 8008. |

## `artifacts/`

The root `artifacts/` directory contains generated presentation outputs:

- `.json` files with presentation data.
- `.html` files with rendered presentation previews.
- `.md` files for presentation markdown.
- `.pptx` files for PowerPoint exports.
- `artifacts/deck_compare/` images and a contact sheet for deck visual comparison.

These assets are useful for demonstrations and reports, but they are not part of the currently enabled runtime generator services.

## Operational Mental Model

The local stack works best if you think of it in three layers:

1. Infrastructure:
   - Postgres
   - Redis
   - Qdrant
   - MinIO
2. Platform:
   - Backend API
   - ARQ worker
   - Frontend
3. Generators:
   - Teacher
   - Quiz
   - Podcast
   - Mindmap

The backend is the center. It owns state, retrieval, and dispatch. Generators are stateless-ish services from the backend's point of view: they receive context and options, stream output, and return learner updates.
