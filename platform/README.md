# TeacherLM — Platform

AI teacher for students. Upload course files, chat with a teacher grounded in
those files, and generate quizzes, reports, flashcards, diagrams, podcasts, and
presentations from the same corpus.

The platform runs the user-facing app and infrastructure. Individual
generators (teacher, quiz, report, etc.) are separate services registered in
`../generators_registry.json` and brought up one at a time.

## Architecture

```
 ┌───────────┐     ┌─────────────┐    ┌──────────────┐
 │ frontend  │◄───►│  backend    │◄──►│  Postgres    │
 │ Next.js   │ SSE │  FastAPI    │    │  (messages,  │
 │ :3000     │     │  :8000      │    │   files,     │
 └───────────┘     └─────┬───────┘    │   learner)   │
                         │            └──────────────┘
                         │    enqueue
                         ▼
                   ┌───────────┐  ┌──────────┐  ┌──────────┐
                   │  Redis    │  │ Qdrant   │  │ MinIO    │
                   │  (arq)    │  │ vectors  │  │ files +  │
                   │  :6379    │  │ :6333    │  │ artifacts│
                   └───────────┘  └──────────┘  │ :9000    │
                         │                      └──────────┘
                         ▼
                   ┌───────────┐
                   │ arq_worker│     (runs ingestion: parse → chunk → embed)
                   └───────────┘
                         │
                         ▼
                   ┌───────────┐
                   │  Ollama   │   (host-side — NOT in compose)
                   │  :11434   │
                   └───────────┘
                         ▲
                         │
                   ┌───────────┐
                   │ generators│   (teacher_gen, quiz_gen, … — separate
                   │  (each    │    services dispatched by the backend
                   │   their   │    through generators_registry.json)
                   │   own svc)│
                   └───────────┘
```

## Prerequisites

- **Docker Desktop** (or Docker Engine + `docker compose` v2)
- **Ollama**, running locally on the host: <https://ollama.com>
  - The compose stack reaches Ollama via `host.docker.internal`.
- **LlamaCloud API key**: <https://cloud.llamaindex.ai> — required for parsing.
  - This project uses `llama-cloud >= 1.0`. Do **not** install `llama-parse`
    or `llama-cloud-services` (deprecated May 2026).
- Python **3.14+** (only needed if you want to run the backend outside Docker).
- Node **20+** (only needed if you want to run the frontend outside Docker).

## Setup

### 1. Install the shared core package (editable)

This step is only required if you plan to run the backend or generators on
your host. Inside Docker the core package is installed automatically.

```bash
cd packages/teacherlm_core
pip install -e .
```

### 2. Copy and fill in environment variables

```bash
cd platform
cp .env.example .env
```

Then edit `.env`:

- `LLAMA_CLOUD_API_KEY` — required.
- `OLLAMA_HOST` — leave as `http://localhost:11434` for host-side usage; the
  compose stack automatically rewrites it to `host.docker.internal:11434`
  inside the network.
- `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` — change from defaults if deploying
  anywhere beyond your laptop.
- `NEXT_PUBLIC_API_BASE_URL` — point the browser at the backend URL users
  will hit (defaults to `http://localhost:8000`).

### 3. Start the stack

```bash
docker compose up -d --build
```

First build takes several minutes (Python 3.14 slim + fastembed wheels).

### 4. Run the bootstrap script

```bash
./scripts/init.sh
```

This pulls the required Ollama models (`llama3.1:8b`, `nomic-embed-text`),
creates the MinIO bucket, and applies Alembic migrations.

Visit <http://localhost:3000>.

## Bringing up generators

All seven generators ship **disabled** in `../generators_registry.json`. That
way the platform boots before any generator service exists, and the frontend
greys out the buttons for missing types.

Workflow per generator:

1. Build the generator service under `generators/<name>_gen/` (each has its
   own `CLAUDE.md` and I/O contract — see the root `CLAUDE.md`).
2. Run it on a reachable URL (local port, another compose service, etc.).
3. Flip `enabled: true` for that entry in `generators_registry.json` and set
   its `url`. The backend picks the change up on next `/api/generators`
   poll (it reloads the registry on startup and the frontend refetches on a
   60 s stale time).
4. The corresponding button in the chat footer becomes active.

Recommended order: `teacher_gen` first (it's the chat default — the app is
unusable without it), then any of the artifact generators.

## Day-to-day

- **Logs**: `docker compose logs -f backend arq_worker`
- **Shell into backend**: `docker compose exec backend bash`
- **New migration**: `docker compose exec backend alembic revision --autogenerate -m "..."`, commit the file, then `./scripts/init.sh` on peers to apply.
- **MinIO console**: <http://localhost:9001> (creds from `.env`).
- **Qdrant dashboard**: <http://localhost:6333/dashboard>.

## Resetting

```bash
./scripts/reset.sh
```

This stops containers and deletes their volumes. Everything — uploads,
conversations, embeddings, learner state — is gone. Run `docker compose up -d`
and `./scripts/init.sh` again to start fresh.

## Running outside Docker

Each service can be run directly:

```bash
# Backend
cd platform/backend
pip install -r requirements.txt
alembic upgrade head
uvicorn main:app --reload --port 8000

# arq worker (separate shell)
arq workers.ingestion_worker.WorkerSettings

# Frontend
cd platform/frontend
npm install
npm run dev
```

You'll still need Postgres / Redis / Qdrant / MinIO somewhere — the easiest
path is `docker compose up -d postgres redis qdrant minio` while running the
app code on your host.

## Compatibility rules (non-negotiable)

These are enforced by the stack and re-stated here for discoverability:

- **Python 3.14+** (strict)
- **Never** LangChain / LangGraph (Pydantic V1 warnings on 3.14)
- **Never** `llama-parse` or `llama-cloud-services` (deprecated May 2026)
- **Use** `llama-cloud >= 1.0`
- **Use** Pydantic V2 only (`>= 2.12`)
- **Use** Ollama's native `format=` for structured outputs (no LangChain wrappers)
- **Use** FastAPI `>= 0.135`
- **Use** `fastembed` in preference to `sentence-transformers`

See the root `CLAUDE.md` for the full contract.
