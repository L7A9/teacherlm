# TeacherLM Project Parts Index

This directory is a detailed reader's map of the TeacherLM codebase as it exists in this checkout. It is meant for someone who wants to understand the project from the outside in, then move into individual files without losing the overall shape.

The project is an AI teacher workspace. Students upload course material, TeacherLM parses and indexes those files, then lets the student chat with a tutor and generate grounded study outputs such as quizzes, podcasts, mind maps, and course lessons. The strongest invariant across the system is grounding: generated answers should be based only on uploaded course files.

## Documentation Set

| File | Part covered | What it explains |
| --- | --- | --- |
| `00-index.md` | Whole project map | System purpose, data flow, part boundaries, and how to read the docs. |
| `01-root-operations.md` | Repository root and operations | Root files, registry, Docker Compose, scripts, top-level generated artifacts, and operational entry points. |
| `02-shared-core.md` | `packages/teacherlm_core` | Immutable generator schemas, learner state models, LLM wrappers, retrieval primitives, confidence scoring, and shared prompts. |
| `03-platform-backend.md` | `platform/backend` | FastAPI app, database, routers, ingestion worker, dispatcher, services, migrations, and backend tests. |
| `04-data-retrieval-learning.md` | Data, RAG, and learning model | Upload-to-retrieval pipeline, chunk model, course structure, concept inventory, learning map, knowledge graph, review tests, and learner progress. |
| `05-platform-frontend.md` | `platform/frontend` | Next.js app, routes, API client, streaming, stores, hooks, workspace layout, chat, artifacts, course builder, and progress UI. |
| `06-generator-services.md` | `generators/*` | Teacher, quiz, podcast, and mindmap generators; shared generator contract; disabled registry entries. |
| `07-tests-evaluations-reports.md` | Tests, evals, reports, artifacts | Test coverage by subsystem, retrieval evaluations, report chart assets, generated decks, and top-level report files. |

## Project Shape

```text
teacherlm/
|-- AGENTS.md
|-- README.md
|-- generators_registry.json
|-- run.sh
|-- packages/
|   `-- teacherlm_core/
|-- platform/
|   |-- backend/
|   |-- frontend/
|   |-- docker-compose.yml
|   `-- scripts/
|-- generators/
|   |-- teacher_gen/
|   |-- quiz_gen/
|   |-- podcast_gen/
|   `-- mindmap_gen/
|-- artifacts/
|-- .claude/
`-- .vscode/
```

The intended top-level structure described in `AGENTS.md` also names `report_gen`, `presentation_gen`, and `chart_gen`, but this checkout only contains the four enabled generator directories above. The registry still has disabled entries for report, presentation, and chart outputs.

## Main Runtime Services

| Service | Location | Purpose |
| --- | --- | --- |
| Frontend | `platform/frontend` | Next.js study workspace used by the student. |
| Backend API | `platform/backend` | FastAPI API for conversations, uploads, retrieval, generation, learning state, and course surfaces. |
| ARQ worker | `platform/backend/workers/ingestion_worker.py` | Background ingestion and course-building jobs. |
| Shared core | `packages/teacherlm_core` | Reusable schemas, retrieval helpers, LLM clients, confidence scoring, and prompts. |
| Teacher generator | `generators/teacher_gen` | Chat tutor that answers, guides, affirms, or quizzes back. |
| Quiz generator | `generators/quiz_gen` | Grounded quiz artifact generator. |
| Podcast generator | `generators/podcast_gen` | Grounded audio/transcript study podcast generator. |
| Mindmap generator | `generators/mindmap_gen` | Grounded visual study map generator. |
| PostgreSQL | Compose service | Persistent relational state. |
| Redis | Compose service | ARQ job queue. |
| Qdrant | Compose service | Vector database for chunk retrieval. |
| MinIO | Compose service | Object storage for uploaded files, parsed text, cleaned text, and artifacts. |
| Ollama or cloud LLMs | External/runtime setting | LLM provider used by backend and generators. |
| llama-cloud | External API | Parser used to convert uploaded documents into markdown. |

## End-to-End Flow

1. The student creates or opens a conversation in the frontend.
2. The student uploads one or more course files.
3. The backend stores the original file in MinIO and records an `UploadedFile` row.
4. The backend enqueues `ingest_file` in Redis for the ARQ worker.
5. The worker parses the file through llama-cloud.
6. The worker stores raw parsed markdown and cleaned markdown in MinIO.
7. The worker extracts a structured course document: document metadata, sections, headings, formulas, tables, key concepts, and timeline events.
8. The worker chunks the structured text and stores chunks in PostgreSQL.
9. The worker embeds chunks with fastembed and upserts them into Qdrant.
10. After all files in a conversation are ready, backend jobs rebuild concept inventory, learning map, knowledge graph, and the generated course surface.
11. Chat and generation requests retrieve relevant context chunks through the backend retrieval layer.
12. The backend dispatches requests to a generator service over HTTP streaming.
13. The generator streams tokens, sources, artifacts, and final metadata back to the backend.
14. The backend persists messages, artifacts, sources, learner updates, and metadata.
15. The frontend consumes POST-SSE events and updates the workspace live.

## Grounding Contract

Every generator receives the same canonical input:

```json
{
  "conversation_id": "string",
  "user_message": "string",
  "context_chunks": [
    {
      "text": "string",
      "source": "string",
      "score": 0.0,
      "chunk_id": "string"
    }
  ],
  "learner_state": {
    "understood_concepts": ["string"],
    "struggling_concepts": ["string"],
    "mastery_scores": {
      "concept": 0.0
    },
    "session_turns": 0
  },
  "chat_history": [
    {
      "role": "user",
      "content": "string"
    }
  ],
  "options": {}
}
```

Every generator returns the same canonical output:

```json
{
  "response": "markdown string",
  "generator_id": "string",
  "output_type": "text",
  "artifacts": [
    {
      "type": "quiz",
      "url": "string",
      "filename": "string"
    }
  ],
  "sources": [
    {
      "text": "string",
      "source": "string",
      "score": 0.0
    }
  ],
  "learner_updates": {
    "concepts_covered": ["string"],
    "concepts_demonstrated": ["string"],
    "concepts_struggled": ["string"]
  },
  "metadata": {}
}
```

The backend owns retrieval, persistence, source filtering, learner state, and dispatch. Generators own the specific output type and should not invent facts outside the provided context chunks.

## Retrieval Modes

| Mode | Intended use | Meaning in the system |
| --- | --- | --- |
| `semantic_topk` | Chat tutor | Finds the top closest chunks to the student's question. |
| `coverage_broad` | Quizzes | Samples broadly across the course, with diversity rather than only nearest neighbors. |
| `narrative_arc` | Podcasts and reports | Selects introduction, key development, and conclusion-style material. |
| `topic_clusters` | Mind maps and presentations | Groups chunks by topic so outputs cover the course structure. |
| `relationship_dense` | Charts and graph-like outputs | Prefers chunks with many entities, relations, formulas, comparisons, or dense facts. |

The backend can request these modes from the shared core and from its own richer context policy layer. The registry associates each generator with the retrieval mode it usually needs.

## Compatibility Rules That Shape The Code

The root `AGENTS.md` makes several technical constraints non-negotiable:

- Python is 3.14 or newer.
- Pydantic must be V2, with version 2.12 or newer.
- FastAPI must be 0.135 or newer.
- The project must not use LangChain or LangGraph.
- The project must not use deprecated `llama-parse` or `llama-cloud-services`.
- Parsing should use `llama-cloud` version 1.0 or newer.
- Structured local Ollama output should use the native `format=` argument.
- Embeddings should use fastembed where possible.
- User-facing text should say "generators" or "output types"; "agent" is internal terminology.

## Current Implementation Boundary

Implemented and enabled:

- Chat tutor: `teacher_gen`
- Quiz generator: `quiz_gen`
- Podcast generator: `podcast_gen`
- Mind map generator: `mindmap_gen`
- Backend course builder and course player surfaces
- Learner state, knowledge checks, review tests, concepts, learning maps, and knowledge graphs

Registered but disabled:

- Report generator
- Presentation generator
- Chart generator

Present as generated or support assets:

- Top-level LaTeX reports and presentations
- `artifacts/` presentation exports
- Backend retrieval evaluation JSON, CSV, chart, notebook, and report assets

## How To Use These Docs

Start with this file, then read `01-root-operations.md` to understand how the repo starts. After that:

- Read `02-shared-core.md` before reading any generator.
- Read `03-platform-backend.md` before debugging API behavior.
- Read `04-data-retrieval-learning.md` before changing ingestion, retrieval, learner state, or course generation.
- Read `05-platform-frontend.md` before changing the workspace UI.
- Read `06-generator-services.md` before changing output generation.
- Read `07-tests-evaluations-reports.md` before changing test coverage, evals, or generated report artifacts.
