# Local TeacherLM Desktop

This folder is the handoff point for building TeacherLM as a local installable
desktop application for students.

The goal is not to make a thin wrapper around the current Docker platform. The
goal is to rebuild the same product as a local-first desktop app where student
files, indexes, learner state, conversations, artifacts, and settings live on
the student's computer.

Prepared from the root project README files and the current product direction:

- no Electron,
- use Tauri for the desktop app,
- keep TeacherLM's generator architecture,
- keep the current RAG quality techniques,
- add HyDE as a first-class local retrieval feature,
- make generators standalone and connectable through MCP-style adapters,
- let external agents, such as a presentation agent, plug into TeacherLM and use
  its local course memory safely.

## Product Goal

Local TeacherLM is a private AI teacher for students.

A student installs the desktop app, uploads course files, and gets:

- grounded teacher chat,
- source citations,
- quizzes,
- mind maps,
- podcasts,
- generated course lessons,
- review tests,
- learner-progress tracking,
- and future output types such as reports, presentations, charts, and diagrams.

All answers and generated artifacts must be grounded in uploaded course files
unless the UI clearly says a feature is using external knowledge.

## Non-Negotiable Rules

These rules carry over from the root project and are strengthened for desktop:

- Do not use Electron.
- Use Tauri for the installable desktop interface.
- Store student data locally by default.
- Do not require Docker.
- Do not require Postgres, Redis, MinIO, or a hosted Qdrant service.
- Do not use LangChain.
- Do not use LangGraph.
- Do not use `llama-parse`.
- Do not use `llama-cloud-services`.
- Use Pydantic V2 only.
- Use Python 3.14+ for Python services.
- Use FastAPI >= 0.135 if a local Python HTTP sidecar is used.
- Use `llama-cloud >= 1.0` only as an optional external parser path.
- Prefer `fastembed` for embeddings and reranking.
- Use the Ollama Python library native `format=` argument for structured local
  output.
- Keep "agent" terminology internal. User-facing language should say
  "generators", "output types", or "connected tools".

## What Must Be Copied From The Current Project

The local app should preserve the current TeacherLM behavior, not replace it
with a simpler chatbot.

Must preserve:

- teacher chat with warm teaching voice,
- grounded answers from uploaded files only,
- source citations with chunk text, source label, chunk ID, and score,
- source-file selection for chat and generators,
- generated course/CourseBuilder behavior,
- learner state and mastery tracking,
- knowledge checks,
- review tests,
- knowledge graph and remediation behavior,
- quiz generation,
- mind map generation,
- podcast transcript and audio generation,
- runtime model settings,
- forced output language,
- artifact rendering and history,
- retrieval evaluations.

Must preserve retrieval techniques:

- structured section-based chunking,
- dense semantic retrieval,
- BM25 keyword retrieval,
- hybrid retrieval,
- reciprocal rank fusion, RRF,
- fastembed cross-encoder reranking,
- graph search,
- graph-neighbor context expansion,
- local neighbor expansion,
- section-summary expansion,
- source-file filtering,
- generated hypothetical questions per chunk,
- formula/table/timeline extraction where available,
- output-specific retrieval modes.

Must add or explicitly implement in the local rewrite:

- HyDE query expansion as a retrieval-only technique.

Current repo note: the existing platform has generated questions per chunk, but
HyDE is not wired as a retrieval feature. The local rewrite should add HyDE
intentionally and keep it separate from chunk-level hypothetical questions.

## Recommended Local Architecture

Use Tauri as the desktop shell and system WebView host. Do not use Electron.

Recommended runtime shape:

```text
local_teacherlm/
  README.md
  apps/
    desktop/                  # Tauri + React/TypeScript UI
  contracts/
    generator.schema.json      # generated from Pydantic models
    mcp_tools.schema.json      # MCP-facing tool contracts
  python/
    local_api/                 # local FastAPI sidecar, 127.0.0.1 only
    teacherlm_core/            # copied/adapted shared package
    generators/
      teacher_gen/
      quiz_gen/
      mindmap_gen/
      podcast_gen/
      presentation_gen/        # optional future built-in or external agent
  rust/
    tauri_shell/               # Tauri commands, local process supervisor
  mcp/
    server/                    # TeacherLM MCP server exposed to agents
    clients/                   # adapters for external MCP generators
  data/
    README.md                  # describes local app-data layout, not user data
```

The exact folders can change during implementation, but the boundaries should
stay stable.

### Runtime Layers

| Layer | Technology | Responsibility |
| --- | --- | --- |
| Desktop shell | Tauri, Rust | Installable app, windows, file dialogs, app-data paths, local sidecar supervision, permissions |
| UI | React, TypeScript, Tailwind/Radix-style components | Workspace, uploads, chat, generators, artifacts, settings, progress |
| Local API sidecar | Python 3.14+, FastAPI, Pydantic V2 | Conversations, ingestion, retrieval, learner state, generator routing, MCP bridge |
| Shared core | Python package | Generator schemas, LLM wrappers, prompts, retrieval primitives, confidence utilities |
| Built-in generators | Python FastAPI services or in-process adapters | Teacher, quiz, mind map, podcast, future built-ins |
| Agent/MCP hub | Python plus optional Rust supervision | Register external agents, expose TeacherLM context tools, call connected generators |
| Local storage | SQLite, local filesystem, local vector index | Everything student-owned and device-local |

### Why This Shape

Tauri gives an installable desktop app without bundling Chromium like Electron.
The tradeoff is that Tauri is still a WebView abstraction, so the app should keep
native responsibilities in the Tauri/Rust shell and avoid browser-like desktop
behavior.

The current product is heavily Python-based. Rewriting all ingestion, retrieval,
LLM, TTS, and generator logic in Rust would delay parity. The fastest path is:

- keep the high-value AI/retrieval/generator logic in Python,
- bundle it as a local sidecar for production,
- let Tauri own installation, windows, permissions, and process lifecycle,
- let React/TypeScript own the student workspace UI.

This preserves the current architecture while making it local and installable.

## Local Data Storage

Everything must live on the student's machine.

Use the OS app-data directory, not the project folder, for runtime data.

Example Windows path:

```text
%APPDATA%/TeacherLM/
```

Example app-data layout:

```text
TeacherLM/
  teacherlm.db
  objects/
    uploads/
    parsed/
    cleaned/
  artifacts/
    quizzes/
    mindmaps/
    podcasts/
    presentations/
    reports/
    charts/
  indexes/
    vector/
    bm25/
    graph/
  models/
    embeddings/
    rerankers/
    tts/
  logs/
  traces/
  plugins/
    generators/
    mcp/
```

### Replacements For Current Server Infrastructure

| Current platform | Local desktop replacement |
| --- | --- |
| Postgres | SQLite database with migrations |
| Redis + ARQ | SQLite job table plus local async worker |
| MinIO | Local filesystem artifact/object store |
| Qdrant Docker service | Embedded/local vector store or bundled local Qdrant sidecar |
| Docker Compose service network | Tauri-supervised local sidecars on localhost or stdio |
| Backend environment variables | Settings stored locally, secrets encrypted |
| Browser frontend | Tauri WebView frontend |

### SQLite Tables To Preserve

The local DB should keep equivalents of:

- conversations,
- messages,
- uploaded files,
- course documents,
- course sections,
- search chunks,
- chunk metadata,
- generated chunk questions,
- formulas,
- tables,
- timeline events,
- concept inventory,
- learning phases,
- learning objectives,
- knowledge graph nodes,
- knowledge graph edges,
- learner state,
- review windows,
- knowledge checks,
- coursebuilder course records,
- artifacts,
- generator registry,
- connected external agents,
- generator run traces,
- background jobs.

## Local File Ingestion

The desktop app should support local parsing first.

Recommended parser stack:

| File type | Local parser path |
| --- | --- |
| PDF | PyMuPDF or pypdf, with OCR fallback later |
| DOCX | python-docx or mammoth |
| PPTX | python-pptx |
| TXT/MD | direct text read |
| HTML | BeautifulSoup/readability-style extraction |
| Images/scans | optional OCR path, not first milestone |

Optional cloud parser:

- `llama-cloud >= 1.0` can remain available if the student explicitly configures
  it.
- The UI must clearly mark it as external/cloud.
- No deprecated LlamaParse packages should be used.

Ingestion pipeline:

1. Student selects files through a Tauri file dialog.
2. Tauri gives the local API a file path or copies the file into the app object
   store.
3. The local API creates an `uploaded_files` record.
4. A local background worker parses the file.
5. Raw parsed text/markdown is stored in `objects/parsed`.
6. Cleaned text is stored in `objects/cleaned`.
7. Course structure is extracted into documents and sections.
8. Deterministic section-based chunks are created.
9. Hypothetical student questions are generated per chunk.
10. Chunks are embedded with fastembed.
11. Vector index, BM25 index, and graph records are updated.
12. Concept inventory, learning map, knowledge graph, and generated course are
    rebuilt.
13. The file is marked ready.

## Retrieval Architecture

Retrieval remains platform-owned. Generators do not search the vector store
directly.

The local orchestrator should expose one retrieval service:

```text
retrieve_for(
  conversation_id,
  user_message,
  output_type,
  source_file_ids,
  options
) -> list[Chunk]
```

### Chunk Schema

Every retrieved evidence unit should keep the existing shape:

```json
{
  "text": "course excerpt or expanded context",
  "source": "lecture_03.pdf",
  "score": 1.24,
  "chunk_id": "stable-local-id",
  "metadata": {
    "document_id": "local-document-id",
    "section_id": "local-section-id",
    "source_file_id": "local-file-id",
    "heading_path": ["Module 1", "Topic 2"],
    "key_concepts": [],
    "generated_questions": [],
    "context_type": "focused_chunk"
  }
}
```

Citation scores are relevance scores for that answer's retrieval run. Higher is
better within the same answer, but they are not percentages and not factuality
guarantees.

### Dense Semantic Search

Use fastembed to embed:

- original chunks,
- optional chunk summaries,
- optional generated chunk questions,
- query text,
- HyDE query text.

Store vectors locally. Candidate vector store options:

- qdrant-client local mode if it gives enough filtering support,
- bundled local Qdrant binary bound to localhost with its data path in app data,
- LanceDB or another embedded vector DB if it gives better desktop packaging,
- sqlite-vec as a possible later simplification.

The first implementation should prefer the option that minimizes changes to the
existing retrieval code while keeping all data local.

### BM25 Keyword Search

Keep lexical retrieval because exact course terms matter.

Search over:

- chunk text,
- section title,
- heading path,
- key concepts,
- formula labels,
- table captions,
- generated hypothetical questions.

Implementation options:

- keep current `rank-bm25` in Python for parity,
- add SQLite FTS5 as a persistent lexical index for faster local startup,
- keep both if useful: FTS5 for candidate IDs, `rank-bm25` for exact parity and
  eval comparison.

### Generated Hypothetical Questions Per Chunk

This is an ingestion-time technique.

For each chunk, generate several student-style questions that the chunk can
answer. Store them in chunk metadata and lexical indexes.

Why:

- students rarely use the same wording as the source,
- BM25 can match the generated question even when the original paragraph does
  not contain the student's exact phrasing,
- it improves exact-term and practical-question recall.

These questions are retrieval hints only. They are not shown as course content
unless a debugging/eval view explicitly asks for them.

### HyDE Query Expansion

HyDE is a query-time technique and must be added to the local rewrite.

Flow:

1. Student asks a question.
2. The retrieval service asks the configured local LLM to write a short
   hypothetical answer/document that would answer the question.
3. The HyDE text is embedded.
4. Dense search runs for both the original query and the HyDE text.
5. BM25 still runs mainly on the original query and optional query expansions.
6. Dense, BM25, graph, and other candidates are fused with RRF.
7. HyDE text is discarded after tracing.

Rules:

- HyDE output is never cited.
- HyDE output is never shown to the student by default.
- HyDE must not become a source of facts.
- HyDE only helps find uploaded chunks.
- Store HyDE traces locally for eval/debugging if tracing is enabled.
- Disable HyDE or reduce it for formula-only queries where exact symbols matter.

Suggested setting:

```text
retrieval_hyde_enabled = true
retrieval_hyde_max_chars = 900
retrieval_hyde_model = local structured/chat model
```

### Reciprocal Rank Fusion

Use RRF to combine rankings whose raw scores are not comparable.

Current core uses:

```text
score(d) = sum(1 / (RRF_K + rank_i(d)))
RRF_K = 60
```

Fuse at least:

- dense original-query hits,
- dense HyDE hits,
- BM25 hits,
- generated-question hits if tracked separately,
- graph candidates when represented as a ranked list.

### Reranking

After fusion, rerank candidate chunks with fastembed cross-encoder reranking.

Default model from current project:

```text
BAAI/bge-reranker-base
```

Reranking should happen after the candidate set is narrowed, not across the full
corpus.

### Graph Search

Graph search must remain a first-class retrieval path.

Build a local knowledge graph from course content:

- concept nodes,
- formula nodes,
- objective nodes,
- example nodes,
- prerequisite edges,
- related-to edges,
- demonstrates edges,
- appears-in-chunk links.

At query time:

1. Extract important query terms and concepts.
2. Match them to graph nodes.
3. Traverse related nodes and edges.
4. Pull linked chunks.
5. Add graph candidates before reranking.
6. Add graph-neighbor expansion to final chunks when helpful.

Graph chunks should carry metadata such as:

```json
{
  "retrieval_via": "knowledge_graph",
  "context_type": "knowledge_graph_neighbor"
}
```

### Context Expansion

Final chunks can be expanded with:

- previous and next chunk,
- parent section summary,
- graph neighbors,
- equations,
- tables,
- course outline,
- module packs.

Do not expand everything for every request. Use the output-type policy.

### Output-Type Retrieval Modes

Keep the current modes:

| Output type | Retrieval mode | Purpose |
| --- | --- | --- |
| `text` | `semantic_topk` | Focused teacher answers |
| `quiz` | `coverage_broad` | Representative assessment coverage |
| `podcast` | `narrative_arc` | Intro, middle, conclusion teaching flow |
| `mindmap` | `topic_clusters` | Course/module/topic overview |
| `report` | `topic_clusters` | Reserved or future built-in |
| `presentation` | `topic_clusters` | Reserved for external/built-in presentation agent |
| `chart` / `diagram` | `relationship_dense` | Relationship-heavy facts and processes |

## Generator Architecture

Generators must stay standalone.

The local desktop app is the orchestrator. It owns retrieval, learner state,
source filtering, persistence, permissions, and artifact storage. Generators
own one output capability.

### Immutable Generator Input

Keep the existing contract:

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
  "learner_state": {
    "understood_concepts": [],
    "struggling_concepts": [],
    "mastery_scores": {},
    "session_turns": 0
  },
  "chat_history": [
    {
      "role": "user",
      "content": "message"
    }
  ],
  "options": {}
}
```

### Immutable Generator Output

Keep the existing output:

```json
{
  "response": "markdown",
  "generator_id": "teacher_gen",
  "output_type": "text",
  "artifacts": [
    {
      "type": "quiz",
      "url": "teacherlm-local://artifact/id",
      "filename": "quiz.json",
      "key": "local-artifact-key"
    }
  ],
  "sources": [],
  "learner_updates": {
    "concepts_covered": [],
    "concepts_demonstrated": [],
    "concepts_struggled": []
  },
  "metadata": {}
}
```

For local artifacts, prefer `teacherlm-local://artifact/<id>` or local API URLs
over raw filesystem paths. The UI should ask the local API/Tauri shell to open
or export files.

### Generator Manifest

Each built-in or external generator should register a manifest:

```json
{
  "generator_id": "presentation_gen",
  "display_name": "Presentation",
  "contract_version": "1.0.0",
  "output_type": "presentation",
  "enabled": true,
  "transport": "mcp",
  "endpoint": "stdio:command-or-server-id",
  "retrieval_mode": "topic_clusters",
  "artifact_types": ["pptx", "pdf"],
  "permissions": {
    "read_context": true,
    "read_learner_state": true,
    "write_artifacts": true,
    "network": false
  },
  "options_schema": {},
  "capabilities": [
    "streaming",
    "grounded_generation",
    "artifact_output"
  ]
}
```

Supported transports should include:

- `local_inprocess` for built-ins running in the same Python sidecar,
- `local_http` for built-ins running as local localhost services,
- `local_process` for subprocess generators over stdio,
- `mcp_stdio` for MCP servers started by TeacherLM,
- `mcp_http` or equivalent for user-configured MCP servers when supported.

## Built-In Generators To Preserve

### Teacher Generator

Purpose:

- default chat,
- explanations,
- guidance,
- quiz-back prompts,
- affirmations,
- refusals when evidence is weak,
- confidence metadata,
- learner update extraction.

Keep:

- `semantic_topk`,
- `analysis` event,
- `sources` event,
- mode selection: `explain`, `guide`, `quiz_back`, `affirm`,
- formula fast path,
- course overview fast path,
- confidence score,
- teacher voice prompt,
- learner update extraction.

### Quiz Generator

Purpose:

- grounded quizzes from selected course files.

Keep:

- `coverage_broad`,
- MCQ and true/false,
- internal room for fill-blank later,
- Bloom levels: remember, understand, apply, analyze,
- learner-aware planning,
- struggling/coverage/stretch mix,
- per-slot structured question generation,
- optional fastembed distractor enhancement,
- strict validation,
- deterministic top-up fallback,
- local quiz JSON artifact.

### Podcast Generator

Purpose:

- educational two-host podcast from course context.

Keep:

- `narrative_arc`,
- duration presets,
- script section-by-section,
- grounding guard retry,
- transcript artifact,
- optional MP3 artifact,
- Piper/Kokoro/pyttsx3 local TTS strategy,
- pydub/ffmpeg composition,
- language enforcement.

For desktop packaging, TTS models and ffmpeg need a clear first-run download or
bundling strategy.

### Mind Map Generator

Purpose:

- interactive course overview mind maps.

Keep:

- `topic_clusters`,
- module-pack fast path,
- batch outline extraction,
- course synthesis,
- parsed-structure fallback,
- theme fallback,
- enrichment,
- language adaptation,
- tree balancing,
- Markmap JSON artifact,
- standalone HTML artifact.

### Future/External Presentation Generator

Purpose:

- generate PPTX/PDF slide decks from TeacherLM course context.

This can be built in or connected as an external MCP agent.

Expected flow:

1. Student connects or enables a presentation generator.
2. TeacherLM reads its manifest and permissions.
3. The UI shows "Presentation" as an output type.
4. Student selects source files and options.
5. TeacherLM retrieves `topic_clusters` context, plus equations/tables when
   useful.
6. TeacherLM sends `GeneratorInput` to the presentation agent.
7. The agent returns a `presentation` output with PPTX/PDF artifacts.
8. TeacherLM stores artifacts locally and shows them in generated items.

The presentation agent should not query the database directly. It should use
TeacherLM-provided context or request additional context through scoped MCP
tools.

## MCP And External Agent Architecture

TeacherLM Desktop should be a local orchestrator platform.

There are two directions:

1. TeacherLM exposes local course-memory tools to external agents.
2. TeacherLM consumes external generator agents as output types.

### TeacherLM MCP Server

Expose a local MCP server that external agents can connect to after user
permission.

Suggested tools:

| Tool | Purpose |
| --- | --- |
| `teacherlm.list_conversations` | List local courses/workspaces allowed for this agent |
| `teacherlm.list_sources` | List uploaded files and readiness state |
| `teacherlm.retrieve_context` | Retrieve grounded chunks for a query/output type |
| `teacherlm.get_chunk` | Fetch one chunk by ID |
| `teacherlm.get_course_outline` | Return local course outline/module structure |
| `teacherlm.get_learner_state` | Return scoped learner state |
| `teacherlm.create_artifact` | Store an artifact in TeacherLM's local artifact store |
| `teacherlm.report_learner_updates` | Let an agent report concepts covered/struggled |
| `teacherlm.get_runtime_options` | Return allowed model/language options |

Suggested resources:

- course outline,
- selected source manifest,
- artifact manifest,
- generator contract schema.

Security rules:

- external agents get no direct SQLite path,
- external agents get no arbitrary filesystem access,
- external agents only see conversations/sources the student approved,
- external agents must declare permissions,
- network access must be explicit,
- write actions must be audited,
- API keys are never exposed to agents unless the user explicitly configures a
  provider for that agent.

### MCP Generator Adapter

The local generator router should support an MCP adapter.

Generic flow:

```text
UI request
  -> local generator router
  -> resolve generator manifest
  -> retrieve context locally
  -> build GeneratorInput
  -> call MCP agent tool
  -> validate GeneratorOutput
  -> store artifacts locally
  -> merge learner updates
  -> stream result to UI
```

The MCP agent may either:

- accept the full `GeneratorInput` in one call,
- or accept a task brief and call TeacherLM MCP tools for context as needed.

The second style is more flexible but needs stricter permissions and tracing.

### Example: Connected Presentation Agent

```text
Student clicks "Connect generator"
  -> selects Presentation Agent MCP server
  -> TeacherLM reads manifest
  -> TeacherLM asks for permissions:
       read selected course context
       read learner state
       create presentation artifacts
  -> student approves
  -> Presentation appears in output buttons
```

Generation:

```text
Student asks for presentation
  -> TeacherLM retrieves topic_clusters context
  -> TeacherLM includes source chunks and learner state
  -> Presentation agent creates deck
  -> Agent stores PPTX/PDF through teacherlm.create_artifact
  -> TeacherLM records output and citations
```

This lets TeacherLM gain new powers without merging every generator into the
main app.

## Local API Surface

The local API can stay close to the current backend routes:

- `/api/health`,
- `/api/conversations`,
- `/api/conversations/{id}/files`,
- `/api/conversations/{id}/chat`,
- `/api/conversations/{id}/generate`,
- `/api/generators`,
- `/api/conversations/{id}/coursebuilder`,
- `/api/conversations/{id}/knowledge-checks`,
- `/api/conversations/{id}/knowledge-graph`,
- `/api/conversations/{id}/review-tests`,
- `/api/settings/runtime`,
- `/api/artifacts/{id}`,
- `/api/mcp/generators`,
- `/api/mcp/permissions`.

The UI should not call built-in or external generator services directly. It
should call the local orchestrator.

## Streaming

Keep SSE-compatible event names because the current frontend already understands
the stream shape.

Events:

- `analysis`,
- `progress`,
- `sources`,
- `token`,
- `chunk`,
- `artifact`,
- `done`,
- `error`.

Inside Tauri, there are two acceptable options:

- keep local HTTP streaming over `127.0.0.1`,
- or bridge streams through Tauri commands/events later.

The first milestone should keep local HTTP SSE because it preserves the existing
frontend and generator behavior.

## Frontend Requirements

The desktop UI should preserve the current workspace:

- sources sidebar,
- upload and ingestion status,
- source-file selection,
- generated course pane,
- chat pane,
- output-type buttons,
- generator dialog,
- learner progress panel,
- generated artifact history,
- runtime settings.

Recommended frontend stack:

- React,
- TypeScript,
- Vite or another Tauri-friendly bundler,
- Tailwind CSS,
- Radix primitives,
- lucide icons,
- React Query,
- Zustand,
- react-markdown,
- KaTeX,
- Markmap,
- Mermaid,
- PDF viewer support,
- podcast/audio rendering.

The current Next.js frontend can be mined for components and logic, but the
desktop app should avoid depending on a hosted Next.js server.

## Model And Provider Settings

Default local model path:

- Ollama for LLM calls,
- fastembed for embeddings and reranking,
- local TTS models for podcasts.

Optional external providers:

- OpenAI,
- Anthropic,
- OpenAI-compatible APIs,
- llama-cloud parser.

Rules:

- local/offline is the default,
- external providers require explicit student configuration,
- settings UI must disclose what leaves the device,
- API keys must be encrypted locally,
- runtime options flow through `GeneratorInput.options["llm"]`,
- forced language flows through `GeneratorInput.options["language"]`.

## Artifact Handling

Replace MinIO with the local artifact store.

Artifact records should include:

```json
{
  "id": "artifact-id",
  "type": "quiz",
  "filename": "quiz.json",
  "local_key": "artifacts/quizzes/...",
  "mime_type": "application/json",
  "created_at": "timestamp",
  "source_message_id": "message-id"
}
```

The UI receives:

```json
{
  "type": "quiz",
  "url": "teacherlm-local://artifact/artifact-id",
  "filename": "quiz.json",
  "key": "artifact-id"
}
```

Supported artifact types:

- quiz JSON,
- mindmap JSON,
- mindmap HTML,
- podcast transcript,
- podcast MP3,
- presentation PPTX,
- presentation PDF,
- report PDF/DOCX,
- chart/diagram JSON/SVG/Mermaid.

## Security And Privacy

The desktop app must be privacy-first.

Rules:

- uploaded files stay local unless the student opts into a cloud parser/model,
- local database path is not exposed to generators,
- external agents get scoped tools, not raw filesystem access,
- generated artifacts remain local unless exported by the student,
- every external agent run has a trace ID,
- every external write has an audit record,
- permissions are per agent and revocable,
- secrets are encrypted at rest,
- logs must not store API keys or full private prompts by default.

## Observability And Evals

Keep local traceability because retrieval and agent routing need debugging.

Log locally:

- ingestion stages,
- parser used,
- chunk counts,
- generated question counts,
- embedding model,
- retrieval query,
- HyDE text hash or preview when tracing is enabled,
- candidate source: dense, HyDE, BM25, graph,
- RRF ranks,
- reranker scores,
- final chunk IDs,
- generator selected,
- external agent selected,
- tool calls,
- artifacts written,
- learner updates merged.

Adapt existing eval scripts to run against local SQLite/vector data:

- retrieval eval,
- course-context eval,
- dense-only vs BM25-only vs hybrid RRF comparison,
- HyDE on/off comparison,
- graph search on/off comparison,
- source-file filtering tests,
- generator contract validation.

Minimum eval variants:

| Variant | Purpose |
| --- | --- |
| dense_only | semantic baseline |
| bm25_only | exact-term baseline |
| dense_plus_hyde | test HyDE contribution |
| hybrid_rrf | current core behavior |
| hybrid_rrf_plus_hyde | target local retrieval |
| hybrid_rrf_plus_graph | graph contribution |
| full_local_rag | production local path |

## Packaging

Target installable desktop builds:

- Windows first,
- macOS later,
- Linux optional.

Packaging concerns:

- bundle the Tauri app,
- bundle or install the local Python sidecar,
- bundle required Python wheels or produce a sidecar executable,
- create app-data directories on first launch,
- run DB migrations on startup,
- download optional models on first use with clear progress,
- detect Ollama availability,
- guide the student through local model setup,
- keep all ports bound to `127.0.0.1`,
- shut down sidecars cleanly when the app exits.

Production sidecar options:

- PyInstaller executable,
- Nuitka executable,
- embedded Python distribution,
- or a Rust-native subset later if Python packaging becomes the bottleneck.

First implementation should prioritize parity and reliability over tiny installer
size.

## Build Milestones

### Milestone 1: Desktop Shell And Local API

- Scaffold Tauri app.
- Port the main workspace UI.
- Start local Python sidecar from Tauri.
- Health-check sidecar.
- Store local settings.
- Show empty workspace.

### Milestone 2: Local Storage

- Add SQLite schema and migrations.
- Add local object/artifact store.
- Port conversation and message APIs.
- Port file records.
- Add artifact read/export endpoints.

### Milestone 3: Local Ingestion

- Implement local parsers.
- Port cleaning, course intake, section extraction, and chunking.
- Generate hypothetical questions per chunk.
- Embed chunks.
- Build local vector/BM25 indexes.
- Rebuild concepts, learning map, graph, and CourseBuilder.

### Milestone 4: Retrieval Parity

- Port `RetrievalOrchestrator`.
- Port output-type retrieval modes.
- Add source-file filtering.
- Add RRF fusion.
- Add reranking.
- Add graph candidates and graph-neighbor expansion.
- Add HyDE query expansion.
- Add eval scripts for local data.

### Milestone 5: Built-In Generators

- Port `teacher_gen`.
- Port `quiz_gen`.
- Port `mindmap_gen`.
- Port `podcast_gen`.
- Replace MinIO artifact storage with local artifacts.
- Keep `GeneratorInput` and `GeneratorOutput` stable.
- Keep SSE events stable.

### Milestone 6: MCP Agent Hub

- Add generator manifest registry.
- Add external-agent permissions UI.
- Add TeacherLM MCP server tools.
- Add MCP generator adapter.
- Connect a sample presentation agent.
- Validate outputs against schemas.
- Store traces and artifacts.

### Milestone 7: Packaging

- Build Windows installer.
- Add model/download onboarding.
- Add startup recovery.
- Add update strategy.
- Add local backup/export.

## Open Implementation Decisions

These should be decided in the next session before coding too deeply:

- Use Vite React or adapt Next.js static export for the Tauri UI.
- Use embedded qdrant-client local mode, bundled Qdrant sidecar, LanceDB, or
  sqlite-vec for vectors.
- Use pure local parsers first, or keep optional `llama-cloud` parser UI in
  milestone 1.
- Run built-in generators in-process inside the local API or as supervised
  localhost/stdio child services.
- Choose PyInstaller, Nuitka, or embedded Python for production packaging.
- Define the first MCP presentation agent contract.

## Definition Of Done For The Local Rewrite

The local desktop app is not done until:

- a student can install it without Docker,
- no Electron dependency exists,
- uploaded files are stored locally,
- parsed text and artifacts are stored locally,
- vector and graph indexes are stored locally,
- teacher chat works with citations,
- citation scores come from local retrieval/reranking,
- quiz generation works,
- mind map generation works,
- podcast transcript works,
- podcast audio works when local TTS is available,
- learner state updates locally,
- source-file selection works,
- RRF, BM25, dense search, graph search, generated chunk questions, reranking,
  and HyDE are all testable,
- a connected external generator can be registered,
- a presentation agent can receive TeacherLM context and return a presentation
  artifact through the generator contract,
- permissions for external agents are visible and revocable,
- eval scripts can compare retrieval variants locally.

