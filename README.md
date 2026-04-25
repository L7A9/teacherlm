# TeacherLM — AI Tutor Grounded in Your Course Materials

> A multi-agent, Retrieval-Augmented Generation (RAG) system that turns uploaded
> course documents into a personal AI teacher. Students chat, generate quizzes,
> create flashcards, and more — with **every answer grounded exclusively in the
> uploaded files**, never the LLM's training data.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [System Architecture](#2-system-architecture)
3. [Technology Stack](#3-technology-stack)
4. [Repository Structure](#4-repository-structure)
5. [The RAG Pipeline — Step by Step](#5-the-rag-pipeline--step-by-step)
6. [Generators (Agents)](#6-generators-agents)
7. [Shared Core Library](#7-shared-core-library-packagesteacherlm_core)
8. [Platform Backend](#8-platform-backend)
9. [Platform Frontend](#9-platform-frontend)
10. [Deployment & Infrastructure](#10-deployment--infrastructure)
11. [Key Design Decisions & Rationale](#11-key-design-decisions--rationale)

---

## 1. Project Overview

**TeacherLM** is a full-stack AI tutoring platform. A student:

1. **Uploads** course files (PDFs, slides, documents).
2. **Chats** with a warm, encouraging AI teacher that only answers from the uploaded materials.
3. **Generates** quizzes, flashcards, and other study artifacts — all derived from the course content.
4. **Tracks progress** as the system monitors mastery per concept over time.

### Core Principles

| Principle | Description |
|---|---|
| **Strict Grounding** | Every factual claim must trace back to a retrieved chunk. If the material doesn't cover it, the system says so — it never hallucinates from training data. |
| **Learner-Aware** | The system maintains a per-conversation **LearnerState** (understood concepts, struggling concepts, mastery scores) and adapts its behaviour accordingly. |
| **Multi-Agent / Plugin Architecture** | Each output type (chat, quiz, flashcards, …) is a self-contained **generator** microservice with its own pipeline, models, and prompts. The backend is a thin orchestrator. |
| **Fully Local LLM** | All inference runs through **Ollama** on the host machine. No OpenAI, no cloud LLM APIs — complete data privacy. |

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Next.js 14)                      │
│   Conversation list • Chat UI • File upload • Quiz/Flashcard UI    │
└──────────────────────────────┬─────────────────────────────────────┘
                               │  HTTP / SSE
┌──────────────────────────────▼─────────────────────────────────────┐
│                         BACKEND (FastAPI)                           │
│  ┌─────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────────────┐   │
│  │ Routers │ │Dispatcher│ │  Retrieval   │ │ Learner Tracker  │   │
│  │(chat,   │ │(Registry │ │ Orchestrator │ │ (mastery math,   │   │
│  │files,   │ │ + Router │ │ (mode picker │ │  concept tracking)│  │
│  │generate)│ │ + Adapt.)│ │  + hybrid)   │ │                  │   │
│  └─────────┘ └────┬─────┘ └──────┬───────┘ └──────────────────┘   │
│                    │              │                                  │
│  ┌─────────────┐   │   ┌──────────▼──────────┐                     │
│  │ Ingestion   │   │   │   Vector Service    │                     │
│  │ Worker (arq)│   │   │ (Qdrant + fastembed)│                     │
│  └──────┬──────┘   │   └─────────────────────┘                     │
│         │          │                                                │
└─────────┼──────────┼────────────────────────────────────────────────┘
          │          │ HTTP / SSE
          │    ┌─────▼──────────────────────────────────┐
          │    │         GENERATOR MICROSERVICES         │
          │    │  ┌───────────┐ ┌─────────┐ ┌─────────┐│
          │    │  │teacher_gen│ │quiz_gen │ │flash_gen││
          │    │  │  :8001    │ │  :8002  │ │  :8005  ││
          │    │  └───────────┘ └─────────┘ └─────────┘│
          │    └────────────────────────────────────────┘
          │
┌─────────▼──────────────────────────────────────────────────────────┐
│                       INFRASTRUCTURE                                │
│  PostgreSQL 16  │  Redis 7  │  Qdrant  │  MinIO  │  Ollama (host)  │
└─────────────────────────────────────────────────────────────────────┘
```

**Data flow for a user message:**

1. Frontend sends a message via `POST /api/conversations/{id}/chat`.
2. Backend persists the user turn in **PostgreSQL**.
3. **Retrieval Orchestrator** picks a retrieval mode (e.g., `semantic_topk` for chat) and fetches relevant chunks from **Qdrant** using hybrid retrieval (dense + BM25 → Reciprocal Rank Fusion).
4. Backend loads the **LearnerState** and assembles a `GeneratorInput` payload.
5. The **Dispatcher** routes the payload over HTTP/SSE to the correct generator.
6. The generator runs its multi-step pipeline (analysis → reranking → LLM generation → learner update extraction) and streams SSE events back.
7. Backend proxies the SSE stream to the frontend and persists the assistant turn + learner state updates on completion.

---

## 3. Technology Stack

### 3.1 Backend

| Technology | Version | Role | Why We Chose It |
|---|---|---|---|
| **Python** | 3.14+ | Core language | Latest async capabilities, modern typing (`type[T]` generics), strict type safety. |
| **FastAPI** | ≥ 0.135 | REST + SSE API framework | Best async Python web framework; automatic OpenAPI docs, dependency injection, Pydantic-native. |
| **Pydantic V2** | ≥ 2.12 | Data validation & settings | Strict schemas for all I/O contracts; `model_json_schema()` feeds directly into Ollama's `format=` for constrained generation. |
| **SQLAlchemy 2.0** | ≥ 2.0.36 | ORM + async DB access | Modern async ORM with `mapped_column`, `DeclarativeBase`; used with **asyncpg** driver. |
| **Alembic** | ≥ 1.14 | Database migrations | Schema evolution management for the PostgreSQL database. |
| **Ollama** | Python lib ≥ 0.4 | Local LLM inference | Runs LLMs entirely on the local machine; native `format=` parameter for JSON-structured outputs — no LangChain needed. |
| **Qdrant** | Client ≥ 1.12 | Vector database | Purpose-built for similarity search; per-conversation collections with metadata filtering. |
| **fastembed** | ≥ 0.4 | Text embeddings | Fast, CPU-friendly embedding (`BAAI/bge-small-en-v1.5`, 384-dim). Lighter than sentence-transformers. |
| **rank-bm25** | ≥ 0.2.2 | Sparse retrieval | BM25Okapi keyword retrieval for the hybrid search pipeline. |
| **LlamaCloud** | ≥ 1.0 | Document parsing | Cloud API that converts PDFs/slides into clean Markdown. NOT llama-parse (deprecated). |
| **MinIO** | SDK ≥ 7.2 | S3-compatible object storage | Stores uploaded files, parsed Markdown, and generated artifacts (quiz JSONs, flashcard exports). |
| **Redis + arq** | Redis 7 / arq ≥ 0.26 | Task queue | Background job processing for the async file ingestion pipeline (parse → chunk → embed). |
| **httpx + httpx-sse** | ≥ 0.28 / ≥ 0.4 | HTTP client | The Dispatcher uses httpx with SSE support to communicate with generator microservices. |
| **sse-starlette** | ≥ 2.1 | Server-Sent Events | Streams generator responses token-by-token to the frontend in real time. |

### 3.2 Frontend

| Technology | Version | Role | Why We Chose It |
|---|---|---|---|
| **Next.js** | 14 | React framework | App Router, server components, file-based routing. |
| **React** | 18.3 | UI library | Component-based UI with hooks. |
| **TypeScript** | 5.6+ | Type safety | Catches errors at compile time across the entire frontend. |
| **TailwindCSS** | 3.4 | Utility-first CSS | Rapid, consistent styling with a custom dark-mode design system. |
| **Zustand** | 5.0 | State management | Lightweight, hooks-based global state (conversation store, progress store, UI store). |
| **TanStack React Query** | 5.59 | Server state management | Caching, refetching, and synchronisation of API data (conversations, files, messages). |
| **Radix UI** | Latest | Accessible primitives | Dialog, Tooltip, Slot — fully accessible, unstyled headless components. |
| **react-markdown** | 9.0 | Markdown rendering | Renders teacher responses (Markdown) with plugins for math (KaTeX), diagrams (Mermaid), and GitHub-Flavored Markdown. |
| **KaTeX** | 0.16 | LaTeX math rendering | Renders inline and display math in teacher responses. |
| **Mermaid** | 11.4 | Diagram rendering | Renders flowcharts and diagrams embedded in teacher responses. |
| **react-dropzone** | 14.3 | File upload UI | Drag-and-drop file uploads for course documents. |
| **Lucide React** | 0.460 | Icon library | Consistent, lightweight SVG icons throughout the interface. |
| **Sonner** | 1.7 | Toast notifications | Non-intrusive success/error feedback. |

### 3.3 LLM Models (via Ollama)

| Model | Parameters | Quantisation | Used By |
|---|---|---|---|
| **Llama 3.1 8B Instruct** | 8B | Q4_K_M | Chat responses (teacher_gen), quiz generation, flashcard generation |
| **Llama 3.2** | 3B | Default | Query analysis, learner-update extraction (lightweight structured output tasks) |

### 3.4 Embedding & Reranking Models (via fastembed)

| Model | Dimensions | Role |
|---|---|---|
| **BAAI/bge-small-en-v1.5** | 384 | Dense text embeddings for semantic search |
| **BAAI/bge-reranker-base** | — | Cross-encoder reranker for refining retrieved chunk relevance |

### 3.5 NLP Models

| Model | Library | Role |
|---|---|---|
| **en_core_web_sm** | spaCy | Named Entity Recognition + noun chunk extraction for flashcard concept mining |

### 3.6 Infrastructure

| Component | Image | Role |
|---|---|---|
| **PostgreSQL 16** | `postgres:16-alpine` | Relational storage for conversations, messages, uploaded files, and learner state |
| **Redis 7** | `redis:7-alpine` | Message broker for the arq background task queue |
| **Qdrant** | `qdrant/qdrant:latest` | Vector database — one collection per conversation for isolated embedding retrieval |
| **MinIO** | `minio/minio:latest` | S3-compatible object store for file uploads, parsed Markdown, and generated artifacts |
| **Docker Compose** | — | Orchestrates all 9 services (Postgres, Redis, Qdrant, MinIO, backend, arq worker, teacher_gen, quiz_gen, flashcard_gen, frontend) |

---

## 4. Repository Structure

```
teacherlm/
├── README.md                       # ← This file
├── CLAUDE.md                       # AI-assistant instructions (project conventions)
├── generators_registry.json        # Plugin registry — declares all generators
├── run.sh                          # CLI to start/stop/rebuild the Docker stack
│
├── packages/
│   └── teacherlm_core/             # Shared Python library
│       ├── pyproject.toml
│       └── teacherlm_core/
│           ├── retrieval/           # Hybrid retriever, BM25, reranker, retrieval modes
│           ├── llm/                 # Ollama client, streaming helpers, structured output
│           ├── schemas/             # GeneratorInput, GeneratorOutput, LearnerState, Chunk
│           ├── confidence/          # Groundedness + coverage scoring
│           └── prompts/             # Shared teacher voice + tone guidelines
│
├── generators/
│   ├── teacher_gen/                 # Chat / Q&A generator (the "teacher")
│   │   ├── pipeline.py             # Multi-step pipeline: analyze → rerank → generate → score
│   │   ├── services/               # Query analyzer, HyDE, response mode, confidence, learner updates
│   │   └── prompts/                # Mode-specific prompt templates
│   ├── quiz_gen/                    # Quiz generator
│   │   ├── pipeline.py             # Extract concepts → plan difficulty → generate questions → validate
│   │   └── services/               # Bloom's taxonomy, distractor engine, quality validation
│   ├── flashcard_gen/              # Flashcard generator
│   │   ├── pipeline.py             # Mine concepts → prioritize → generate cards → SM-2 schedule
│   │   └── services/               # Concept miner (spaCy), cloze deletion, SM-2 scheduler
│   └── mindmap_gen/                # (Planned — not yet implemented)
│
└── platform/
    ├── docker-compose.yml          # Defines all 10 services
    ├── .env / .env.example         # Environment variables
    ├── backend/
    │   ├── main.py                 # FastAPI app creation + lifespan
    │   ├── config.py               # Centralised settings (Pydantic BaseSettings)
    │   ├── db/                     # SQLAlchemy models + session management
    │   ├── routers/                # API endpoints (chat, generate, files, conversations, health)
    │   ├── services/               # Core services (retrieval, vector, chunking, parsing, storage, learner tracker)
    │   ├── dispatcher/             # Generator registry + routing + adapters (API, MCP)
    │   ├── workers/                # arq background worker for file ingestion
    │   └── schemas/                # Pydantic models for API request/response
    └── frontend/
        ├── app/                    # Next.js App Router pages
        ├── components/             # UI: chat, artifacts, files, progress, workspace
        ├── hooks/                  # React hooks (useChatStream, useFiles, useConversations)
        ├── stores/                 # Zustand global state
        └── lib/                    # API client, SSE parser, TypeScript types, utilities
```

---

## 5. The RAG Pipeline — Step by Step

### Step 1: Document Ingestion

When a user uploads a file:

1. **Upload** — The file is persisted in **MinIO** object storage and a record is inserted into PostgreSQL with status `uploaded`.
2. **Parse** — An **arq** background worker picks up the job. It sends the raw bytes to **LlamaCloud** (>= 1.0), which returns clean **Markdown**. The Markdown is stored back in MinIO.
3. **Chunk** — The `ChunkingService` splits the Markdown into semantic chunks using a **paragraph → sentence → sliding-window merge** strategy:
   - Split on paragraph boundaries (`\n\n`).
   - Further split on sentence-ending punctuation.
   - Merge sentences into chunks targeting **512 tokens** with **50-token overlap** between consecutive chunks.
   - Token count is approximated at 1.3 tokens per whitespace word (avoids loading a tokeniser).
4. **Embed** — The `VectorService` embeds each chunk using **fastembed** (`BAAI/bge-small-en-v1.5`, 384-dim) and upserts them into a **Qdrant** collection scoped to the conversation (`conv_{conversation_id}`).
5. **Index** — Payload indices are created on `source` and `file_id` fields for efficient filtering.

### Step 2: Retrieval

When a query arrives, the **Retrieval Orchestrator** selects a retrieval mode based on the requested output type:

| Output Type | Retrieval Mode | Strategy |
|---|---|---|
| `text` (chat) | `semantic_topk` | Top-K chunks closest to the query via hybrid retrieval |
| `quiz`, `flashcards` | `coverage_broad` | MMR (Maximal Marginal Relevance) — balances relevance with diversity |
| `report`, `presentation` | `topic_clusters` | K-means clustering on chunk embeddings; one representative per cluster |
| `podcast` | `narrative_arc` | Intro-like + query-relevant middle + conclusion-like chunks |
| `chart` | `relationship_dense` | Ranks chunks by entity + verb density (regex heuristic) |

#### Hybrid Retrieval (Core)

The `HybridRetriever` combines **two signals**:

1. **Dense retrieval** — Embeds the query with fastembed and searches Qdrant.
2. **Sparse retrieval (BM25)** — Tokenises the query and all chunks, then ranks by BM25Okapi.

Results are fused using **Reciprocal Rank Fusion (RRF)**:
```
score(doc) = Σ  1 / (K + rank_i(doc))    where K = 60
```

**Why hybrid?** Dense retrieval captures semantic meaning ("What causes photosynthesis?" matches "light-dependent reactions"), while BM25 captures exact keyword matches ("What is the Henderson-Hasselbalch equation?" matches that exact term). The fusion ensures both are covered.

### Step 3: Reranking

After initial retrieval, chunks are refined by a **cross-encoder reranker** (`BAAI/bge-reranker-base` via fastembed). The cross-encoder scores each (query, chunk) pair jointly, producing more precise relevance scores than the bi-encoder used for initial retrieval.

**Optionally**, the teacher_gen uses **HyDE (Hypothetical Document Embeddings)**: the LLM first generates a hypothetical textbook answer to the query, then the reranker scores chunks against `query + hypothetical answer`. This closes the vocabulary gap between a student's question phrasing and textbook language.

### Step 4: Generation

The retrieved, reranked chunks are injected into the generator's prompt as `CONTEXT CHUNKS` and the LLM generates a response. See the [Generators section](#6-generators-agents) for details on each pipeline.

### Step 5: Confidence Scoring

After generation, the system scores response quality:

- **Groundedness score** — Mean per-sentence overlap: what fraction of non-stopword terms in each sentence of the response also appear in the retrieved chunks. Near 1.0 = well-grounded; near 0.0 = potential hallucination.
- **Coverage score** — What fraction of distinct query keywords appear at least once across the top retrieved chunks. Measures whether the retrieval captured the question's scope.

### Step 6: Learner State Update

After each interaction, the **Learner Tracker** updates the student's mastery model:

- **Demonstrated understanding**: mastery += 0.2 × (1 − current_mastery)
- **Showed confusion**: mastery × = 0.7
- **Thresholds**: ≥ 0.7 → "understood"; ≤ 0.3 → "struggling"
- **Turns since progress** is tracked to detect "stuck" students

This state feeds back into future interactions — the teacher adapts its tone, quizzes focus on weak areas, and flashcards prioritise struggling concepts.

---

## 6. Generators (Agents)

Generators are the **"agents"** of the system. Each is a self-contained FastAPI microservice with its own Docker container, prompt templates, and multi-step processing pipeline.

### 6.1 Generator I/O Contract

All generators share a strict input/output schema defined in `teacherlm_core`:

**Input** (`GeneratorInput`):
```json
{
  "conversation_id": "uuid",
  "user_message": "string",
  "context_chunks": [{"text", "source", "score", "chunk_id"}],
  "learner_state": {
    "understood_concepts": [],
    "struggling_concepts": [],
    "mastery_scores": {},
    "session_turns": 0
  },
  "chat_history": [{"role", "content"}],
  "options": {}
}
```

**Output** (`GeneratorOutput`):
```json
{
  "response": "markdown string",
  "generator_id": "teacher_gen",
  "output_type": "text",
  "artifacts": [{"type", "url", "filename"}],
  "sources": [{"text", "source", "score"}],
  "learner_updates": {
    "concepts_covered": [],
    "concepts_demonstrated": [],
    "concepts_struggled": []
  },
  "metadata": {}
}
```

### 6.2 Teacher Generator (`teacher_gen`) — Port 8001

**Purpose**: The core chat agent. Acts as a warm, supportive teacher that explains concepts, answers questions, and guides the student — strictly from the uploaded materials.

**Pipeline (6 steps):**

| Step | Service | Technique | Why |
|---|---|---|---|
| 1. **Query Analysis** | `query_analyzer` | LLM structured output → `QueryAnalysis` schema (intent, confusion level, target concept) | Understand what the student is asking and their emotional state |
| 2. **Response Mode Selection** | `response_mode` | Rule-based decision tree on analysis + learner state | Choose between `explain`, `guide`, `quiz_back`, or `affirm` modes |
| 3. **HyDE + Reranking** | `hyde_generator` + `CrossEncoderReranker` | Generate hypothetical answer → concat with query → cross-encoder rerank | Close vocabulary gap between student language and textbook language |
| 4. **Off-topic Guard** | Relevance threshold check | Cross-encoder score < threshold → refuse politely | Prevent hallucination on questions not covered by the materials |
| 5. **LLM Response Generation** | `llm_service` | Streaming Ollama chat with mode-specific system prompt | Stream tokens to user in real-time with appropriate teaching style |
| 6. **Post-generation Analysis** | `confidence_scorer` + `learner_analyzer` | Groundedness scoring + LLM extraction of concepts covered/demonstrated/struggled | Update learner state and measure response quality |

**Response Modes**:
- **Explain** — Direct explanation from the materials (default for new questions or stuck students).
- **Guide** — Socratic guidance, asks leading questions (when confusion detected).
- **Quiz Back** — Tests understanding before confirming (when student seeks confirmation but isn't confident).
- **Affirm** — Validates correct understanding (when student demonstrates mastery).

**Models used**: 3 separate Ollama models —
- `llama3.1:8b-instruct-q4_K_M` for main chat responses (higher quality).
- `llama3.2:latest` for query analysis (structured output, lighter model).
- `llama3.2:latest` for learner-update extraction (structured output, lighter model).

### 6.3 Quiz Generator (`quiz_gen`) — Port 8002

**Purpose**: Generate adaptive quizzes with multiple question types, grounded in course materials and tailored to the student's mastery level.

**Pipeline (7 steps):**

| Step | Service | Technique | Why |
|---|---|---|---|
| 1. **Concept Extraction** | `concept_extractor` | LLM structured output → concepts grouped by **Bloom's Taxonomy** (remember, understand, apply, analyze) | Map source material to testable concepts at different cognitive levels |
| 2. **Difficulty Planning** | `difficulty_adapter` | Learner-aware slot allocation (60% struggling / 30% coverage / 10% stretch) | Target weak areas while still covering breadth and pushing mastery |
| 3. **Question Generation** | `question_generator` | LLM structured output → MCQ, True/False, Fill-in-the-Blank per slot | Multiple question types keep the quiz engaging and test different skills |
| 4. **Distractor Enhancement** | `distractor_engine` | LLM generates plausible wrong answers from same context chunks | Better distractors make MCQs more pedagogically valuable |
| 5. **Quality Validation** | `quality_validator` | Schema validation + deduplication + Bloom distribution check | Ensures every question has valid fields and the quiz is well-balanced |
| 6. **Adaptive Intro** | LLM generation | Personalised intro message aware of learner's strengths/weaknesses | Encourages the student and sets expectations |
| 7. **Artifact Export** | `artifact_store` | Serialise to JSON → upload to MinIO → return presigned URL | The quiz is downloadable and renderable in the frontend |

**Question Types**: MCQ, True/False, Fill-in-the-Blank.

**Bloom's Taxonomy Integration**: Every concept is tagged with a Bloom's level. The difficulty adapter allocates questions across levels, pushing "understood" concepts to higher-order thinking (apply/analyze = "stretch") and reinforcing "struggling" concepts at lower levels (remember/understand).

### 6.4 Flashcard Generator (`flashcard_gen`) — Port 8005

**Purpose**: Generate spaced-repetition flashcard decks with SM-2 scheduling, prioritised by learner weaknesses.

**Pipeline (8 steps):**

| Step | Service | Technique | Why |
|---|---|---|---|
| 1. **Concept Mining** | `concept_miner` | **spaCy NER** + noun chunks + regex definition patterns ("X is Y") | Extract candidate concepts from text without an LLM call (fast, deterministic) |
| 2. **Boilerplate Filtering** | Regex patterns | Filter out authors, universities, copyright, page numbers, URLs | Course PDFs are full of metadata that shouldn't become flashcards |
| 3. **Priority Selection** | `priority_selector` | Rank by: struggling concepts first → high-occurrence concepts → rest | Focus study effort where it's needed most |
| 4. **Basic Card Generation** | `basic_card_gen` | LLM structured output → Q&A-style cards with source grounding | Classic flashcard format — question on front, answer on back |
| 5. **Cloze Deletion Cards** | `cloze_card_gen` | Regex-based sentence transformation (blank key term) | Fill-in-the-blank cards test active recall differently than Q&A |
| 6. **Deduplication** | `deduplicator` | Token-set overlap threshold | Remove near-duplicate cards |
| 7. **SM-2 Scheduling** | `sm2_scheduler` | Attach SuperMemo 2 metadata (ease factor, interval, repetitions) | Enable spaced repetition — cards are due at scientifically optimal intervals |
| 8. **Export** | `exporter` | JSON upload to MinIO | Downloadable deck for the frontend renderer |

**Card Types**: Basic (Q&A) and Cloze (fill-in-the-blank).

**SM-2 Algorithm**: Each card gets initial ease factor (2.5), interval (1 day), and a `due_at` timestamp. The frontend can later implement review sessions that update these values according to the SM-2 spaced repetition formula.

### 6.5 Planned Generators (Not Yet Implemented)

- **Mindmap Generator** — Concept relationship diagrams
- **Report Generator** — Study reports on covered material
- **Presentation Generator** — Slide-style output
- **Podcast Generator** — Audio-style explainers
- **Chart Generator** — Concept diagrams

---

## 7. Shared Core Library (`packages/teacherlm_core`)

A reusable Python package installed by every generator and the platform backend. Ensures consistency across the entire system.

### 7.1 `retrieval/` — Hybrid Retrieval Engine

| Module | Description |
|---|---|
| `hybrid_retriever.py` | Combines dense (Qdrant) + sparse (BM25) retrieval with **Reciprocal Rank Fusion** (RRF). |
| `bm25.py` | Wrapper around `rank_bm25.BM25Okapi`. Tokenises text with a Unicode-aware regex. |
| `reranker.py` | Cross-encoder reranker using fastembed's `TextCrossEncoder` (`BAAI/bge-reranker-base`). Rescores (query, chunk) pairs jointly. |
| `retrieval_modes.py` | Five retrieval strategies: `semantic_topk`, `coverage_broad` (MMR), `narrative_arc`, `topic_clusters` (K-means), `relationship_dense` (entity density). |

### 7.2 `llm/` — Ollama Integration

| Module | Description |
|---|---|
| `ollama_client.py` | Async wrapper around `ollama.AsyncClient`. Provides `chat()`, `chat_structured[T]()` (JSON-schema-constrained output), and `stream_chat()`. |
| `structured.py` | `generate_structured[T]()` — calls the LLM with a Pydantic schema constraint, with automatic retry + self-repair on validation failures. |
| `streaming.py` | `stream_as_sse()` — wraps an async text-delta iterator into SSE frames. |

### 7.3 `schemas/` — Shared Data Contracts

| Schema | Description |
|---|---|
| `Chunk` | A piece of retrieved text with `text`, `source`, `score`, `chunk_id`, and `metadata`. |
| `GeneratorInput` | The universal input contract every generator receives. |
| `GeneratorOutput` | The universal output contract every generator returns. |
| `LearnerState` | Per-conversation mastery model: understood/struggling concepts, mastery scores, session turns. |
| `LearnerUpdates` | What a generator reports back: concepts covered, demonstrated, struggled. |

### 7.4 `confidence/` — Response Quality Scoring

| Module | Description |
|---|---|
| `groundedness.py` | Token-overlap-based groundedness: how much of the response is traceable to the retrieved chunks. |
| `coverage.py` | Query keyword coverage: what fraction of the question's content words appear in the retrieved chunks. |

### 7.5 `prompts/` — Shared Prompt Templates

| File | Description |
|---|---|
| `teacher_voice.txt` | Core personality prompt: warm, patient, grounding-first. Shared by all generators. |
| `citation_rules.txt` | Rules for how to cite sources ("from {filename}"). |
| `tone_guidelines.txt` | Emotional tone rules: encouraging, never condescending, adapts to demonstrated skill level. |

---

## 8. Platform Backend

### 8.1 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/conversations` | Create a new conversation |
| `GET` | `/api/conversations` | List all conversations |
| `GET` | `/api/conversations/{id}` | Get a single conversation |
| `PATCH` | `/api/conversations/{id}` | Update conversation title |
| `DELETE` | `/api/conversations/{id}` | Delete conversation + Qdrant collection |
| `POST` | `/api/conversations/{id}/files` | Upload a file → triggers ingestion pipeline |
| `GET` | `/api/conversations/{id}/files` | List files for a conversation |
| `DELETE` | `/api/conversations/{id}/files/{fid}` | Delete file + vectors |
| `GET` | `/api/conversations/{id}/messages` | List message history |
| `POST` | `/api/conversations/{id}/chat` | Send a message → SSE stream |
| `POST` | `/api/conversations/{id}/generate` | Generate a specific output type (quiz, flashcards) → SSE stream |
| `GET` | `/api/generators` | List available generators |
| `GET` | `/api/health` | Health check |

### 8.2 Dispatcher — Plugin Routing

The Dispatcher is the key architectural piece that makes the system extensible:

1. **Generator Registry** (`generators_registry.json`) — A JSON file declaring every generator with its ID, endpoint URL, output type, enabled status, and transport type (`api` or `mcp`).
2. **Registry** (`dispatcher/registry.py`) — Loads and caches the JSON registry. Provides lookups by ID, output type, or "chat default".
3. **Router** (`dispatcher/router.py`) — Resolves which generator to call and dispatches through the correct transport adapter.
4. **Adapters**:
   - `ApiAdapter` — HTTP POST to the generator's `/run` endpoint. Supports both one-shot JSON and SSE streaming.
   - `McpAdapter` — (Placeholder) for future MCP (Model Context Protocol) integration.

**Why this design?** Adding a new generator requires only: (1) implement a new microservice with a `/run` endpoint, (2) add an entry to `generators_registry.json`. Zero changes to the backend code.

### 8.3 Background Ingestion Worker

The `arq` worker runs a separate process that handles the slow file-ingestion pipeline asynchronously:
```
Upload → "uploaded" → Parse (LlamaCloud) → "parsing" → Chunk → "chunking" → Embed (fastembed) → "embedding" → "ready"
```

Each status transition is persisted in PostgreSQL so the frontend can show real-time progress. On failure, the status becomes `"failed"` with the error message.

---

## 9. Platform Frontend

### 9.1 Tech & Architecture

- **Next.js 14** with the App Router.
- **Dark mode first** design system with TailwindCSS.
- **Zustand** stores for global state: `conversationStore` (conversation list, active ID), `progressStore` (file ingestion status), `uiStore` (sidebar visibility).
- **TanStack React Query** for server-state (API data caching, automatic refetch, optimistic updates).
- **SSE streaming** via a custom `sse.ts` parser for real-time chat token delivery.

### 9.2 Key Components

| Component | Description |
|---|---|
| `ChatInput` | Message input with keyboard shortcuts and loading states |
| `MessageList` | Renders conversation history with scroll-to-bottom |
| `MessageBubble` | Renders a single message — user or assistant. Supports Markdown (react-markdown), math (KaTeX), and diagrams (Mermaid). |
| `OutputTypeButtons` | Toolbar to generate quizzes, flashcards, etc. |
| `GeneratorDialog` | Modal for configuring generation options (topic, count) |
| `QuizRenderer` | Interactive quiz UI — MCQ radio buttons, true/false toggles, fill-in-the-blank inputs, score calculation |
| `FlashcardRenderer` | Card-flip UI with swipe/keyboard navigation for reviewing flashcard decks |
| `ArtifactRenderer` | Routes artifact types to the correct specialised renderer |
| `FileDownload` | Handles binary artifact downloads |
| `ChartRenderer` | Renders SVG/Mermaid charts with pan-zoom |
| `PodcastPlayer` | Audio player for podcast-style artifacts |

### 9.3 Data Flow (Chat)

1. User types a message → `useChatStream` hook.
2. `POST /api/conversations/{id}/chat` → opens SSE connection.
3. SSE events arrive: `analysis`, `sources`, `token` (streamed deltas), `done`.
4. Tokens are appended to a local buffer → rendered in real time in the `MessageBubble`.
5. On `done`, the full response + metadata is committed to the conversation store.

---

## 10. Deployment & Infrastructure

### Running the Full Stack

```bash
# Start everything (first run builds all images):
./run.sh up

# Rebuild from scratch:
./run.sh rebuild

# View logs:
./run.sh logs [service_name]

# Stop:
./run.sh stop
```

### Required: Ollama on the Host

Ollama must be running on the host machine with the required models pulled:
```bash
ollama pull llama3.1:8b-instruct-q4_K_M
ollama pull llama3.2:latest
```

Docker services reach Ollama via `host.docker.internal:11434`.

### Ports

| Service | Port |
|---|---|
| Frontend | `3000` |
| Backend API | `8000` |
| teacher_gen | `8001` |
| quiz_gen | `8002` |
| flashcard_gen | `8005` |
| PostgreSQL | `5432` |
| Redis | `6379` |
| Qdrant | `6333`/`6334` |
| MinIO | `9000`/`9001` |

### Persistent Volumes

Docker Compose defines 5 named volumes:
- `postgres_data` — Database files
- `qdrant_data` — Vector embeddings
- `minio_data` — Uploaded files and artifacts
- `fastembed_cache` — Downloaded embedding models
- `hf_cache` — HuggingFace model cache

---

## 11. Key Design Decisions & Rationale

### Why local LLMs (Ollama) instead of OpenAI?

- **Data privacy**: Student course materials never leave the machine.
- **Cost**: No per-token API charges — unlimited usage once hardware is provisioned.
- **Transparency**: Fully reproducible; no vendor lock-in.
- **Offline capability**: Works without internet (after models are downloaded).

### Why not LangChain / LangGraph?

- **Pydantic V1 warnings** on Python 3.14 break strict mode.
- **Unnecessary abstraction** for our use case — Ollama's native `format=` parameter for structured output, plus a simple `OllamaClient` wrapper, replaces the entire chain/agent framework.
- **Debugging simplicity** — direct control over every LLM call instead of opaque chain internals.

### Why a microservice-per-generator architecture?

- **Independent scaling**: The teacher_gen (chat) handles real-time streaming and needs low latency. The quiz_gen makes multiple sequential LLM calls and can tolerate higher latency. Different resource profiles.
- **Independent deployment**: Update the quiz pipeline without restarting the chat service.
- **Independent models**: teacher_gen uses 3 models (chat + analysis + extraction). quiz_gen uses different model configurations. Each container manages its own Ollama client.
- **Plugin extensibility**: Add a new generator without modifying existing code.

### Why Reciprocal Rank Fusion (RRF) instead of just dense retrieval?

- Dense embedding models excel at **semantic similarity** but can miss **exact keyword matches** (especially domain-specific terms like equation names or acronyms).
- BM25 excels at **keyword/term matching** but misses **paraphrases** and **synonyms**.
- RRF is a principled, parameter-free fusion method that produces consistently better retrieval quality than either signal alone.

### Why 5 retrieval modes?

Different output types have fundamentally different information needs:
- Chat needs the **most relevant** chunks (semantic_topk).
- Quizzes need **broad coverage** to avoid repetitive questions (coverage_broad / MMR).
- Reports need **topic structure** (topic_clusters / K-means).
- Each mode is optimised for its downstream task.

### Why fastembed instead of sentence-transformers?

- **Smaller dependency footprint** (no PyTorch required).
- **CPU-optimised** via ONNX runtime — faster embedding on machines without a GPU.
- **Includes a cross-encoder reranker** via `TextCrossEncoder` — one library for both embedding and reranking.

### Why SM-2 for flashcards?

The **SuperMemo 2 algorithm** is the foundation of modern spaced-repetition systems (Anki, SuperMemo). By attaching SM-2 metadata at generation time, the flashcards are ready for review scheduling — the frontend can implement timed review sessions where card intervals grow exponentially with successful recalls.

### Why LlamaCloud for parsing instead of local PDF extraction?

- Course PDFs contain complex layouts: multi-column text, tables, embedded images, equations, headers/footers.
- LlamaCloud's parsing produces **clean, structured Markdown** that preserves document hierarchy.
- Local alternatives (PyMuPDF, pdfplumber) struggle with complex layouts and produce noisy output that degrades chunking and retrieval quality.
