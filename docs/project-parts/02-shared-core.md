# Shared Core Package

`packages/teacherlm_core` is the shared Python library installed by the backend and by every generator service. It carries the cross-service contract: schemas, retrieval helpers, LLM utilities, confidence scoring, and shared prompts.

The package is intentionally small compared with the backend. It does not know about SQLAlchemy models, MinIO objects, FastAPI routers, frontend state, or ARQ jobs. It provides reusable building blocks that other services compose.

## Package Metadata

| Path | Purpose |
| --- | --- |
| `packages/teacherlm_core/pyproject.toml` | Python package definition. Requires Python 3.14 or newer. Defines dependencies such as Pydantic V2, pydantic-settings, ollama, qdrant-client, fastembed, rank-bm25, httpx, and numpy. |
| `packages/teacherlm_core/README.md` | Package overview and local usage notes. |
| `packages/teacherlm_core/CLAUDE.md` | Local assistant guidance for the core package. |
| `packages/teacherlm_core/teacherlm_core/__init__.py` | Package initializer. |

## Configuration

| Path | Main object | Details |
| --- | --- | --- |
| `teacherlm_core/config.py` | `CoreSettings` | Pydantic settings object with `TEACHERLM_` environment prefix. Holds defaults for Ollama, Qdrant, embedding model, reranker model, retrieval parameters, and confidence thresholds. |

The core config is not the platform config. The backend has its own richer settings in `platform/backend/config.py`; core settings are for reusable low-level defaults.

## Shared Schemas

### `schemas/chunk.py`

Defines the chunk object shared by retrieval and generator inputs:

- `text`: chunk text shown to the model.
- `source`: human-readable source label.
- `score`: retrieval score.
- `chunk_id`: stable chunk identifier.
- `metadata`: flexible dictionary for section, heading, file, concept, and retrieval metadata.

This schema is intentionally generic so both Qdrant-backed backend chunks and generator-local operations can use it.

### `schemas/generator_io.py`

Defines the immutable generator contract.

Input:

- `GeneratorInput`
  - `conversation_id`
  - `user_message`
  - `context_chunks`
  - `learner_state`
  - `chat_history`
  - `options`

Output:

- `GeneratorArtifact`
  - `type`
  - `url`
  - `filename`
  - optional `key`
- `LearnerUpdates`
  - `concepts_covered`
  - `concepts_demonstrated`
  - `concepts_struggled`
- `GeneratorOutput`
  - `response`
  - `generator_id`
  - `output_type`
  - `artifacts`
  - `sources`
  - `learner_updates`
  - `metadata`

The backend uses these objects to validate generator requests and responses. Generators use them to keep `/run` endpoints consistent.

### `schemas/learner_state.py`

Defines the canonical learner progress model.

Important objects:

- `KnownConcept`: named concept with optional canonical ID and status.
- `ConceptProgress`: concept-level mastery, confidence, last-seen information, demonstrated/struggled flags, and linked objectives.
- `LearningPhase`: high-level learning phase data.
- `ObjectiveProgress`: objective-level progress and status.
- `PhaseProgress`: phase-level rollup.
- `LearnerState`: combines simple legacy fields with canonical concept, objective, and phase progress.

The simple fields remain important because generator input still includes:

- `understood_concepts`
- `struggling_concepts`
- `mastery_scores`
- `session_turns`

The richer fields let the backend and frontend show phase/objective mastery and remediation paths.

## LLM Utilities

### `llm/ollama_client.py`

The class name is `OllamaClient`, but the implementation supports several provider styles:

- `ollama`
- `openai`
- `openai_compatible`
- `anthropic`

Core responsibilities:

- Build chat payloads for the selected provider.
- Stream text chunks for chat-style output.
- Generate structured output using provider-native JSON controls when possible.
- Use Ollama's native `format=` parameter for structured local output.
- Use OpenAI-compatible `response_format` JSON schema when available.
- For Anthropic, convert system and user messages into the provider's expected shape and inject the schema into instructions.
- Parse JSON content that may arrive wrapped in code fences or surrounding prose.
- Raise errors that include response body text when HTTP calls fail.

This wrapper is used by backend services and generator-specific LLM services.

### `llm/structured.py`

Provides `generate_structured()`, a helper around LLM JSON generation.

Behavior:

- Calls a client with a Pydantic schema.
- Validates model output against the schema.
- On validation failure, retries with a repair prompt.
- Returns a typed Pydantic model rather than raw JSON.

This keeps generator code from duplicating validation/retry loops.

### `llm/streaming.py`

Provides SSE helpers:

- `format_sse`: converts event names and data into SSE text.
- `stream_as_sse`: wraps an async stream of events.
- `safe_sse_stream`: catches errors and emits friendly SSE error payloads.

Generator apps use this to expose streaming `/run` endpoints.

### `llm/runtime.py`

Holds request-scoped runtime LLM override data in a context variable.

Important behavior:

- `set_current_llm_options()` stores per-request options.
- `get_current_llm_options()` reads them.
- `build_llm_client_kwargs()` resolves provider, base URL, model, and API key.
- Defaults remain Ollama unless an override is explicitly enabled.

This allows frontend settings and backend runtime settings to flow into services without threading many arguments through every function.

### `llm/language.py`

Holds request-scoped forced language data.

Important behavior:

- Stores language code in a context variable.
- Converts codes such as `en-us`, `fr-fr`, `pt-br`, `de`, `ja`, `cmn`, and `hi` into language directives.
- `inject_language_directive()` appends a strict answer-language instruction to a system prompt.

This is used when the student selects a forced response language in settings.

## Retrieval Utilities

### `retrieval/bm25.py`

Defines a lightweight BM25 sparse search index.

Important details:

- Uses regex tokenization.
- Indexes chunk text.
- Also indexes metadata fields when present:
  - `heading_path`
  - `section_title`
  - `key_concepts`
  - `generated_questions`
- Returns core `Chunk` objects with BM25 scores.

The generated-questions metadata is important because ingestion can ask an LLM to create student-style questions for each chunk. Those questions improve lexical matching when the student's wording differs from the source wording.

### `retrieval/hybrid_retriever.py`

Defines `HybridRetriever`.

Responsibilities:

- Query Qdrant for dense semantic results.
- Query the local BM25 index for sparse lexical results.
- Fuse both ranked lists with reciprocal rank fusion.
- Convert Qdrant points into core `Chunk` objects.
- Expose `index_bm25()` so callers can refresh lexical search from known chunks.

Key constant:

- `RRF_K = 60`

The backend has its own retrieval orchestrator, but this core class captures the common hybrid retrieval pattern.

### `retrieval/reranker.py`

Defines a cross-encoder reranker wrapper around fastembed `TextCrossEncoder`.

Behavior:

- Lazily loads a reranker model.
- Scores query/document pairs.
- Runs scoring in a thread so async callers do not block the event loop.
- Returns top-k chunks with updated scores.

### `retrieval/retrieval_modes.py`

Defines reusable retrieval mode transformations:

- `semantic_topk`: keep nearest chunks.
- `coverage_broad`: prefer diverse coverage using an MMR-like Jaccard diversity step.
- `narrative_arc`: select introduction-like, middle/key-point, and conclusion-like chunks.
- `topic_clusters`: cluster chunk embeddings using a small k-means implementation.
- `relationship_dense`: prefer chunks dense with entities, verbs, and relational/factual language.

The backend uses richer policies on top of these ideas, but these functions encode the retrieval modes named in the root contract.

### `retrieval/evaluation.py`

Defines reusable retrieval evaluation types and metrics.

Important objects:

- `RetrievalCase`
- `CaseResult`

Important functions:

- `evaluate_case`
- `summarize_results`
- `result_to_dict`

Metrics:

- Hit rate
- Precision
- Recall
- Mean reciprocal rank
- nDCG
- Section recall
- Citation precision
- Source-document hit
- Latency

This module is used by backend scripts and tests to evaluate retrieval quality without requiring frontend involvement.

## Confidence Utilities

### `confidence/groundedness.py`

Computes groundedness by comparing generated answer sentences with source chunk vocabulary.

Important behavior:

- Splits answer text into sentences.
- Removes stopwords.
- Computes overlap between answer content tokens and chunk content tokens.
- Produces a mean score.

This is not a proof of factual correctness. It is a lightweight support signal for "does the answer use words and concepts present in sources?"

### `confidence/coverage.py`

Computes how much of the user query is covered by retrieved chunks.

Important behavior:

- Extracts query keywords.
- Checks whether those keywords appear in context chunks.
- Returns a coverage fraction.

Teacher generation combines coverage and groundedness into user-facing confidence metadata.

## Shared Prompts

| Path | Purpose |
| --- | --- |
| `prompts/teacher_voice.txt` | Shared warm teacher personality prompt. It tells generators how the tutor should sound. |
| `prompts/citation_rules.txt` | Citation and grounding rules. It tells generators to stay inside uploaded files and cite sources where appropriate. |
| `prompts/tone_guidelines.txt` | Additional tone and formatting guidance for educational explanations. |

The prompts are shared so each generator does not drift into a different personality.

## Core Tests

| Path | What it checks |
| --- | --- |
| `tests/test_bm25_generated_questions.py` | BM25 search indexes generated question metadata. |
| `tests/test_hybrid_retriever.py` | Hybrid retrieval behavior and result fusion. |
| `tests/test_llm_runtime.py` | Runtime LLM option resolution. |
| `tests/test_retrieval_evaluation.py` | Retrieval evaluation metrics and summaries. |

## Design Notes

The core package intentionally avoids direct dependencies on:

- FastAPI routers
- SQLAlchemy models
- ARQ worker objects
- MinIO buckets
- Frontend types
- Generator-specific prompts

That boundary is useful. If a generator can import only `teacherlm_core` and its own files, it remains portable and easy to test. If the backend wants to change database layout, the generator contract does not have to change.
