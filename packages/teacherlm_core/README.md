# teacherlm_core

`teacherlm_core` is the shared local package installed by the platform and every implemented generator. It holds the stable generator contract, shared learner-state schemas, LLM wrappers, structured-output helpers, retrieval primitives, confidence utilities, and teacher voice prompts.

The package is intentionally small and framework-free. It avoids orchestration frameworks and keeps the project compatible with Python 3.14+, Pydantic V2, FastAPI, fastembed, and the Ollama Python client native `format=` API.

## Package Areas

```text
teacherlm_core/
  confidence/
  llm/
  prompts/
  retrieval/
  schemas/
```

| Area | Purpose |
| --- | --- |
| `schemas/` | Stable `GeneratorInput`, `GeneratorOutput`, `LearnerState`, and `Chunk` models |
| `llm/` | Async LLM wrapper, provider adapters, native structured output, streaming helpers |
| `retrieval/` | BM25, hybrid dense+sparse retrieval, RRF fusion, reranker, retrieval modes, eval metrics |
| `confidence/` | Lightweight groundedness and query coverage scoring |
| `prompts/` | Shared teacher personality, tone, and citation guidance |

## Dependencies And Why

| Dependency | Used for |
| --- | --- |
| Pydantic V2 | Stable schemas and structured output validation. |
| pydantic-settings | Shared environment settings with `TEACHERLM_` prefix. |
| ollama | Local chat and native `format=` structured output. |
| httpx | OpenAI-compatible and Anthropic provider calls. |
| qdrant-client | Dense vector retrieval primitives. |
| fastembed | Text embeddings and cross-encoder reranking. |
| rank-bm25 | Transparent lexical search. |
| numpy | Clustering and vector math in retrieval modes. |

`teacherlm_core` deliberately does not depend on LangChain, LangGraph, llama-parse, or sentence-transformers.

## What Core Owns And Does Not Own

Core owns:

- immutable generator input/output schemas,
- learner-state schemas shared across services,
- generic LLM and structured-output helpers,
- language and runtime LLM option context,
- retrieval primitives,
- confidence utilities,
- shared prompts.

Core does not own:

- upload parsing,
- database records,
- source-file selection,
- production retrieval policy,
- generator service routing,
- artifact persistence.

Those live in the platform or individual generators.

## Stable Generator Contract

The generator I/O contract is immutable for all generator services. The platform prepares the input, calls a generator service, and persists the output.

Input:

```json
{
  "conversation_id": "string",
  "user_message": "string",
  "context_chunks": [
    {
      "text": "string",
      "source": "string",
      "score": 0.0,
      "chunk_id": "string",
      "metadata": {}
    }
  ],
  "learner_state": {
    "conversation_id": "string",
    "understood_concepts": [],
    "struggling_concepts": [],
    "mastery_scores": {},
    "session_turns": 0,
    "turns_since_progress": 0,
    "known_concepts": [],
    "concept_progress": [],
    "learning_phases": [],
    "objective_progress": [],
    "phase_progress": [],
    "remediation_paths": []
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

Output:

```json
{
  "response": "markdown string",
  "generator_id": "string",
  "output_type": "text",
  "artifacts": [
    {
      "type": "quiz",
      "url": "https://...",
      "filename": "quiz.json",
      "key": "optional-storage-key"
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

Why this contract exists:

- Generators can run as independent HTTP services.
- The platform can own retrieval, learner state, source-file filtering, persistence, and dispatch.
- New output types can be added without changing chat history, learner tracking, or artifact handling.
- Frontend renderers can use `output_type`, `artifacts`, and `metadata` consistently.

## Chunk Schema

`Chunk` is the common evidence unit:

```json
{
  "text": "course excerpt or expanded context",
  "source": "lecture.pdf",
  "score": 0.82,
  "chunk_id": "stable-id",
  "metadata": {}
}
```

The platform enriches metadata with values such as:

- `document_id`
- `section_id`
- `source_file_id`
- `heading_path`
- `chunk_index`
- `prev_chunk_id`
- `next_chunk_id`
- `key_concepts`
- `section_summary`
- `context_type`
- formula/table/timeline counts

Generators should treat chunk text as the primary evidence and metadata as citation, rendering, planning, or context-policy support.

## Learner State

`LearnerState` carries both simple and canonical progress fields:

- `understood_concepts`
- `struggling_concepts`
- `mastery_scores`
- `known_concepts`
- `concept_progress`
- `learning_phases`
- `objective_progress`
- `phase_progress`
- `remediation_paths`

This lets generator services adapt without needing direct database access. For example:

- `teacher_gen` uses it to choose explain, guide, quiz-back, or affirm modes.
- `quiz_gen` uses it to bias questions toward struggling concepts and stretch understood concepts.
- review and remediation features use canonical concept/objective/phase progress.

Generators report changes through `LearnerUpdates`; the backend decides how to merge those updates.

## LLM Layer

The class name `OllamaClient` is kept for compatibility, but it supports multiple provider types:

- Ollama
- OpenAI
- Anthropic
- OpenAI-compatible APIs

For Ollama structured output, `OllamaClient.chat_structured()` passes the Pydantic schema through the Ollama Python library native `format=` argument. This is important for Python 3.14 compatibility and avoids extra orchestration frameworks.

For OpenAI-compatible providers, schemas are translated to `response_format` JSON schema. For Anthropic, the schema is injected into the system instructions because Anthropic does not expose the same response-format shape.

Runtime provider overrides can be passed in `GeneratorInput.options["llm"]` by the platform settings layer. Generators can therefore use configured cloud providers without changing their own call sites.

## Runtime LLM Options

`llm/runtime.py` stores per-request LLM overrides in a `ContextVar`.

Each generator starts by calling:

```python
set_current_llm_options(options)
```

Then service-level clients call `build_llm_client_kwargs()` to resolve:

- provider,
- base URL,
- model,
- API key.

If no enabled override is present, the generator falls back to its configured Ollama host and model. Supported provider names are:

- `ollama`,
- `openai`,
- `anthropic`,
- `openai_compatible`.

This lets the frontend settings page affect all generator LLM calls without changing every prompt call site.

## Forced Language Context

`llm/language.py` stores a forced language in a `ContextVar`.

Generators call:

```python
token = set_current_language(options.get("language"))
```

`OllamaClient` then injects a strict language directive into the first system message for every LLM call in that async context. Supported language labels include:

- English US/UK,
- French,
- Spanish,
- Italian,
- Portuguese Brazil,
- German,
- Japanese,
- Mandarin Chinese,
- Hindi.

This is why the frontend can set one language preference and have teacher chat, quiz intros, podcast scripts, and mind map labels follow it consistently.

## Structured Output Helper

`llm/structured.py` exposes `generate_structured()`.

It:

1. calls the configured client with a Pydantic schema,
2. validates the returned JSON into that schema,
3. retries on validation failures,
4. sends the validation error back to the model as a repair prompt,
5. raises a clear runtime error if all attempts fail.

Why this is used:

- quiz questions, narrative arcs, podcast sections, and mind map outlines must have predictable shapes,
- Pydantic V2 validation catches missing fields and wrong types,
- retrying gives local models a chance to repair malformed JSON,
- generators can keep business logic separate from JSON cleanup.

## SSE Helpers

`llm/streaming.py` provides shared SSE helpers:

- `format_sse()` and `_format_sse()` format event/data frames,
- `stream_as_sse()` wraps async text chunks as SSE,
- `safe_sse_stream()` catches generator exceptions and emits a final `error` event with a friendly message.

All implemented generators use `safe_sse_stream()` at their FastAPI `/run` endpoint, so a pipeline exception becomes an SSE `error` event instead of a broken HTTP response.

## Retrieval Primitives

The platform owns production retrieval, but the reusable retrieval building blocks live here.

### BM25

`retrieval/bm25.py` wraps `rank-bm25`.

It tokenizes lowercased word characters and searches over:

- chunk text,
- `heading_path`,
- `section_title`,
- `key_concepts`,
- `generated_questions`.

BM25 is used because exact terms matter in courses. Acronyms, formulas, command names, theorem names, and field-specific vocabulary are often better captured by lexical matching than by dense embeddings alone.

### Hybrid Retrieval

`retrieval/hybrid_retriever.py` combines:

- dense semantic search through Qdrant,
- sparse BM25 search over the same allowed chunk set,
- reciprocal rank fusion with `RRF_K = 60`,
- deduplication by `chunk_id`,
- fused scores written back into returned `Chunk.score`.

Dense retrieval finds meaning. BM25 preserves exact terms. RRF rewards candidates that rank well in either list, especially those that appear in both.

### Reranker

`retrieval/reranker.py` wraps fastembed `TextCrossEncoder`, defaulting to:

```text
BAAI/bge-reranker-base
```

The reranker scores `(query, document)` pairs and returns the top chunks. It is slower than candidate retrieval, so it is used after dense/BM25/RRF have narrowed the pool.

### Retrieval Modes

`retrieval/retrieval_modes.py` provides mode-level helpers:

| Mode | Technique | Product fit |
| --- | --- | --- |
| `semantic_topk` | Hybrid top-k | Chat answers need the closest evidence |
| `coverage_broad` | MMR-like token diversity over a hybrid pool | Quizzes need representative breadth |
| `narrative_arc` | Intro-like chunks, query-relevant middle, conclusion-like chunks | Podcasts/reports need teachable flow |
| `topic_clusters` | Embedding clustering and representative chunks | Mind maps/presentations need topic coverage |
| `relationship_dense` | Entity/verb density heuristic over retrieved chunks | Diagrams need relationships, processes, and facts |

The backend wraps these modes with course-aware policies, graph search, reranking, and context expansion.

## Confidence Utilities

The confidence package provides lightweight signals:

- `score_groundedness(response, chunks)`: mean per-sentence overlap between response content words and retrieved chunk vocabulary.
- `score_coverage(query, chunks)`: fraction of query keywords present in the retrieved context.

`teacher_gen` combines these as:

```text
overall = 0.7 * groundedness + 0.3 * coverage
```

These scores are not a full factuality proof. They are cheap, transparent guardrails that help the teacher report whether the answer stayed close to available course evidence.

## Shared Prompts

`prompts/teacher_voice.txt` is the shared teacher personality prompt. It keeps outputs:

- warm,
- encouraging,
- student-centered,
- evidence-grounded,
- honest when evidence is missing.

Generator-specific prompts live inside each generator, but they compose with the shared teacher voice where appropriate.

## Evaluation Utilities

`retrieval/evaluation.py` supports retrieval evaluation cases with metrics such as:

- hit rate at K,
- precision at K,
- recall at K,
- MRR,
- nDCG,
- section recall,
- citation precision,
- source-document hit,
- latency.

The backend scripts in `platform/backend/scripts` use these utilities to evaluate retrieval and course-context policies.

## Settings

`CoreSettings` reads environment variables with the `TEACHERLM_` prefix.

Important defaults:

| Setting | Default |
| --- | --- |
| `TEACHERLM_OLLAMA_BASE_URL` | `http://localhost:11434` |
| `TEACHERLM_OLLAMA_MODEL` | `llama3.1:8b` |
| `TEACHERLM_EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` |
| `TEACHERLM_RERANKER_MODEL` | `BAAI/bge-reranker-base` |
| `TEACHERLM_RETRIEVAL_TOP_K` | `10` |
| `TEACHERLM_BM25_TOP_K` | `20` |
| `TEACHERLM_DENSE_TOP_K` | `20` |

The platform has its own richer settings object for production retrieval and storage. Core settings are the shared-library defaults.

## Development

Install in editable mode from this package directory:

```bash
pip install -e .
```

Run shared-core tests from the repository root:

```bash
pytest packages/teacherlm_core/tests
```

Project compatibility rules:

- Python `3.14+`.
- Pydantic V2 only.
- No LangChain or LangGraph.
- No deprecated LlamaParse packages.
- Prefer `fastembed` where embeddings or reranking are needed.
