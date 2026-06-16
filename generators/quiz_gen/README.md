# quiz_gen

`quiz_gen` creates grounded quizzes from retrieved course context. It is designed for broad coverage across the selected ready source files, with learner-aware planning and strict validation before the quiz reaches the student.

The generator prefers fewer high-quality grounded questions over padding the quiz with generic filler.

## Service

Default port: `8002`

Endpoints:

- `GET /health`
- `GET /info`
- `POST /run`

`/run` streams server-sent events.

Common events:

- `progress`: concept extraction, planning, generation, validation, artifact stages.
- `token`: final short teacher-facing response text.
- `done`: final `GeneratorOutput`.
- `error`: failure details from the shared safe SSE wrapper.

Quiz artifact metadata is included in the final `done` payload when storage succeeds.

## Generator Info

| Field | Value |
| --- | --- |
| Generator id | `quiz_gen` |
| Output type | `quiz` |
| Retrieval mode | `coverage_broad` |
| Supported question kinds | `mcq`, `true_false` |
| Bloom levels | `remember`, `understand`, `apply`, `analyze` |

The backend retrieves context first. If the student selected source files in the frontend, retrieval is filtered before `quiz_gen` receives chunks.

`GET /info` advertises:

- question kinds: `mcq`, `true_false`,
- Bloom levels,
- learner-state adaptation,
- optional distractor engine support,
- chat, extraction, and generation model names,
- distractor embedding model.

## Platform Connection

`quiz_gen` connects through the generic generation route:

1. The frontend opens the generator dialog for output type `quiz`.
2. The dialog sends `POST /api/conversations/{conversation_id}/generate` with `output_type: "quiz"`, options, and selected `source_file_ids`.
3. The backend resolves the enabled `quiz_gen` registry entry.
4. For no-topic UI requests, the backend uses the broad quiz context policy: course outline, representative sections, equations, and tables.
5. If a topic is supplied through an API caller, the backend uses topic sections plus retrieved hits.
6. The backend builds `GeneratorInput` and posts it to `POST /run`.
7. `quiz_gen` streams `progress`, `token`, and `done`.
8. The backend persists the final response, quiz artifact metadata, source chunks, and covered concepts.

The generator receives already-grounded evidence. It does not parse files, query Qdrant, or mutate learner records.

## Why `coverage_broad`

Quizzes are assessment tools, not narrow Q&A answers. A useful quiz should test representative course coverage, especially when the student clicks "Generate quiz" without a topic.

`coverage_broad` fits because the backend:

- avoids using only the nearest paragraph,
- samples across the selected course material,
- includes course outline and representative sections for no-topic quizzes,
- adds equations and tables as structured context,
- preserves source-file filtering.

This gives the quiz planner enough variety to test more than one local snippet.

## Options

Supported request options include:

| Option | Notes |
| --- | --- |
| `question_count` | Default `8`, minimum `3`, maximum `30` |
| `n_questions` | Alias for `question_count` |
| `count` | Alias for `question_count` |
| `question_types` | Strict question kind selector: `mcq` or `true_false` |
| `types` | Alias for `question_types` |
| `kinds` | Alias for `question_types` |
| `language` | Optional output language, normally supplied by platform settings |

Frontend aliases are normalized:

- `multiple_choice` becomes `mcq`.

Implementation detail: if multiple `question_types` are supplied, the current pipeline keeps only the first recognized kind. This matches the current frontend, which exposes a single quiz-type selector.

The current frontend intentionally does not expose a topic field for quiz generation, so the default UI path produces broad course quizzes over selected source files.

`fill_blank` exists in internal schemas and prompts for future expansion, but `/info` and the current UI expose only MCQ and true/false.

## Pipeline

### 1. Receive `GeneratorInput`

The platform sends:

- broad or topic-filtered `context_chunks`,
- current `learner_state`,
- runtime `options`,
- recent chat history.

The generator applies runtime LLM options and language settings through the shared LLM runtime helpers.

### 2. Resolve Quiz Shape

`pipeline.py` resolves:

- target question count,
- allowed question kind(s),
- title,
- language settings.

Question count is clamped between configured minimum and maximum values. Unsupported question kinds are ignored rather than passed through to generation.

### 3. Extract Testable Concepts

`services/concept_extractor.py` extracts concepts grouped by Bloom level using structured LLM output.

It returns `ExtractedConcepts` with lists for:

- `remember`,
- `understand`,
- `apply`,
- `analyze`.

Concept extraction receives formatted source chunks with chunk IDs, source labels, headings, and text. The extractor cleans results by:

- dropping boilerplate metadata,
- rejecting author/institution-style labels,
- keeping only source chunk IDs that exist,
- falling back to chunk metadata such as `key_concepts` if the LLM extraction fails or returns nothing useful.

Why this step exists:

- quizzes should test concepts, not arbitrary sentences,
- Bloom levels let the quiz mix recall, understanding, application, and analysis,
- source chunk IDs keep every question connected to evidence.

### 4. Plan The Question Mix

`services/difficulty_adapter.py` builds a `QuizPlan`.

It uses:

- extracted concepts,
- learner struggling concepts,
- learner understood concepts,
- configured mix ratios,
- allowed question kinds,
- Bloom levels.

Default planning intent:

- emphasize struggling concepts,
- include general coverage concepts,
- stretch understood concepts with harder Bloom levels,
- rotate across concepts so the quiz is not repetitive.

Why this step exists:

- an AI teacher should adapt assessment to the learner,
- struggling concepts deserve more practice,
- understood concepts can be tested at a higher level instead of repeated as easy recall.

### 5. Generate Structured Questions

`services/question_generator.py` generates each question slot independently.

For each slot:

1. choose the best source chunk for the concept,
2. select the prompt for `mcq` or `true_false`,
3. call the generation model through `generate_structured()`,
4. validate the returned Pydantic question model,
5. normalize concept, Bloom level, and source chunk ID back to the planned slot.

Why per-slot generation:

- failures are isolated,
- ordering is predictable,
- the model has a small focused task,
- each question remains grounded to a chosen source chunk.

After generation, the service forces the concept, Bloom level, and source chunk ID back to the planned slot. This keeps analytics and citations honest when the model returns valid JSON but drifts on metadata.

### 6. Optional Distractor Enhancement

`services/distractor_engine.py` can improve MCQ distractors with fastembed similarity.

It:

- extracts candidate phrases from source chunks,
- embeds candidates and the correct answer,
- selects semantically plausible but not-too-similar distractors,
- keeps the original LLM options if not enough good distractors are found.

This is controlled by:

```text
QUIZ_GEN_ENHANCE_DISTRACTORS
```

It is off by default because phrase pools can contain fragments that degrade otherwise good LLM choices.

Why this exists:

- good distractors should be plausible,
- semantic hard negatives can make MCQs more educational,
- the generator should not degrade quality when the candidate pool is weak.

### 7. Validate And Deduplicate

`services/quality_validator.py` removes weak questions.

Validation checks include:

- MCQs have enough options,
- exactly one correct answer index is valid,
- true/false answers are boolean,
- fill-blank schema support exists internally but current exposed kinds are MCQ and true/false,
- questions are not generic source-aware prompts such as "which source says...",
- answers are not obviously shown in the question,
- ambiguous list-style MCQs are filtered,
- duplicates are removed by concept, chunk, and wording signature.
- if too few model questions survive, deterministic grounded top-up questions are attempted from source sentences.

Why validation exists:

- local models can drift even with structured output,
- students should not see giveaway or self-referential questions,
- grounded quality matters more than reaching a requested count.

### 8. Build Teacher Intro

The chat model writes a short teacher-style intro using:

- learner struggling concepts,
- learner understood concepts,
- plan summary,
- Bloom distribution.

This keeps the quiz experience encouraging and contextual instead of feeling like a raw data export.

### 9. Store Quiz Artifact

The final `QuizOutput` is serialized to JSON and uploaded through the MinIO artifact store.

Artifact shape:

```json
{
  "type": "quiz",
  "url": "...",
  "filename": "quiz.json",
  "key": "conversations/.../artifacts/..."
}
```

If artifact upload fails, generation still succeeds and the failure is streamed as a `progress` event.

Why this exists:

- the frontend renders quiz JSON interactively,
- artifact URLs can be re-signed later using stored keys,
- storage outages should not erase an otherwise valid quiz.

### 10. Return Learner Updates

The generator reports `concepts_covered` from the kept questions. The backend merges those updates into learner state after the `done` event.

## Output

The final output includes:

- a short markdown response,
- `metadata.quiz_data`,
- `metadata.plan`,
- `metadata.bloom_distribution`,
- `metadata.dropped_questions`,
- `metadata.top_up_questions`,
- quiz artifact metadata when upload succeeds,
- all source chunks used as generator context,
- learner updates for covered concepts.

## Technology Choices

| Technology | Why it is used |
| --- | --- |
| FastAPI | Independent generator service with stable health/info/run endpoints |
| SSE | Streams progress so long quiz generation feels alive |
| Pydantic V2 | Validates question schemas, quiz plans, and final output |
| Ollama native `format=` | Structured local generation without orchestration frameworks |
| `teacherlm_core` | Shared generator contract, learner state, prompts, LLM runtime |
| fastembed | Optional semantic distractor selection |
| MinIO | Stores quiz JSON artifacts for frontend rendering/history |
| Backend-owned `coverage_broad` retrieval | Keeps quiz context broad, filtered, and consistent |
| Per-slot structured generation | Isolates failures and keeps each question tied to one source chunk |
| Deterministic top-up fallbacks | Preserve grounded output when LLM generation is thin |

## Environment

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `QUIZ_GEN_OLLAMA_HOST` | Ollama base URL override |
| `QUIZ_GEN_CHAT_MODEL` | model for teacher intro |
| `QUIZ_GEN_EXTRACTION_MODEL` | model for concept extraction |
| `QUIZ_GEN_GENERATION_MODEL` | model for question generation |
| `QUIZ_GEN_CHAT_TEMPERATURE` | teacher-intro temperature |
| `QUIZ_GEN_EXTRACTION_TEMPERATURE` | concept extraction temperature |
| `QUIZ_GEN_GENERATION_TEMPERATURE` | question generation temperature |
| `QUIZ_GEN_DEFAULT_QUESTION_COUNT` | default number of questions |
| `QUIZ_GEN_MIN_QUESTION_COUNT` | lower bound |
| `QUIZ_GEN_MAX_QUESTION_COUNT` | upper bound |
| `QUIZ_GEN_MIX_STRUGGLING` | learner-struggle planning ratio |
| `QUIZ_GEN_MIX_COVERAGE` | broad-coverage planning ratio |
| `QUIZ_GEN_MIX_STRETCH` | harder/stretch planning ratio |
| `QUIZ_GEN_ENHANCE_DISTRACTORS` | enable optional fastembed distractor pass |
| `QUIZ_GEN_EMBEDDING_MODEL` | distractor embedding model |
| `QUIZ_GEN_DISTRACTOR_SIM_MIN` | minimum cosine similarity for optional hard negatives |
| `QUIZ_GEN_DISTRACTOR_SIM_MAX` | maximum cosine similarity for optional hard negatives |
| `QUIZ_GEN_DISTRACTOR_POOL_SIZE` | phrase candidate pool size for optional distractors |
| `QUIZ_GEN_DISTRACTORS_PER_MCQ` | number of optional semantic distractors per MCQ |
| `QUIZ_GEN_MINIO_*` | artifact storage configuration |
| `QUIZ_GEN_ARTIFACT_URL_TTL_S` | presigned artifact URL lifetime |
| `QUIZ_GEN_REQUEST_TIMEOUT_S` | request timeout setting |
| `OLLAMA_HOST` | shared Ollama fallback host |
| `OLLAMA_CHAT_MODEL` | shared model fallback |

## Docker Notes

The Dockerfile:

- builds from the repository root,
- installs `teacherlm_core`,
- installs FastAPI, Ollama, Pydantic V2, SSE, httpx, fastembed, MinIO, and NumPy,
- pre-downloads `BAAI/bge-small-en-v1.5` so the optional distractor path does not stall on first use,
- exposes port `8002`,
- adds a `/health` healthcheck.

## Local Run

From this directory:

```bash
pip install -e ../../packages/teacherlm_core
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8002
```

Through Docker Compose:

```bash
cd ../../platform
docker compose up -d quiz_gen
```

## Tests

From the repository root:

```bash
pytest generators/quiz_gen/tests
```
