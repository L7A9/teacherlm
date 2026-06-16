# teacher_gen

`teacher_gen` is the default chat generator for TeacherLM. It answers student questions in a warm teacher voice while staying grounded in course chunks retrieved by the backend.

It is the main conversational teaching experience: explanation, guidance, quick checks, affirmation, refusal when evidence is missing, confidence reporting, and learner-update extraction.

## Service

Default port: `8001`

Endpoints:

- `GET /health`
- `GET /info`
- `POST /run`

`/run` streams server-sent events.

Common events:

- `analysis`: query analysis and selected teacher response mode.
- `sources`: source chunks used by the response.
- `token`: streamed markdown response text.
- `done`: final `GeneratorOutput`.
- `error`: failure details from the shared safe SSE wrapper.

## Generator Info

| Field | Value |
| --- | --- |
| Generator id | `teacher_gen` |
| Output type | `text` |
| Retrieval mode | `semantic_topk` |
| Max context chunks consumed | `TEACHER_GEN_MAX_CONTEXT_CHUNKS`, default `16` |

The backend retrieves and filters context first. `teacher_gen` uses the provided `context_chunks`; it does not query Qdrant, BM25, or the knowledge graph directly.

`GET /info` advertises:

- modes: `explain`, `guide`, `quiz_back`, `affirm`,
- backend-owned context ranking,
- learner-update reporting,
- confidence metadata,
- chat, analysis, and extraction model names.

## Platform Connection

`teacher_gen` connects to the platform through the default chat path:

1. The backend loads `generators_registry.json`.
2. `teacher_gen` is enabled, has `output_type: "text"`, and is marked `is_chat_default: true`.
3. The frontend sends chat turns to `/api/conversations/{conversation_id}/chat`.
4. The backend interaction router may answer simple conversational or obvious outside-files turns directly.
5. For course turns, the backend retrieves `semantic_topk` context with source-file filters applied.
6. The backend builds `GeneratorInput` with `user_message`, `context_chunks`, `learner_state`, `chat_history`, and resolved runtime options.
7. `ApiAdapter` posts that input to `teacher_gen` at `POST /run`.
8. `teacher_gen` streams `analysis`, `sources`, `token`, and final `done` events.
9. The backend persists the answer, sources, confidence metadata, and learner updates.

The generator never writes learner state itself. It only reports `LearnerUpdates`; the backend decides how to merge them.

## Why `semantic_topk`

Teacher chat usually starts with a focused student question: "Explain SVD", "Why is this formula used?", "Compare X and Y", or "I do not understand this part."

`semantic_topk` fits that interaction because the backend:

- retrieves the closest evidence with dense semantic search,
- also includes BM25 exact-term candidates,
- fuses dense and BM25 results with RRF,
- adds graph-relevant candidates for concept relationships,
- reranks the candidate set,
- expands final chunks with neighbors and section summaries.

That gives the teacher a compact, relevant evidence pack while preserving enough local context to teach clearly.

## Pipeline

### 1. Receive `GeneratorInput`

The platform sends:

- `conversation_id`,
- `user_message`,
- retrieved `context_chunks`,
- current `learner_state`,
- recent `chat_history`,
- runtime `options`.

Options may include runtime LLM provider overrides or a forced language. The generator applies those through shared `teacherlm_core.llm.runtime` helpers.

Relevant options:

| Option | Effect |
| --- | --- |
| `language` | Forces the response language through shared language context. |
| `llm.enabled` | Enables runtime provider override. |
| `llm.provider` | `ollama`, `openai`, `anthropic`, or `openai_compatible`. |
| `llm.model` | Model name for overridden provider calls. |
| `llm.base_url` / `llm.api_base_url` | Provider endpoint override. |
| `llm.api_key` | Provider API key, resolved by backend runtime settings. |

### 2. Analyze The Student Turn

`services/query_analyzer.py` classifies the message into structured `QueryAnalysis`.

It looks at:

- intent,
- confusion level,
- whether the student is asking for help,
- whether the student appears correct,
- target concept when available,
- recent chat history,
- learner state.

It uses structured output with the configured analysis model. If analysis fails, a local fallback heuristic still produces a usable analysis.

Why this step exists:

- the same question can require explanation, Socratic guidance, affirmation, or a quick check,
- confusion needs a different teacher behavior than confident review,
- learner state should influence tone and next step.

### 3. Select Response Mode

`services/response_mode.py` maps analysis plus learner state into one of four modes:

| Mode | When used | Teacher behavior |
| --- | --- | --- |
| `explain` | Normal concept question | Clear grounded explanation with sources |
| `guide` | Confusion or stuck turns | Socratic, step-by-step guidance |
| `quiz_back` | Student asks to practice or check understanding | Short check-back question |
| `affirm` | Student appears correct | Encouragement plus next learning step |

Each mode has its own prompt:

- `mode_explain.txt`
- `mode_guide.txt`
- `mode_quiz_back.txt`
- `mode_affirm.txt`

The prompts compose with the shared `teacher_voice.txt` and tone guidelines from `teacherlm_core`.

### 4. Emit Analysis And Sources

The generator streams:

1. `analysis` with intent, confusion, target concept, and selected mode.
2. `sources` with chunk text, source, score, and chunk ID.

The backend/front-end can show sources while the answer streams.

### 5. Check Evidence Strength

The generator receives backend-ranked chunks and performs a final safety check.

It considers:

- top retrieval score,
- whether the user message has evidence terms in the chunks,
- whether the request is a course overview,
- configured `TEACHER_GEN_MIN_RELEVANCE_SCORE`.

If the retrieved evidence is too weak, it refuses politely and redirects the student back to uploaded course material.

Why this exists:

- TeacherLM must stay grounded in uploaded files,
- retrieval can occasionally return weak context,
- a clear refusal is better than inventing unsupported course content.

### 6. Formula Fast Path

If the student asks a formula-only question and formula snippets are present, the generator builds a deterministic formula-card response instead of asking the LLM to improvise.

It extracts:

- formula lines,
- source headings,
- symbol definitions when available,
- concise explanation text.

Why this exists:

- math answers need precision,
- formulas are easier to verify when pulled directly from evidence,
- deterministic responses reduce hallucination for symbolic content.

### 7. Course Overview Fast Path

If the student asks what the course is about or where to start, the generator can build a structured overview from course outline/module/section context.

It extracts:

- likely course title,
- major topics,
- key concepts,
- formulas when relevant,
- source labels.

Why this exists:

- broad orientation is common at the start of study,
- a normal top-k answer can feel arbitrary,
- course-overview context lets the teacher give a roadmap.

### 8. Stream The Teacher Answer

For normal turns, `services/llm_service.py` streams markdown from the chat model.

The system prompt includes:

- shared teacher voice,
- tone guidelines,
- selected mode prompt,
- formatted retrieved chunks,
- understood concepts,
- struggling concepts,
- current user message.

If a configured cloud provider fails in a recoverable way, the service can fall back to the local Ollama chat model and records fallback metadata in the final output.

### 9. Score Confidence

After streaming completes, `services/confidence_scorer.py` computes:

- groundedness: response vocabulary overlap with retrieved chunks,
- coverage: query keyword coverage across chunks,
- overall score: `0.7 * groundedness + 0.3 * coverage`,
- confidence label.

Why this exists:

- it gives the platform and student a transparent signal,
- it catches answers that drift away from evidence,
- it is lightweight enough to run on every teacher response.

### 10. Extract Learner Updates

`services/learner_analyzer.py` asks the extraction model to return concepts:

- covered by the answer,
- demonstrated by the student,
- struggled with by the student.

The backend merges these into learner state after the `done` event.

Why this exists:

- chat is not just Q&A; it updates the student's learning profile,
- future guidance, quizzes, review tests, and remediation can adapt.

## Output

The final `done` event contains:

- markdown response,
- `generator_id: teacher_gen`,
- `output_type: text`,
- no artifacts,
- source chunks,
- learner updates,
- metadata with mode, analysis, confidence, backend context ranker, and fallback information.

## Technology Choices

| Technology | Why it is used |
| --- | --- |
| FastAPI | Small independent HTTP service with `/health`, `/info`, and `/run` |
| SSE | Token streaming keeps chat responsive |
| Pydantic V2 | Validates structured analysis and final contract models |
| `teacherlm_core` | Shared schemas, prompts, LLM wrapper, confidence scoring |
| Ollama native `format=` | Structured local output without extra orchestration frameworks |
| Runtime provider overrides | Lets platform settings route LLM calls to Ollama, OpenAI, Anthropic, or compatible APIs |
| Backend-owned RAG | Keeps retrieval, filtering, reranking, and graph search consistent across generators |
| Deterministic formula and overview fast paths | Avoids unnecessary LLM improvisation for high-risk formulas and broad orientation turns |
| Local fallback heuristics | Keeps query analysis useful even when structured model calls fail |

## Environment

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `TEACHER_GEN_OLLAMA_HOST` | Ollama base URL override |
| `TEACHER_GEN_CHAT_MODEL` | model for final streamed responses |
| `TEACHER_GEN_ANALYSIS_MODEL` | model for query analysis |
| `TEACHER_GEN_EXTRACTION_MODEL` | model for learner-update extraction |
| `TEACHER_GEN_CHAT_TEMPERATURE` | response-generation temperature |
| `TEACHER_GEN_ANALYSIS_TEMPERATURE` | query-analysis temperature |
| `TEACHER_GEN_EXTRACTION_TEMPERATURE` | learner-update extraction temperature |
| `TEACHER_GEN_MAX_CONTEXT_CHUNKS` | maximum chunks consumed by the generator |
| `TEACHER_GEN_MIN_RELEVANCE_SCORE` | off-topic refusal threshold |
| `TEACHER_GEN_CONFUSION_GUIDE_THRESHOLD` | threshold for stronger guidance behavior |
| `TEACHER_GEN_STUCK_TURNS_THRESHOLD` | turns without progress before guide behavior is favored |
| `TEACHER_GEN_REQUEST_TIMEOUT_S` | request timeout setting |
| `OLLAMA_HOST` | shared Ollama fallback host |
| `OLLAMA_CHAT_MODEL` | shared chat model fallback |
| `OLLAMA_ANALYSIS_MODEL` | shared analysis model fallback |
| `OLLAMA_EXTRACTION_MODEL` | shared extraction model fallback |

## Docker Notes

The Dockerfile:

- builds from the repository root,
- installs Python `3.14-slim`,
- installs `teacherlm_core`,
- installs only the generator's runtime HTTP/LLM dependencies,
- exposes port `8001`,
- adds a `/health` healthcheck,
- can optionally pre-download the fastembed reranker if `TEACHER_GEN_PRELOAD_RERANKER=true`.

Reranking is currently owned by the backend; the optional image predownload is only a cache/warmup convenience.

## Local Run

From this directory:

```bash
pip install -e ../../packages/teacherlm_core
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8001
```

Through Docker Compose:

```bash
cd ../../platform
docker compose up -d teacher_gen
```

## Tests

From the repository root:

```bash
pytest generators/teacher_gen/tests
```
