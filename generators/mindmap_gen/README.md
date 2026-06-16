# mindmap_gen

`mindmap_gen` creates interactive course mind maps from uploaded materials. It emits Markmap-compatible markdown plus JSON and standalone HTML artifacts.

The generator is built for overview learning: students use it to see the whole subject, major modules, subtopics, details, and relationships before diving into chat or quizzes.

## Service

Default port: `8008`

Endpoints:

- `GET /health`
- `GET /info`
- `POST /run`
- `GET /artifacts/...`

`/run` streams server-sent events. `/artifacts` serves generated mind map files with permissive CORS so the frontend can fetch JSON/HTML directly.

Common events:

- `progress`: outline, fallback, enrichment, balancing, artifact stages.
- `artifact`: JSON or HTML artifact metadata.
- `done`: final `GeneratorOutput`.
- `error`: failure details from the shared safe SSE wrapper.

## Generator Info

| Field | Value |
| --- | --- |
| Generator id | `mindmap_gen` |
| Output type | `mindmap` |
| Retrieval mode | `topic_clusters` |
| Artifact types | `mindmap` JSON and standalone `html` |
| Renderer format | Markmap-compatible markdown |

The backend supplies topic-clustered and course-structure context. Source-file selection is applied before the generator receives context.

`GET /info` advertises:

- generator id and output type,
- supported sizes,
- language behavior,
- retrieval mode,
- streaming support,
- generation model.

## Platform Connection

`mindmap_gen` connects through the generic generation route:

1. The frontend requests `output_type: "mindmap"` through `/api/conversations/{conversation_id}/generate`.
2. The backend injects defaults for the current UI path: `llm_refine: true`, `max_nodes: 110`, `size: "standard"`, `force_regenerate: true`, and a fresh `generation_id`.
3. The backend resolves the enabled registry entry and applies selected `source_file_ids`.
4. With no topic, the backend sends mind map course outline plus per-document module packs.
5. With a topic, the backend sends topic context.
6. The backend builds `GeneratorInput` and posts it to `POST /run`.
7. `mindmap_gen` streams progress, emits `artifact` events for JSON/HTML files, then emits `done`.
8. The frontend fetches the JSON artifact directly from `mindmap_gen` under `/artifacts/...` and renders it with Markmap.

The generator writes artifacts to its own artifact directory and serves them over HTTP. It does not store them in MinIO.

## Why `topic_clusters`

Mind maps need breadth and structure. A narrow nearest-neighbor query would over-focus on a single topic and miss the course shape.

`topic_clusters` and the platform mind map context policy provide:

- course outline,
- document sequence,
- per-document module packs,
- section summaries,
- key concepts,
- formulas/tables when present in section facts,
- source-file filtering.

This lets the generator build a bird's-eye view rather than a local explanation.

## Options

Supported request options include:

| Option | Notes |
| --- | --- |
| `size` | `concise`, `standard`, or `comprehensive`; default `standard` |
| `max_nodes` | Optional explicit node cap |
| `llm_refine` | Enable LLM outline refinement; frontend default is `true` |
| `force_regenerate` | Request a fresh generation path/layout |
| `regenerate` | Alias-style regenerate flag |
| `generation_id` | Optional stable generation ID |
| `language` | Optional output language |

Size presets:

| Size | Main branches | Node cap |
| --- | ---: | ---: |
| `concise` | 4 | 30 |
| `standard` | 6 | 110 |
| `comprehensive` | 9 | 150 |

Generator default `MAX_NODES` is `60`, but the frontend currently starts standard mind maps with `max_nodes: 110`.

## Core Design Choice

The generator builds a validated hierarchy first, then compiles it to renderable artifacts.

It does not depend on the LLM writing raw diagram code.

Why:

- LLM-generated diagram syntax is fragile,
- Pydantic can validate a recursive hierarchy,
- the balancer can enforce readability before rendering,
- programmatic markdown/HTML generation gives predictable artifacts,
- the frontend can render the JSON directly and also open a standalone HTML version.

## Pipeline

### 1. Receive `GeneratorInput`

The platform sends:

- `context_chunks`,
- current learner state,
- recent chat history,
- runtime LLM/provider options,
- mind map options.

The generator applies runtime LLM options and language settings through shared `teacherlm_core` helpers.

### 2. Resolve Size And Generation Settings

`pipeline.py` resolves:

- size preset,
- node cap,
- `llm_refine`,
- force-regeneration flags,
- generation ID,
- optional language.

If `force_regenerate` is true, a fresh generation hint is included so repeated mind map generations do not collapse into identical layouts when a non-LLM fallback path is used.

### 3. Module-Pack Fast Path

If the backend supplied `mindmap_module_pack` chunks, `llm_refine` is false, and regeneration is not forced, the generator can build directly from module packs.

Module packs contain:

- document order,
- document title,
- document role (`main` or `supporting`),
- major headings,
- study outline details,
- key concepts,
- formulas/tables/timelines summarized into section facts.

Why this path exists:

- course structure is already known,
- no LLM call is needed,
- it is fast and deterministic,
- it preserves document/module order.

### 4. Batch Outline Extraction

When LLM refinement is enabled, the generator splits course context into outline batches.

For module-pack context, each module can become a batch. Otherwise the generator builds broad text batches from available chunks.

For each batch:

1. stream keepalive `progress` events while the LLM call runs,
2. call `LLMService.build_batch_outline()`,
3. validate the returned `CourseOutline`,
4. collect partial `MindMap` outlines.

Why batching exists:

- whole-course context can exceed comfortable prompt size,
- smaller outline tasks are more reliable on local models,
- partial outlines preserve coverage before synthesis.

### 5. Course Synthesis

If partial outlines exist, `LLMService.synthesize_course_outline()` merges them into one coherent course hierarchy with the requested number of main branches.

Why this exists:

- a mind map should be one map, not one disconnected map per file,
- course-level synthesis removes duplicate headings,
- it can group related modules under cleaner study themes.

### 6. One-Shot Outline Fallback

If batch extraction or synthesis fails, the generator tries a one-shot whole-course outline with `build_course_outline()`.

This is useful when:

- context is small enough,
- module packs are absent,
- batch extraction failed for one file but the whole combined context is still readable.

### 7. Parsed Course Structure Fallback

If LLM outline building fails, `services/course_structure.py` builds a hierarchy directly from parsed heading paths.

Why this exists:

- uploaded courses already contain useful structure,
- a non-LLM fallback is better than failing,
- headings often map well to mind map branches.

### 8. Module-Pack Fallback

If parsed structure is not enough, the generator tries module packs again as a fallback path.

This lets the generator recover from LLM unavailability even when the fast path was skipped because `llm_refine` or regeneration was requested.

### 9. Theme Hierarchy Fallback

The older fallback path remains available:

1. infer a central topic,
2. extract high-level themes,
3. build branches in parallel,
4. expand subtopics from chunks.

`services/hierarchy_builder.py` uses cheap token overlap to pick chunks per theme instead of loading another reranker model.

Why this exists:

- it gives the generator one more recovery route,
- it can produce a useful overview from generic chunk context,
- it avoids total failure when course structure is weak.

### 10. Refine And Enrich

After a draft map exists, the pipeline refines it with source labels and course context.

Enrichment can:

- merge supporting modules into the best matching main branch,
- add distinct source-derived labels,
- attach details to thin branches,
- improve sparse trees,
- preserve a max-node budget.

Why this exists:

- mind maps with only headings are not very useful,
- source chunks often contain details the outline missed,
- enrichment improves educational value while staying grounded.

### 11. Optional Language Adaptation

If a target language is supplied, the generator asks the LLM to translate labels while preserving hierarchy and meaning.

If translation fails, it emits a `language_adapt_fallback` progress event and keeps the existing labels.

### 12. Balance The Tree

`services/balancer.py` makes the map readable.

It:

- promotes singleton chains,
- caps over-wide direct child lists,
- trims leaves until the node cap is respected,
- preserves stronger branches where possible.

Why balancing exists:

- a mind map should be scan-friendly,
- huge flat branches are hard to read,
- node caps keep frontend rendering responsive,
- students need a useful overview, not an exhaustive dump.

### 13. Compile And Render Artifacts

The final hierarchy is compiled to Markmap-compatible markdown with `services/markdown_compiler.py`.

`services/html_renderer.py` writes:

- a JSON payload for inline frontend rendering,
- a standalone HTML file using the Jinja2 template.

The service returns artifact URLs under:

```text
{PUBLIC_BASE_URL}/artifacts/{filename}
```

The JSON artifact is the inline frontend payload. The HTML artifact is a standalone Markmap viewer for opening or sharing outside the chat message.

### 14. Return Learner Updates

The generator reports every node label as `concepts_covered`. The backend merges those into learner state after the `done` event.

## Output

The final response includes:

- teacher-style markdown summary,
- JSON mind map artifact,
- standalone HTML artifact,
- source chunks preview,
- learner updates,
- metadata:
  - `markdown`,
  - `node_count`,
  - `depth`,
  - `central_topic`,
  - `main_branches`,
  - `language`.

## Technology Choices

| Technology | Why it is used |
| --- | --- |
| FastAPI | Independent generator service and static artifact serving |
| SSE | Long LLM/fallback flows need progress and keepalive events |
| Pydantic V2 | Validates recursive `MindMapNode`, `MindMap`, and outline models |
| Ollama native `format=` | Structured local hierarchy generation |
| `teacherlm_core` | Shared generator contract and LLM runtime |
| Jinja2 | Produces standalone HTML from a controlled template |
| Markmap markdown | Fits hierarchical mind maps and frontend rendering |
| Programmatic rendering | Avoids fragile raw diagram syntax from the LLM |
| Backend-owned `topic_clusters`/module context | Provides broad course structure and source filtering |
| Module-pack fast path | Uses platform course structure directly when LLM refinement is disabled |
| Fresh generation hints | Encourages alternate faithful layouts on repeated generations |
| Tree balancing | Keeps large maps readable and within node budgets |

## Environment

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `OLLAMA_URL` | LLM base URL |
| `MINDMAP_GEN_OLLAMA_HOST` | Alternate LLM base URL override |
| `OLLAMA_HOST` | Shared LLM base URL fallback |
| `MODEL_NAME` | Generation model |
| `MINDMAP_GEN_MODEL` | Alternate generation model override |
| `ARTIFACTS_DIR` | artifact output directory |
| `DEFAULT_SIZE` | default size preset |
| `MAX_NODES` | default node cap |
| `LLM_CALL_TIMEOUT_S` | max time for an LLM outline call |
| `LLM_KEEPALIVE_INTERVAL_S` | progress keepalive interval |
| `PUBLIC_BASE_URL` | browser-facing base URL for `/artifacts` |

Docker Compose sets:

- `ARTIFACTS_DIR=/app/generators/mindmap_gen/artifacts`,
- `PUBLIC_BASE_URL=http://localhost:8008` by default,
- a persistent `mindmap_artifacts` volume.

## Docker Notes

The Dockerfile:

- builds from the repository root,
- installs Python `3.14-slim`,
- installs only runtime packages needed by this service,
- installs `teacherlm_core` with `--no-deps` to avoid unused heavy retrieval dependencies,
- creates `/app/generators/mindmap_gen/artifacts`,
- exposes port `8008`,
- adds a `/health` healthcheck.

## Local Run

From this directory:

```bash
pip install -e ../../packages/teacherlm_core
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8008
```

Through Docker Compose:

```bash
cd ../../platform
docker compose up -d mindmap_gen
```

## Tests

From the repository root:

```bash
pytest generators/mindmap_gen/tests
```
