# mindmap_gen

`mindmap_gen` creates interactive course mind maps from uploaded material. It emits Markmap-compatible markdown plus JSON and HTML artifacts.

## Service

Default port: `8008`

Endpoints:

- `GET /health`
- `GET /info`
- `POST /run`
- `GET /artifacts/...`

`/run` streams server-sent events. `/artifacts` serves generated mind map files with permissive CORS for the frontend.

Common events:

- `progress`: outline, refinement, artifact, and completion stages.
- `artifact`: JSON or HTML artifact metadata.
- `done`: final generator output.
- `error`: failure details.

## Generator Info

| Field | Value |
| --- | --- |
| Generator id | `mindmap_gen` |
| Output type | `mindmap` |
| Retrieval mode | `topic_clusters` |

The backend supplies topic-clustered and course-structure context. Source-file selection is applied before the generator receives context.

## Options

Supported request options include:

| Option | Notes |
| --- | --- |
| `size` | `concise`, `standard`, or `comprehensive`; default `standard` |
| `max_nodes` | Optional explicit node cap |
| `llm_refine` | Enable LLM outline refinement; frontend default is `true` |
| `force_regenerate` | Ignore cached generation paths |
| `regenerate` | Alias-style regenerate flag |
| `generation_id` | Optional stable generation id |
| `language` | Optional output language |

Size presets:

| Size | Main branches | Node cap |
| --- | ---: | ---: |
| `concise` | 4 | 30 |
| `standard` | 6 | 110 |
| `comprehensive` | 9 | 150 |

The frontend currently starts standard mind maps with `max_nodes: 110`.

## Pipeline

The generator can build a mind map from several levels of context:

1. Module packs when available.
2. LLM-refined batch outlines.
3. Synthesized course outline.
4. Course structure.
5. Section summaries and key concepts.
6. Older theme hierarchy fallback.

After outline creation it balances the tree, compiles Markmap markdown, writes JSON and HTML artifacts, and returns metadata for the frontend renderer.

## Output

The final response includes:

- Markdown summary.
- Markmap markdown in metadata.
- Node count and depth.
- Central topic and main branches.
- JSON and HTML artifacts.
- Learner updates for covered nodes.

The frontend can render the mind map directly and also open generated artifacts.

## Environment

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `MINDMAP_GEN_OLLAMA_URL` | Ollama base URL override |
| `MINDMAP_GEN_MODEL` | mind map planning model |
| `MINDMAP_GEN_ARTIFACTS_DIR` | local artifact directory |
| `MINDMAP_GEN_PUBLIC_BASE_URL` | public artifact base URL |
| `MINDMAP_GEN_DEFAULT_SIZE` | default size preset |
| `MINDMAP_GEN_MAX_NODES` | default node cap if not overridden |

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
