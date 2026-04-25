# mindmap_gen

Generates hierarchical **mind maps** (cartes mentales) from a student's
uploaded course materials. The student gets a bird's-eye view of an
entire subject as an interactive Mermaid diagram, rendered in a
self-contained HTML file with pan/zoom and SVG export.

Port: **8008**

## What it produces

For every `/run` call the generator returns:

- A `response` written in the shared teacher voice
- Two artifacts saved under `./artifacts/` and served from `/artifacts/`:
  - `mindmap_<id>.json` (artifact `type: "mindmap"`) — `{markdown, central_topic,
    main_branches}` payload that the platform's `MindmapRenderer` fetches to
    render the mind map **inline** in chat using Markmap (horizontal tree,
    rounded boxes, soft curves — NotebookLM-style). Each node is **clickable**:
    clicking sends a "Explain X" chat message to the Teacher generator.
  - `mindmap_<id>.html` (artifact `type: "html"`) — standalone Markmap viewer
    for offline viewing or sharing
- `metadata` with `markdown`, `node_count`, `depth`, `central_topic`,
  `main_branches`
- `learner_updates.concepts_covered` — every node label in the map

## Pipeline

1. **Central topic** — short LLM call: "What is the overall subject?"
2. **Theme extraction** — structured Pydantic call returns the main
   branches (mutually-exclusive, collectively-exhaustive)
3. **Hierarchy build** — for each theme, filter chunks by token-overlap
   then ask the LLM (structured) for 2-5 sub-topics with 1-4 leaves each.
   Branches are expanded in parallel via `asyncio.gather`
4. **Balance** — promote single-child chains, cap fan-out to 7,
   trim deepest leaves until total nodes ≤ `max_nodes`
5. **Compile** — Pydantic tree → Mermaid `mindmap` syntax (no LLM
   involved at this stage, so output is always valid)
6. **Render** — Jinja2 template → standalone HTML

## Run standalone

From the repo root:

```bash
pip install -e packages/teacherlm_core
pip install -r generators/mindmap_gen/requirements.txt

PYTHONPATH=generators uvicorn mindmap_gen.app:app --host 0.0.0.0 --port 8008
```

Or with Docker (build context = repo root):

```bash
docker build -f generators/mindmap_gen/Dockerfile -t teacherlm/mindmap_gen:latest .
docker run --rm -p 8008:8008 teacherlm/mindmap_gen:latest
```

Health check: `GET http://localhost:8008/health`
Capabilities: `GET http://localhost:8008/info`

## Enable in the platform

Add (or flip `enabled` to `true` on) the entry in
`generators_registry.json` at the repo root:

```json
{
  "id": "mindmap_gen",
  "name": "Mind Map",
  "type": "api",
  "endpoint": "http://mindmap_gen:8008/run",
  "enabled": true,
  "output_type": "mindmap",
  "icon": "🗺️",
  "description": "Generate a mind map of your course materials",
  "is_chat_default": false
}
```

The platform should request retrieval mode `topic_clusters` for this
generator (see root CLAUDE.md) — it needs broad coverage of every
document topic, not narrow query matching.

## Size options

Pass via `options.size` on the request payload:

| Size            | Main branches | Default `max_nodes` |
| --------------- | ------------- | ------------------- |
| `concise`       | ~4            | 30                  |
| `standard` (default) | ~6       | 60                  |
| `comprehensive` | ~9            | 100                 |

`options.max_nodes` overrides the per-size default if you need a
hard cap (the balancer trims deepest, longest-labelled leaves first).

## Language support

The mind map is produced in the **same language as the source
content**. Both the theme-extraction and subtopic-expansion prompts
explicitly instruct the LLM to mirror the source language, so French
input yields a French map without any extra configuration.

## Configuration

Environment variables (see `config.py`):

| Var              | Default                  | Purpose                          |
| ---------------- | ------------------------ | -------------------------------- |
| `OLLAMA_URL`     | `http://localhost:11434` | Ollama daemon                    |
| `MODEL_NAME`     | `llama3.1:8b`            | Model used for all LLM calls     |
| `ARTIFACTS_DIR`  | `./artifacts`            | Where `.html` and `.mmd` are saved |
| `DEFAULT_SIZE`   | `standard`               | Used when `options.size` absent  |
| `MAX_NODES`      | `60`                     | Reference cap (per-size override applies) |
| `PUBLIC_BASE_URL`| `http://localhost:8008`  | Browser-facing origin used to build artifact URLs |

Place overrides in a `.env` file next to `app.py` or export them in
the environment.

## Why we don't ask the LLM for raw Mermaid

Mind-map quality depends on a balanced, well-organized hierarchy.
Coaxing an LLM to produce nested Mermaid syntax that parses cleanly
is error-prone. Asking it for nested JSON validated by a recursive
Pydantic model is reliable — and once we have a valid tree, the
Mermaid string is deterministic to compile. See
`services/mermaid_compiler.py`.
