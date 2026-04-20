# teacherlm_core

Shared library for the TeacherLM platform and generators. Installed as an editable local dependency by every generator and by the platform backend.

## Modules

- `retrieval/` — hybrid retriever, BM25, cross-encoder reranker, retrieval-mode strategies.
- `llm/` — async Ollama client, streaming helper, structured-output helper.
- `prompts/` — shared teacher personality prompt (`teacher_voice.txt`), tone guidelines, citation rules. All generators prepend `teacher_voice.txt` to their mode-specific system prompts.
- `confidence/` — groundedness and coverage scoring.
- `schemas/` — Pydantic V2 models for the cross-generator contract (`GeneratorInput`, `GeneratorOutput`, `LearnerState`, `Chunk`).

## Install

From the repo root:

```bash
pip install -e packages/teacherlm_core
```

Or, from inside a generator's Dockerfile build context (with the repo root as context):

```dockerfile
COPY packages/teacherlm_core /app/packages/teacherlm_core
RUN pip install -e /app/packages/teacherlm_core
```

## Requirements

- Python 3.14+
- Pydantic V2 (>=2.12)
- No LangChain / LangGraph (incompatible with Pydantic V2 on Python 3.14)
- No `llama-parse` / `llama-cloud-services` (deprecated May 2026)

## Versioning

This package is versioned together with the platform release. Generators pin it via a local editable install — there is no separate PyPI publication.
