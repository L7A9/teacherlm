# teacherlm_core

`teacherlm_core` is the shared package installed by the platform and generator services. It holds the stable generator contract, retrieval primitives, LLM wrappers, shared prompts, and grounding utilities.

## Package Areas

```text
teacherlm_core/
  confidence/
  llm/
  prompts/
  retrieval/
  schemas/
```

## Schemas

The generator I/O contract is defined here and should remain stable.

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
      "chunk_id": "string"
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
  "artifacts": [],
  "sources": [],
  "learner_updates": {
    "concepts_covered": [],
    "concepts_demonstrated": [],
    "concepts_struggled": []
  },
  "metadata": {}
}
```

Artifact records may include fields such as `type`, `url`, `filename`, and `key`.

## LLM Layer

The shared `OllamaClient` wrapper supports:

- Ollama
- OpenAI
- Anthropic
- OpenAI-compatible providers

For Ollama structured output, the wrapper uses the Ollama Python library native `format=` argument. Runtime provider overrides can be passed in generator input options by the platform.

## Retrieval

Core retrieval modules provide:

- Hybrid dense plus sparse retrieval.
- Reciprocal rank fusion.
- Optional cross-encoder reranking through `fastembed`.
- Retrieval mode helpers.
- Evaluation utilities.

Supported retrieval modes:

- `semantic_topk`
- `coverage_broad`
- `narrative_arc`
- `topic_clusters`
- `relationship_dense`

The platform maps output types to these modes and applies output-specific context policies before calling generators.

## Confidence

The confidence utilities compute lightweight groundedness and coverage signals from generated text and retrieved source chunks. Generators can include those signals in output metadata.

## Shared Prompt

`prompts/teacher_voice.txt` contains the shared teacher personality prompt used by teacher-facing outputs. It keeps explanations warm, student-centered, and grounded in course evidence.

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
