# flashcard_gen

Port: **8005** — generates spaced-repetition flashcards from retrieved course chunks and exports them as JSON, CSV, and Anki `.apkg`.

## Pipeline

```
mine concepts (spaCy NER + noun chunks + regex definitions)
  → prioritize by learner_state (boost struggling, drop mastered)
  → generate basic cards (ollama, structured batch)
  + generate cloze cards (pure spaCy, no LLM)
  → dedupe (fastembed cosine ≥ dedupe_similarity)
  → attach SM-2 scheduling metadata
  → export JSON + CSV + Anki .apkg to MinIO
```

## Endpoints

- `GET /health` — liveness
- `GET /info` — capabilities + model config
- `POST /run` — SSE stream; body is `GeneratorInput` from `teacherlm_core`

## Options (via `GeneratorInput.options`)

| key               | default | notes                                       |
| ----------------- | ------- | ------------------------------------------- |
| `card_count`      | 8       | also accepts `count`, `n_cards`             |
| `topic`           | —       | narrows title; retrieval is handled upstream |

## SSE events

`progress` → `{stage, ...}` for each pipeline stage
`token`    → `{delta}` with the teacher intro message
`done`     → full `GeneratorOutput`, including `metadata.deck_data` for client-side rendering without a second fetch

## Dev

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
uvicorn flashcard_gen.app:app --host 0.0.0.0 --port 8005 --reload
```

## Adaptation rules (from CLAUDE.md)

- Prioritize cards covering `learner_state.struggling_concepts`.
- Skip concepts in `learner_state.understood_concepts` with mastery ≥ `mastery_skip_threshold` (default 0.85).
