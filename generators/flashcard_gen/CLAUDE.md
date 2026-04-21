# Flashcard Generator — CLAUDE.md

## Port: 8005 | Python 3.14+
## Purpose: Spaced-repetition flashcards + Anki export

## Stack
- FastAPI, ollama, teacherlm_core
- spacy>=3.8 + en_core_web_sm (NER, cloze generation)
- fastembed (dedup)
- genanki (Anki export)

## Module Map
flashcard_gen/
├── CLAUDE.md, app.py, config.py, schemas.py, pipeline.py
├── services/
│   ├── concept_miner.py
│   ├── basic_card_gen.py
│   ├── cloze_card_gen.py
│   ├── deduplicator.py
│   ├── sm2_scheduler.py
│   ├── priority_selector.py       # NEW: uses learner_state
│   └── exporter.py
├── prompts/
│   └── card_generation.txt
├── artifacts/, requirements.txt, Dockerfile, README.md

## Adaptation via learner_state
Prioritize cards covering struggling_concepts.
Skip concepts already in understood_concepts (mastery > 0.85).