# Quiz Generator — CLAUDE.md

## Port: 8002 | Python 3.14+
## Purpose: Interactive quizzes from uploaded content

## Stack
- FastAPI, ollama (native format=)
- teacherlm_core
- fastembed (distractor similarity)
- keybert (concept extraction) + fastembed backend

## Module Map
quiz_gen/
├── CLAUDE.md
├── app.py
├── config.py
├── schemas.py                     # MCQ, TrueFalse, FillBlank, QuizOutput
├── pipeline.py
├── services/
│   ├── concept_extractor.py       # Bloom's taxonomy grouping
│   ├── question_generator.py      # per-type via ollama format=
│   ├── distractor_engine.py       # semantic hard negatives
│   ├── difficulty_adapter.py      # NEW: uses learner_state
│   ├── quality_validator.py
│   └── llm_service.py
├── prompts/
│   ├── concept_extraction.txt
│   ├── mcq_generation.txt
│   ├── true_false_generation.txt
│   ├── fill_blank_generation.txt
│   └── adaptive_guidance.txt      # how to frame quiz based on state
├── requirements.txt
├── Dockerfile
└── README.md

## Key Teacher Adaptation
learner_state.struggling_concepts → bias quiz toward those
learner_state.understood_concepts → don't re-quiz easy stuff
Intro message from teacher voice: 
  "Let's test what you know! I've focused on areas we've been 
   working on: {struggling}. Ready?"