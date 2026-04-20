# Teacher Generator — CLAUDE.md

## Port: 8001 | Python 3.14+
## Purpose: Chat Q&A with teacher personality + adaptive guidance

## Key Responsibility
This IS the main chat experience. It must:
1. Answer questions grounded in context chunks
2. Detect when student is confused → switch to guiding mode
3. Detect when student gets something right → affirm + advance
4. Track learner state and report updates back
5. Use consistent warm teacher voice (from teacherlm_core prompts)
6. Always cite sources
7. Return confidence score

## Stack
- FastAPI, ollama
- teacherlm_core (shared package)
- fastembed (for reranking, via teacherlm_core)
- pydantic >=2.12

## Module Map
teacher_gen/
├── CLAUDE.md
├── app.py                         # FastAPI /run /health /info
├── config.py
├── schemas.py                     # QueryAnalysis, TeacherResponse
├── pipeline.py
├── services/
│   ├── query_analyzer.py          # classify intent + detect confusion
│   ├── response_mode.py           # picks: explain | guide | quiz_back | affirm
│   ├── hyde_generator.py          # hypothetical doc for retrieval boost
│   ├── confidence_scorer.py       # uses teacherlm_core.confidence
│   ├── learner_analyzer.py        # extracts updates from turn
│   └── llm_service.py
├── prompts/
│   ├── query_analysis.txt
│   ├── mode_explain.txt
│   ├── mode_guide.txt             # Socratic questioning
│   ├── mode_quiz_back.txt         # check understanding
│   ├── mode_affirm.txt            # encouragement + next step
│   └── learner_update_extraction.txt
├── requirements.txt
├── Dockerfile
└── README.md