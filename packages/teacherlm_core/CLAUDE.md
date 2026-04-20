# teacherlm_core — Shared Library

## Purpose
Reusable utilities used by platform AND all generators. Published as 
local editable package: pip install -e packages/teacherlm_core

## Module Map (files < 250 lines)
teacherlm_core/
├── pyproject.toml
├── teacherlm_core/
│   ├── __init__.py
│   ├── schemas/
│   │   ├── generator_io.py        # GeneratorInput/Output
│   │   ├── learner_state.py
│   │   └── chunk.py
│   ├── retrieval/
│   │   ├── hybrid_retriever.py    # BM25 + dense + RRF
│   │   ├── reranker.py             # fastembed cross-encoder
│   │   ├── retrieval_modes.py     # 5 modes from root CLAUDE.md
│   │   └── bm25.py
│   ├── llm/
│   │   ├── ollama_client.py       # async wrapper
│   │   ├── structured.py          # format= helper
│   │   └── streaming.py           # SSE generator
│   ├── prompts/
│   │   ├── teacher_voice.txt      # SHARED personality prompt
│   │   ├── tone_guidelines.txt
│   │   └── citation_rules.txt
│   ├── confidence/
│   │   ├── groundedness.py
│   │   └── coverage.py
│   └── config.py
└── tests/