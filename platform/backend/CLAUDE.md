# Platform Backend — CLAUDE.md

## Python 3.14+ | FastAPI 0.135+

## Depends On: packages/teacherlm_core (editable install)

## Module Map (files < 250 lines)
backend/
├── main.py
├── config.py
├── db/
│   ├── session.py
│   ├── models.py                  # Conversation, Message, File, LearnerState
│   └── migrations/
├── schemas/
│   ├── conversation.py
│   ├── message.py
│   └── file.py
├── routers/
│   ├── conversations.py
│   ├── files.py
│   ├── chat.py                    # main chat endpoint (SSE)
│   ├── generate.py                # button-triggered generations
│   ├── generators.py              # list available generators
│   └── health.py
├── services/
│   ├── parsing_service.py         # llama-cloud wrapper
│   ├── chunking_service.py
│   ├── vector_service.py
│   ├── storage_service.py         # MinIO
│   ├── learner_tracker.py         # NEW: updates learner state
│   └── retrieval_orchestrator.py  # picks retrieval mode, uses teacherlm_core
├── dispatcher/
│   ├── registry.py
│   ├── router.py
│   └── adapters/
│       ├── api_adapter.py
│       └── mcp_adapter.py
├── workers/
│   └── ingestion_worker.py        # arq task
└── requirements.txt

## Key New Behavior: Learner State

Every chat message + generation updates the learner state for the 
conversation. Stored in Postgres:

LearnerState(conversation_id PK, state_json, updated_at)

When the Teacher generator (or any generator) returns learner_updates,
learner_tracker.py merges them:
- concepts_covered: bumps encounter count
- concepts_demonstrated: mastery += 0.2 * (1 - mastery)
- concepts_struggled: mastery *= 0.7

Before dispatching any generator, platform loads current learner_state 
and passes it in GeneratorInput.

## Two Entry Points for Generation

1. POST /api/conversations/{id}/chat (SSE)
   - Always routes to Teacher generator
   - Used when student types in chat

2. POST /api/conversations/{id}/generate (SSE)
   - Body: {output_type: "quiz"|"report"|..., options: {...}, 
           topic?: str}  # optional topic narrows retrieval
   - Used when student clicks output-type button
   - Routes to the matching generator

## Retrieval Mode Selection
Based on output_type:
- chat → semantic_topk
- quiz → coverage_broad
- report → topic_clusters
- presentation → topic_clusters
- podcast → narrative_arc
- chart → relationship_dense