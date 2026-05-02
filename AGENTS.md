# TeacherLM — Root AGENTS.md

## Product
AI teacher for students. Upload course files → get a personal tutor that:
- Chats naturally (warm, encouraging teacher voice)
- Generates quizzes, reports, mind maps, diagrams, podcasts, presentations
- Tracks what the student has learned vs struggled with
- Always grounds answers in uploaded files only

## Python: 3.14+ (strict)

## Compatibility Rules (Non-Negotiable)
- NEVER use LangChain / LangGraph (Pydantic V1 warnings on 3.14)
- NEVER use llama-parse / llama-cloud-services (deprecated May 2026)
- USE llama-cloud >= 1.0
- USE Pydantic V2 only (>=2.12)
- USE ollama python lib's native format= for structured outputs
- USE FastAPI >= 0.135
- USE fastembed (not sentence-transformers where avoidable)

## Top-Level Structure
teacherlm/
├── AGENTS.md                      # you are here
├── generators_registry.json       # plugin registry
├── packages/
│   └── teacherlm_core/            # shared library used by everything
├── platform/
│   ├── backend/
│   ├── frontend/
│   └── docker-compose.yml
└── generators/
    ├── teacher_gen/               # chat Q&A + guidance
    ├── quiz_gen/
    ├── report_gen/
    ├── presentation_gen/
    ├── chart_gen/
    └── podcast_gen/

## Naming Rule
"Agent" terminology is internal. User-facing: "output types" or 
"generators". Backend code uses "generator" consistently.

## Shared Core Package (packages/teacherlm_core)
- retrieval/ (hybrid retriever, reranker, retrieval modes)
- llm/ (ollama wrapper, structured output helpers)
- prompts/teacher_voice.txt (SHARED teacher personality prompt)
- confidence/ (groundedness scoring)
- schemas/ (GeneratorInput, GeneratorOutput, LearnerState)

Every generator pip-installs this as local dependency.

## Generator I/O Contract (immutable)
Input:
{
  "conversation_id": str,
  "user_message": str,
  "context_chunks": [{"text", "source", "score", "chunk_id"}],
  "learner_state": {                     # NEW: tracks student progress
    "understood_concepts": [str],
    "struggling_concepts": [str],
    "mastery_scores": {str: float},
    "session_turns": int
  },
  "chat_history": [{"role", "content"}],
  "options": dict
}

Output:
{
  "response": str,                       # markdown, teacher voice
  "generator_id": str,
  "output_type": "text|quiz|report|presentation|chart|podcast|mindmap",
  "artifacts": [{"type", "url", "filename"}],
  "sources": [{"text", "source", "score"}],
  "learner_updates": {                   # generators report back
    "concepts_covered": [str],
    "concepts_demonstrated": [str],      # student proved understanding
    "concepts_struggled": [str]          # student showed confusion
  },
  "metadata": dict                       # generator-specific (confidence, etc.)
}

## Retrieval Modes (platform provides, generators request)
- "semantic_topk": top-K closest to query (for chat)
- "coverage_broad": sampled across document (for quizzes)
- "narrative_arc": intro + key points + conclusion (for podcasts, reports)
- "topic_clusters": chunks grouped by topic (for presentations, reports)
- "relationship_dense": chunks with many entities/relations (for charts)

## When Working in Subdirectories
Each has its own AGENTS.md. Read it first.
Do NOT open files outside your current subdirectory unless the 
AGENTS.md explicitly references them.