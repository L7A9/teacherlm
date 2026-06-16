# Data, Retrieval, And Learning Model

This file explains how TeacherLM turns uploaded files into searchable course knowledge, how retrieval is assembled for generators, and how learner progress is tracked.

It overlaps with backend services, but this view follows the data rather than the directory tree.

## Main Stored Data Types

| Data type | Stored in | Why it exists |
| --- | --- | --- |
| Original upload | MinIO | Preserve the exact student-uploaded file. |
| Uploaded file row | PostgreSQL `UploadedFile` | Track filename, status, size, storage key, parser metadata, errors, and ingestion lifecycle. |
| Raw parsed markdown | MinIO | Keep the direct llama-cloud parser output for debugging and reprocessing. |
| Cleaned markdown | MinIO | Keep normalized text used for course extraction and chunking. |
| Course document | PostgreSQL `CourseDocumentRecord` | One structured representation of a parsed file. |
| Course section | PostgreSQL `CourseSectionRecord` | Hierarchical sections extracted from headings and markdown structure. |
| Search chunk | PostgreSQL `SearchChunkRecord` | Grounding unit for retrieval, citations, and generation. |
| Vector point | Qdrant | Embedding and payload for semantic search. |
| Concept | PostgreSQL `CourseConceptRecord` | Canonical concept inventory for the course. |
| Learning phase/objective | PostgreSQL learning map tables | Course progression model for progress UI and course generation. |
| Knowledge graph node/edge | PostgreSQL graph tables | Prerequisites, dependencies, and remediation paths. |
| Learner state | PostgreSQL `LearnerStateRecord` | Per-conversation learner progress. |
| Generated course | PostgreSQL coursebuilder tables | Course builder chapters, lessons, blocks, quizzes, and progress events. |
| Artifacts | MinIO or generator static volume | Quiz JSON, podcast audio/transcripts, mindmap HTML/JSON, and other generated outputs. |

## Upload And Ingestion Lifecycle

The upload lifecycle starts in the frontend and ends with ready chunks in both PostgreSQL and Qdrant.

1. Student uploads a file in `FileUploader`.
2. Frontend sends multipart upload to `files.py`.
3. Backend stores the original bytes in MinIO.
4. Backend creates or updates an `UploadedFile` row.
5. Backend enqueues `ingest_file` with ARQ.
6. Worker marks file as processing.
7. Worker fetches bytes from MinIO.
8. Worker parses bytes with llama-cloud.
9. Worker writes raw parsed markdown to MinIO.
10. Worker cleans markdown.
11. Worker writes cleaned markdown to MinIO.
12. Worker extracts course document structure.
13. Worker chunks sections into retrieval units.
14. Worker may generate student-style questions for chunk metadata.
15. Worker replaces document, section, and chunk rows for that file.
16. Worker deletes old Qdrant vectors for that file.
17. Worker embeds chunks with fastembed.
18. Worker upserts vector points into Qdrant.
19. Worker marks file ready.
20. If all files for the conversation are ready, worker rebuilds concepts, learning map, graph, and course builder state.

The file status shown in the frontend comes from `UploadedFile`.

## Parser Layer

The parser layer is in `parsing_service.py`.

Inputs:

- Uploaded bytes.
- Original filename.
- MIME/type hints where available.
- Runtime parser settings.

Output:

- Markdown text.
- Parser job ID or metadata.
- Any parser-specific result metadata.

The project intentionally uses `llama-cloud` rather than deprecated llama parsing packages.

## Cleaning Layer

The cleaning layer is in `document_cleaning_service.py`.

It removes or reduces:

- Parser boilerplate.
- Repeated lines.
- Footer/header-like noise.
- Presentation extraction artifacts.
- Low-value punctuation-only lines.
- Some malformed parser output.

The cleaner returns both text and statistics. The stats are useful because noisy parser output can damage every later step.

## Course Structure Extraction

The structured course model is created by `course_structure_service.py`.

Main objects:

- `CourseDocument`
- `CourseSection`
- `CourseTable`

Extracted information can include:

- Document title.
- Section hierarchy.
- Heading paths.
- Section summaries.
- Key concepts.
- Equations.
- Tables.
- Timeline events.
- Intake metadata.
- Source file information.

Stable IDs are important. They let chunks, sections, concepts, and generated course blocks keep traceable references across rebuilds where possible.

## Chunking

The chunking layer is in `chunking_service.py`.

Each chunk should be:

- Small enough for retrieval and context windows.
- Large enough to preserve meaning.
- Tied to a document, file, section, and heading path.
- Linked to neighboring chunks when possible.
- Rich in metadata for filtering, reranking, citation, and context expansion.

Typical metadata:

- `conversation_id`
- `file_id`
- `document_id`
- `section_id`
- `chunk_id`
- `source`
- `heading_path`
- `section_title`
- `key_concepts`
- `generated_questions`
- `equations`
- `tables`
- `timeline_events`
- previous/next chunk IDs

The generated questions are not shown as course content. They are retrieval hints: a chunk about a topic can match a student question even if the exact wording is not in the original source.

## Vector Storage

`vector_service.py` owns Qdrant access.

Main responsibilities:

- Create a per-conversation Qdrant collection.
- Load fastembed embedding models lazily.
- Embed passages and queries.
- Create payload indexes.
- Upsert chunks with payloads.
- Delete all vectors for a conversation.
- Delete vectors for one file.
- Search vectors.
- Scroll all vector points when needed.
- Close model/client resources on shutdown.

The per-conversation collection model makes deletion simple and keeps course data scoped to one study workspace.

## Hybrid Retrieval

The project uses multiple signals:

- Dense semantic search from Qdrant.
- Sparse lexical search from BM25.
- Generated question metadata for better lexical recall.
- Reciprocal rank fusion.
- Optional reranking with a cross-encoder.
- Context expansion to neighboring chunks.
- Knowledge graph expansion.
- Formula/table/timeline-aware handling.
- Source-file filtering.

The backend's main implementation is `retrieval_orchestrator.py`.

The shared core also has reusable retrieval primitives:

- `teacherlm_core.retrieval.bm25`
- `teacherlm_core.retrieval.hybrid_retriever`
- `teacherlm_core.retrieval.reranker`
- `teacherlm_core.retrieval.retrieval_modes`

## Retrieval Orchestrator

`retrieval_orchestrator.py` receives a conversation ID, query, mode, source filters, and options.

It can:

- Search dense vectors.
- Search lexical/BM25-style signals.
- Merge and deduplicate chunks.
- Rerank results.
- Add nearby chunks for context.
- Balance comparison queries so multiple compared topics are represented.
- Detect course-overview questions.
- Detect formula-heavy questions.
- Include graph neighbors.
- Respect selected `source_file_ids`.
- Return source-rich context for generator dispatch.

Important behaviors:

- It tries not to answer from files the student did not select.
- It can expand from a hit to previous/next chunks when a local section needs continuity.
- It can avoid overfitting to a single near-neighbor result when the request needs broad coverage.

## Course Context Service

`course_context_service.py` provides higher-level context policies.

It can return:

- Relevant chunks.
- Full course outline.
- Section summaries.
- Representative context across the course.
- Topic-specific context.
- Mindmap context.
- Equations.
- Tables.
- Timeline entries.
- Generator-specific context packs.
- Module packs for mind maps.

It is used when a generator needs more than "top k chunks." For example, a mind map benefits from module/topic packs, and a podcast benefits from narrative structure.

## Source File Selection

The frontend lets the student select which uploaded files are active. The selected file IDs travel through chat/generation requests as `source_file_ids`.

Backend behavior:

- Retrieval filters chunks by selected files.
- Chat/generate requests should not answer from unselected files.
- File deletion removes its content and vectors.
- If only one ready file exists, the frontend forces it selected.
- If multiple ready files exist, the frontend defaults all ready files to selected.
- The frontend prevents deselecting the last selected ready file.

This is important for multi-document courses where the student may ask about only one lecture or chapter.

## Generator Dispatch Context

Before a generator receives a request, the backend creates:

- `conversation_id`
- `user_message`
- retrieved `context_chunks`
- current `learner_state`
- recent `chat_history`
- `options`

The generator should treat `context_chunks` as its available source material. It returns:

- Streamed chunks/tokens.
- Sources.
- Artifacts.
- Learner updates.
- Final metadata.

The backend persists the final output and connects artifacts to messages.

## Concept Inventory

`concept_inventory_service.py` builds course concepts.

Inputs:

- Search chunks.
- Section metadata.
- Key concept metadata.
- Optional LLM structured extraction.

Outputs:

- Canonical concept rows.
- Concept labels.
- Source chunk IDs.
- Importance/frequency-like metadata.

The service uses fallback extraction when the LLM is unavailable or returns weak data. It also filters noisy concept names so UI and learning state do not fill with parser artifacts.

## Learning Map

`learning_map_service.py` builds course progression.

It creates:

- Learning phases.
- Learning objectives.
- Source references.
- Phase/objective order.
- Objective-to-concept links where possible.

It can use LLM output or deterministic fallback. The learning map feeds:

- Progress panel.
- Knowledge checks.
- Review tests.
- Course player/course builder logic.

## Knowledge Graph

`knowledge_graph_service.py` builds a graph of course concepts.

Graph nodes can represent concepts, formulas, methods, topics, or similar learning objects. Edges can represent prerequisites, supports, examples, contrasts, applications, and related links.

The graph is used for:

- Remediation paths.
- Related concept hints.
- Context expansion.
- Course player hints.
- Knowledge graph API responses.

Fallback graph construction keeps the system useful when LLM graph generation fails.

## Learner State

`learner_tracker.py` owns learner progress updates.

Learner state includes simple fields:

- `understood_concepts`
- `struggling_concepts`
- `mastery_scores`
- `session_turns`

It also includes canonical progress fields:

- Concepts with IDs.
- Objectives.
- Phases.
- Demonstrated concepts.
- Struggling concepts.
- Assessment history.
- Remediation links.

Updates come from:

- Teacher generator learner extraction.
- Quiz artifact attempts.
- Knowledge checks.
- Review tests.
- Course builder quizzes.

The frontend has an optimistic local store, but the backend state is canonical.

## Knowledge Checks

`knowledge_assessment_service.py` creates and grades checks.

Question types include:

- Multiple choice.
- True/false.
- Fill-like or short answer depending on generated check.
- Short answer grading with heuristic and optional LLM support.

The service chooses concepts/objectives/phases based on learner state and course structure. It applies assessment results back to learner state.

## Review Tests

`review_test_service.py` schedules review based on course discussion history.

Process:

1. Chat identifies learning questions and records answered course questions.
2. Review windows become due after configured intervals.
3. Student can start a review test.
4. Service generates or selects review questions.
5. Student submits answers.
6. Learner state is updated.
7. Window can be snoozed or dismissed.

This gives the platform spaced-review behavior instead of only one-off chat.

## Course Builder

`coursebuilder_service.py` builds the main generated course surface.

It uses:

- Parsed course structure.
- Chunks and citations.
- Concept inventory.
- Learning map.
- Knowledge graph hints.
- Runtime LLM settings.
- Fallback deterministic builders.

It outputs:

- Course title and description.
- Chapters.
- Lessons.
- Lesson blocks.
- Tables, equations, charts, examples, markdown explanations.
- Chapter quizzes.
- Citations.
- Progress events.

The course builder is careful about source support. It validates block types and tries to avoid unsupported generated claims.

## Course Player

`course_player_service.py` is an older/adaptive course surface.

It still matters because routes and UI exist for it. It can:

- Rebuild a course plan.
- Read chapters and lessons.
- Unlock chapters.
- Submit chapter quizzes.
- Include remediation hints.

The course builder appears to be the main current generated course UI, but course player is still part of the backend.

## Artifact Storage

Artifacts can be stored in different places:

- Quiz JSON and podcast files are commonly stored in MinIO by their generators.
- Mindmap files are stored in the mindmap generator's static artifact directory and served by that generator.
- Backend message records store artifact metadata and, when possible, MinIO object keys.
- Conversation reads can refresh presigned artifact URLs.

This split matters when deploying. MinIO artifacts need signed URLs; generator-static artifacts need the generator public URL.

## Failure And Fallback Philosophy

Most pipeline stages have fallbacks:

- Parser failure marks file failed and retryable.
- Cleaning tries to preserve usable text even with noisy parser output.
- Course structure extraction has deterministic heading/markdown fallback.
- Concept inventory can fall back to metadata/text heuristics.
- Learning map can fall back to ordered chunks/sections.
- Knowledge graph can fall back to simple concept relationships.
- Course builder can fall back to source-structured chapters and blocks.
- Generators can emit no-material or insufficient-context responses instead of inventing content.

The goal is graceful degradation while preserving grounding.
