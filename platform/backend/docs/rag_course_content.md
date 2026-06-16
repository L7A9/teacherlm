# TeacherLM RAG And Course Content Reference

This document is the deep reference for TeacherLM retrieval-augmented generation. It explains how uploaded course files become searchable evidence, how the backend retrieves context, why each retrieval technique exists, and how that context reaches generators.

The key boundary: the platform owns RAG. Generators receive `context_chunks` through `GeneratorInput`; they do not parse files or query Qdrant themselves.

## High-Level Architecture

```text
upload
  -> MinIO original object
  -> Postgres uploaded file record
  -> ARQ ingestion worker
  -> LlamaCloud markdown parse
  -> local markdown cleaning
  -> course document/section extraction
  -> structured-section chunking
  -> optional generated search questions
  -> Postgres course_documents/course_sections/search_chunks
  -> fastembed passage embeddings
  -> Qdrant vectors
  -> concept inventory / learning map / knowledge graph / CourseBuilder rebuilds

chat or generation request
  -> load learner state and history
  -> apply source-file selection
  -> choose retrieval mode from output type
  -> retrieve dense candidates from Qdrant
  -> retrieve sparse candidates with BM25
  -> fuse with RRF
  -> add formula/comparison/graph candidates when relevant
  -> rerank with cross-encoder
  -> expand with neighbors, summaries, graph context, equations, tables, outlines
  -> dispatch GeneratorInput to generator service
```

## Data Stores And Responsibilities

| Store | Responsibility |
| --- | --- |
| MinIO | Original uploads, raw parsed markdown, cleaned text, generated artifacts |
| Postgres | Source of truth for files, documents, sections, chunks, learner state, concepts, graph nodes/edges |
| Qdrant | Dense vector index for search chunks |
| Redis | ARQ job queue |

Postgres remains the authoritative course model. Qdrant is an acceleration index for dense search and stores payload copies so dense hits can be converted back to `Chunk` objects quickly.

## Ingestion Model

The ingestion worker processes one uploaded file at a time through `ingest_file`.

1. The backend accepts a multipart upload and stores it in MinIO.
2. The worker parses the bytes with `llama-cloud >= 1.0`.
3. Raw parsed markdown is stored in MinIO.
4. `DocumentCleaningService` removes parser noise, slide footers, repeated boilerplate, mostly-punctuation lines, and low-value presentation fragments.
5. Cleaned markdown is stored in MinIO.
6. `CourseStructureExtractor` turns cleaned markdown into a `CourseDocument`.
7. `ChunkingService` creates deterministic search chunks from sections.
8. `ChunkQuestionGenerator` can add generated search questions into chunk metadata.
9. `CourseContentStore.replace_document()` replaces old document/section/chunk records for that upload in Postgres.
10. `VectorService` deletes old file vectors and upserts new vectors to Qdrant.
11. When all files in a conversation are ready, the worker rebuilds concepts, the learning map, the knowledge graph, and CourseBuilder output.

The status flow includes states such as `uploaded`, `queued`, `parsing`, `chunking`, `embedding`, `ready`, and `failed`.

## Course Model

The backend stores three content levels.

### `course_documents`

One row per parsed upload:

- conversation ID,
- uploaded file ID,
- source file ID/object key,
- source filename,
- document title,
- raw markdown object key,
- cleaned text object key,
- cleaned text hash,
- metadata.

### `course_sections`

Structured sections extracted from cleaned markdown:

- parent/child section relationships,
- heading path,
- order index,
- page hints when available,
- full section text,
- local summary,
- key concepts,
- equations,
- tables,
- timeline events,
- extractor metadata.

Sections let broad generators use course structure instead of raw chunk dumps.

### `search_chunks`

Stable, searchable evidence units:

- chunk ID,
- document ID,
- section ID,
- source filename,
- source file ID,
- chunk text,
- chunk index,
- token count,
- previous/next chunk IDs,
- page hints,
- heading path,
- chunk metadata.

Search chunks are the citation units passed to generators as `Chunk` objects.

## Chunking Strategy

Chunks are created from structured sections, not from raw document text. This matters because course headings, section summaries, formulas, and key concepts stay attached to the evidence.

`ChunkingService`:

- splits section text into paragraph/sentence units,
- preserves structured blocks such as markdown tables, bullet lists, and displayed equations,
- targets `CHUNK_MAX_TOKENS` with overlap from `CHUNK_OVERLAP_TOKENS`,
- creates deterministic UUIDs from source file ID, section ID, section chunk index, and text hash,
- records heading path, section title, key concepts, formulas/tables/timeline counts, and source file ID,
- links each chunk to previous and next chunks.

Why deterministic IDs:

- reindexing the same content creates stable citations,
- eval datasets can refer to chunk IDs,
- old vectors can be replaced cleanly,
- learner/review features can connect answers to evidence.

Why neighbor links:

- a retrieved chunk may start in the middle of an explanation,
- nearby chunks often contain definitions, setup, or follow-up examples,
- context expansion can recover local continuity without retrieving the entire section.

## Searchable Metadata

BM25 and some context policies use more than raw chunk text. Searchable text may include:

- `heading_path`,
- `section_title`,
- `key_concepts`,
- `generated_questions`.

Generated questions are not shown as course facts. They are search affordances: a chunk about "SVD factorization" may include question-like metadata such as "How does SVD decompose a matrix?", making lexical retrieval better for student-style queries.

## Dense Semantic Search

Dense search is handled by fastembed and Qdrant.

During ingestion:

1. `VectorService` loads a fastembed `TextEmbedding` model.
2. Passage embeddings are computed in batches.
3. Qdrant stores vectors in a per-conversation collection named `conv_{conversation_id}`.
4. Payload indexes are created for filterable fields such as source, file ID, source file ID, document ID, and section ID.

During retrieval:

1. The query is embedded with `query_embed` when available.
2. Qdrant performs cosine vector search.
3. Dense hits are converted back to `Chunk` objects using payload text, source, chunk ID, score, and metadata.

Default embedding settings:

| Setting | Default |
| --- | --- |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-large` |
| `EMBEDDING_DIM` | `1024` |
| `EMBEDDING_BATCH_SIZE` | `8` |

Why semantic search is used:

- students often ask in their own words,
- definitions and explanations may use synonyms,
- multilingual course material benefits from multilingual embeddings,
- dense retrieval finds conceptually similar passages even when exact words differ.

Where semantic search is weak:

- exact acronyms,
- symbols and formulas,
- short entity names,
- version numbers,
- quoted course labels,
- rare technical terms.

Those weaknesses are why TeacherLM also uses BM25, formula boosts, graph search, and reranking.

## Keyword Search And BM25

Keyword search uses `rank-bm25` through `teacherlm_core.retrieval.bm25.BM25Index`.

Tokenization is simple and transparent:

```text
lowercase text -> regex word tokens -> BM25Okapi
```

BM25 is built over the allowed chunk set for a request. If source-file selection is active, only chunks from selected ready files are indexed.

Why BM25 is used:

- exact course terms matter,
- acronyms such as "SVD", "RNN", or "NCF" should not disappear inside dense embedding space,
- formulas and symbols often require lexical overlap,
- headings and generated questions can make exact user wording recoverable,
- BM25 provides a useful fallback when Qdrant is unavailable.

BM25-only fallback:

- If the vector service cannot load,
- or if the Qdrant collection does not exist,
- the backend logs the issue and returns BM25 results.

Fallback retrieval still applies formula boosts and comparison balancing where possible.

## Reciprocal Rank Fusion

Hybrid retrieval combines dense and sparse rankings with reciprocal rank fusion (RRF).

Formula:

```text
score(document) = sum(1 / (RRF_K + rank_i(document)))
```

TeacherLM uses:

```text
RRF_K = 60
```

How it works:

1. Dense search returns top semantic candidates.
2. BM25 returns top keyword candidates.
3. Each candidate receives rank-based credit from each list where it appears.
4. Scores are summed by `chunk_id`.
5. Duplicates are removed.
6. Returned chunks receive the fused score.

Why RRF instead of choosing one retriever:

- dense search captures meaning,
- BM25 captures exact terms,
- a chunk that ranks well in both lists should be trusted more,
- rank-based fusion avoids comparing incompatible raw score scales,
- the system remains useful when one retrieval method is noisy for a query.

## Reranking

Reranking happens after candidate retrieval and before context expansion.

Default settings:

| Setting | Default |
| --- | --- |
| `RETRIEVAL_RERANK_ENABLED` | `true` |
| `RETRIEVAL_RERANK_WARMUP_ENABLED` | `true` |
| `RETRIEVAL_RERANKER_MODEL` | `BAAI/bge-reranker-base` |
| `RETRIEVAL_RERANK_CANDIDATE_K` | `50` |
| `RETRIEVAL_RERANK_TOP_K` | `16` |

The reranker is a fastembed cross-encoder. It scores each `(query, chunk_text)` pair directly, which is more precise than comparing separate query/document embeddings.

Why reranking is used:

- dense/BM25/RRF should retrieve a broad candidate set,
- the final prompt needs the best evidence at the top,
- cross-encoders are better at query-document relevance,
- reranking improves source precision without replacing broad candidate retrieval.

Fallback behavior:

- If reranker loading or scoring fails, the backend logs the exception and falls back to fused candidates.
- Retrieval remains available even if the reranker model is missing or slow to load.

Warmup:

- The backend starts reranker warmup during application startup.
- If warmup fails, retrieval lazy-loads the model later.

## Comparison Query Balancing

Comparison questions are a special case. A query such as "compare SVD and PCA" can be poorly served if all top chunks are about only one term.

TeacherLM detects comparison markers such as:

- `compare`,
- `comparison`,
- `difference`,
- `versus`,
- `vs`,
- `between`,
- French comparison terms,
- Arabic comparison terms.

It extracts compared labels, runs retrieval per label, tags candidates with `matched_query_term`, and merges them round-robin. If reranking is enabled, it reranks each group and then interleaves groups so both sides stay represented.

Why this exists:

- comparisons need evidence for both sides,
- a single high-frequency topic can dominate top-k retrieval,
- balanced groups produce better explanations and fairer citations.

## Formula And Equation Handling

Formula queries are also special. The backend detects formula intent through terms and math symbols such as:

- `formula`,
- `equation`,
- `derive`,
- `calculate`,
- `compute`,
- `=`,
- `+`,
- `-`,
- `/`,
- `^`,
- `_`.

When detected, `_merge_formula_hits()` scans allowed chunks for math-like text:

- displayed LaTeX,
- inline math,
- common math commands,
- equals signs and operators,
- subscript/superscript patterns.

Formula chunks receive scores based on query token overlap and math density, then are merged ahead of normal hits.

Why this exists:

- embeddings can blur formula-heavy content,
- exact symbolic evidence is often required for student math questions,
- formulas may appear in tables or short chunks that dense retrieval under-ranks.

## Knowledge Graph Search

The knowledge graph gives retrieval a relationship-aware path beyond text similarity.

Graph rebuild uses:

- course documents,
- sections,
- chunks,
- concept inventory,
- learning phases,
- learning objectives,
- optional LLM extraction.

Node types include:

- `course`,
- `file`,
- `section`,
- `chunk`,
- `phase`,
- `objective`,
- `concept`,
- `skill`,
- `procedure`,
- `formula`,
- `example`,
- `misconception`,
- `assessment`.

Edge types include:

- `part_of`,
- `teaches`,
- `requires`,
- `prerequisite_of`,
- `supports`,
- `explains`,
- `applies`,
- `example_of`,
- `formula_for`,
- `contrasts_with`,
- `causes`,
- `solves`,
- `assessed_by`,
- `remediates`.

Graph search flow:

1. Important query terms are extracted after stopword filtering.
2. Active graph nodes are loaded, or a fallback graph is rebuilt if none exists.
3. Nodes are scored by exact/fuzzy matches against labels, descriptions, and aliases.
4. Preferred node types such as concept, objective, skill, procedure, formula, and example get a small boost.
5. Matching node source chunks and neighboring edge chunks are collected.
6. Chunk IDs are deduped and loaded from Postgres.
7. Results are tagged with `retrieval_via: knowledge_graph`.

Why graph search is used:

- concepts are connected to objectives, examples, formulas, sections, and prerequisites,
- a student can ask about a concept whose best evidence is in related chunks,
- remediation and review need prerequisite paths,
- relationship-aware retrieval supports explanations that go beyond isolated snippets.

## Graph Context Expansion

Graph search can add candidates before reranking. Graph context expansion can also add related chunks after final selection.

For each selected chunk:

1. The orchestrator asks the knowledge graph for related chunk IDs around the selected chunks.
2. Related chunks are loaded from Postgres.
3. Source-file filters are applied.
4. The related text is trimmed and tagged as `context_type: knowledge_graph_neighbor`.
5. It is included in the expanded context text around the focused chunk.

Why this is separate from graph search:

- graph search can retrieve graph-relevant candidates from the query,
- graph expansion can enrich already-good hits with surrounding conceptual relationships.

## Low-Information Filtering

Before retrieval, the context service removes low-value chunks when possible.

Filtered examples include:

- empty text,
- very short chunks,
- page numbers,
- slide counters,
- "thank you",
- "questions?",
- table of contents-only fragments,
- chunks with too few alphabetic characters.

Why this exists:

- parsers and slide decks often produce fragments,
- noisy chunks can rank high for generic queries,
- generators should not waste prompt budget on non-teaching content.

If filtering removes everything, the original chunks are kept so retrieval does not become empty unnecessarily.

## Source-File Filtering

The frontend sends `source_file_ids` for chat and generator requests.

The backend applies that filter when loading:

- documents,
- sections,
- chunks,
- graph-relevant chunks,
- graph neighbor chunks,
- equations,
- tables,
- mind map module packs,
- representative course context.

An explicit empty list is rejected by chat/generate routes. `None` means no filter.

CourseBuilder intentionally ignores source-file selection and uses all course materials.

## Retrieval Modes

The platform maps output type to retrieval mode:

| Output type | Mode |
| --- | --- |
| `text` / `chat` | `semantic_topk` |
| `quiz` | `coverage_broad` |
| `podcast` | `narrative_arc` |
| `mindmap` | `topic_clusters` |
| `report` | `topic_clusters` |
| `presentation` | `topic_clusters` |
| `chart` / `diagram` | `relationship_dense` |

### `semantic_topk`

Used by teacher chat.

Behavior:

- runs hybrid dense+BM25 retrieval,
- fuses candidates with RRF,
- adds formula candidates for formula queries,
- adds graph candidates for non-empty queries,
- reranks candidates,
- expands final chunks with local neighbors, graph neighbors, section path, and section summary.

Why it fits chat:

- students usually ask focused questions,
- answer quality depends on the closest evidence,
- neighbor expansion preserves explanation continuity,
- graph candidates help conceptual questions.

### `coverage_broad`

Used by quiz generation.

Behavior:

- retrieves a larger hybrid pool,
- uses an MMR-like token-diversity selection,
- balances query relevance against diversity,
- skips neighbor expansion in the orchestrator so broad coverage is not narrowed by local context.

Why it fits quizzes:

- quizzes should cover the selected course material,
- repeated near-duplicate chunks make weak quizzes,
- representative breadth creates better assessment coverage.

When quiz has no topic, the platform bypasses narrow retrieval and sends full-course context: outline, representative sections, equations, and tables.

### `narrative_arc`

Used by podcast generation.

Behavior:

- picks introduction-like chunks,
- retrieves query-relevant middle chunks,
- picks conclusion-like chunks,
- dedupes the result.

When podcast has no topic, the platform sends course outline, narrative-arc retrieval, and representative sections.

Why it fits podcasts:

- audio learning needs a beginning, development, and wrap-up,
- a pure top-k list can feel jumpy,
- course-level flow helps the script sound like a lesson.

### `topic_clusters`

Used by mind maps and reserved disabled report/presentation entries.

Behavior:

- retrieves a broad pool,
- embeds pool chunks,
- clusters embeddings,
- returns representative chunks from each cluster.

For mind maps, the platform often uses richer course-structure packs instead of raw clustered chunks:

- course outline,
- per-document module packs,
- section summaries,
- key concepts,
- equations/tables when useful.

Why it fits overview outputs:

- mind maps need topic coverage,
- a single nearest-neighbor list over-focuses,
- cluster representatives preserve course breadth.

### `relationship_dense`

Reserved for disabled chart/diagram output.

Behavior:

- retrieves a hybrid pool,
- scores chunks by entity and verb density,
- returns chunks likely to contain relationships, processes, comparisons, formulas, or numeric facts.

Why it fits diagrams:

- diagrams need relationships and processes more than prose summaries,
- entity/action density is a lightweight signal for chartable content.

## Course-Overview Handling

Vague course-wide chat requests often fail with normal semantic top-k because there is no narrow query target.

The orchestrator detects requests like:

- "What is this course about?"
- "Teach me this course."
- "Where should I start?"
- "Prepare me for the exam."
- similar French and Arabic course-overview requests.

For these, it returns:

- mind map course context,
- full course outline,
- representative course sections.

If those are unavailable, it falls back to `coverage_broad`.

Why this exists:

- students often start with broad orientation questions,
- semantic top-k may pick arbitrary early chunks,
- course overview needs structure, not just nearest paragraphs.

## Section And Item Context

The context service can return structured context beyond raw chunks:

- full course outline,
- full course sections,
- representative course sections,
- topic sections,
- equations,
- tables,
- timeline events,
- mind map course outline,
- mind map module packs.

These are still delivered as `Chunk` objects so the generator contract remains stable.

`context_type` metadata tells generators and evaluators what kind of context they are seeing.

## Generator Context Policies

The orchestrator applies output-specific policies before dispatch.

| Generator/output | Topic supplied | No topic supplied |
| --- | --- | --- |
| `teacher_gen` / `text` | focused `semantic_topk` over query | course-overview detection may supply outline/representative context |
| `quiz_gen` / `quiz` | focused topic sections plus hits | outline, representative sections, equations, tables |
| `podcast_gen` / `podcast` | topic sections plus hits | outline, narrative arc, representative sections |
| `mindmap_gen` / `mindmap` | topic context | mind map course outline plus per-document module packs |
| disabled `presentation` | topic context plus equations/tables | representative course context plus equations/tables |
| disabled `chart` / `diagram` | relationship-dense context | relationship-dense context |

Why policies live in the backend:

- retrieval remains consistent across generators,
- source-file filtering is enforced once,
- generator services stay simple and stateless,
- future generators can reuse existing context policies.

## Context Expansion

After reranking, the orchestrator may expand final chunks.

Expansion can include:

- section path,
- section summary,
- previous/next chunks within a configured neighbor window,
- graph-related chunks.

The focused chunk is kept at the end of the expanded text:

```text
Section path: ...

Section summary: ...

neighbor or graph context...

Focused chunk:
...
```

Expansion is capped by `RETRIEVAL_EXPANSION_MAX_CHARS`.

Why expansion is used:

- top chunks are often snippets,
- teaching needs local context,
- citations remain stable because the focused chunk ID stays the returned chunk ID,
- graph neighbors can add examples or prerequisites without changing the main hit.

`coverage_broad` skips this expansion so quiz context remains broad.

## Generator Dispatch

After retrieval, the backend builds `GeneratorInput`:

```json
{
  "conversation_id": "...",
  "user_message": "...",
  "context_chunks": [],
  "learner_state": {},
  "chat_history": [],
  "options": {}
}
```

Dispatch flow:

1. `GeneratorRegistry` loads `generators_registry.json`.
2. `GeneratorRouter` resolves the enabled generator by chat default or `output_type`.
3. `ApiAdapter` posts JSON to the generator `/run` endpoint.
4. The adapter reads generator SSE events.
5. The backend proxies events to the frontend.
6. On `done`, the backend persists the assistant message, artifacts, sources, and learner updates.

Preserved generator endpoints:

- `GET /health`
- `GET /info`
- `POST /run`

Preserved SSE events:

- `token`
- `chunk`
- `sources`
- `artifact`
- `progress`
- `done`
- `error`

## Evaluation

Retrieval eval files can include:

```json
{
  "conversation_id": "...",
  "cases": [
    {
      "query": "What is collaborative filtering?",
      "relevant_chunk_ids": ["..."],
      "expected_section_ids": ["..."],
      "expected_source_document": "course.pdf",
      "answer_facts": ["..."],
      "output_type": "text"
    }
  ]
}
```

Run retrieval metrics:

```powershell
python platform/backend/scripts/eval_retrieval.py run platform/backend/evals/example_retrieval_eval.json --k-values 5,10,20
```

Dump chunk IDs for building an eval file:

```powershell
python platform/backend/scripts/eval_retrieval.py dump-corpus <conversation-id> --out platform/backend/evals/corpus_dump.json
```

Benchmark embedding candidates:

```powershell
python platform/backend/scripts/benchmark_embeddings.py platform/backend/evals/corpus_dump.json
```

Inspect context policies:

```powershell
python platform/backend/scripts/eval_course_context.py <conversation-id> --topic "matrix factorization"
```

Metrics include:

- Recall@K,
- Precision@K,
- HitRate@K,
- MRR@K,
- nDCG@K,
- section recall,
- citation precision,
- source-document hit,
- latency.

Use these evaluations to compare:

- dense-only behavior through embedding benchmarks,
- BM25 fallback quality,
- hybrid/RRF candidate quality,
- reranker improvements,
- graph-search additions,
- source-file filter correctness,
- broad vs focused generator context policies.

## Design Rationale

TeacherLM uses a layered retrieval design because student course questions vary widely.

- Dense search handles paraphrase and semantic similarity.
- BM25 handles exact terminology, acronyms, symbols, generated questions, and headings.
- RRF avoids comparing incompatible score scales and combines both retrieval styles.
- Reranking improves final evidence order after broad candidate recall.
- Graph search adds relationships, prerequisites, examples, and formulas.
- Context expansion gives final chunks enough local teaching context.
- Output-specific policies keep chat focused, quizzes broad, podcasts narrative, and mind maps structural.

The result is a RAG pipeline that is still inspectable: every stage returns `Chunk` objects with source, score, ID, and metadata, and every generator sees the same stable evidence contract.
