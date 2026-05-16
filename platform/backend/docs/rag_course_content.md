# TeacherLM RAG + Course Content

## Ingestion Pipeline

Uploaded files are stored in MinIO, parsed with `llama-cloud`, cleaned locally, converted into a structured course document, chunked by section, embedded with fastembed, and indexed in Qdrant.

Pipeline:

```text
uploaded file
-> parse markdown
-> clean parser/slide noise
-> extract document structure
-> store cleaned full text in MinIO
-> store documents/sections/chunks in Postgres
-> embed search chunks
-> store vectors in Qdrant
```

Postgres is the source of truth for searchable metadata and course structure. Qdrant is only the dense vector index.

## Course Model

The backend stores three content levels:

- `course_documents`: one parsed upload, including title, raw markdown key, cleaned text key, source filename, and text hash.
- `course_sections`: document sections with heading path, order, text, summary, key concepts, equations, tables, and timeline events.
- `search_chunks`: stable deterministic chunks linked to document and section IDs, with heading path, neighbor IDs, token count, and citation metadata.

## Chunking Strategy

Chunks are created from structured sections, not from raw document text. Chunk IDs are deterministic from source file ID, section ID, section chunk index, and text hash. Each chunk keeps:

- `document_id`
- `section_id`
- `parent_section_id`
- `heading_path`
- `chunk_index`
- `prev_chunk_id`
- `next_chunk_id`
- source filename and source file ID

Focused retrieval can expand to neighboring chunks and section summaries without losing citations.

## Retrieval Modes

- `semantic_topk`: hybrid dense + BM25 search, rerank, neighbor expansion.
- `coverage_broad`: representative coverage across course content.
- `narrative_arc`: outline plus representative section flow for reports and podcasts.
- `topic_clusters`: topic-oriented section/chunk context.
- `relationship_dense`: chunks/sections with entities, processes, comparisons, equations, or numeric facts.

Hybrid retrieval uses dense candidates, BM25 candidates, RRF fusion, deduplication, optional cross-encoder reranking, and section-aware expansion.

## Generator Context Policies

- `teacher_gen/text`: retrieved chunks only, reranked and expanded with local neighbors.
- `podcast_gen`: topic uses focused sections/chunks; no topic uses outline plus representative sections.
- `quiz_gen`: topic uses focused section coverage; no topic samples major sections across the full course.
- `mindmap_gen`: outline plus section summaries/key concepts, not a raw chunk dump.
- `presentation_gen`: topic uses focused context plus equations/tables; no topic uses course coverage plus equations/tables.
- `chart/diagram`: relationship-dense context with equations, processes, comparisons, and numeric facts.

The `GeneratorInput` and `GeneratorOutput` contracts stay unchanged. Richer course context is still delivered as `context_chunks` with metadata.

## Evaluation And Benchmarks

Retrieval eval dataset shape:

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

Metrics include Recall@K, Precision@K, MRR@10, nDCG@10, section recall, citation precision, source-document hit, and latency.
