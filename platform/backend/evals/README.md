# Retrieval And Course Context Evaluation

This folder is for gold retrieval datasets. Each dataset points to one already
ingested conversation and contains student-style queries plus the chunk/section
ids and source document that should be retrieved.

1. Dump chunk ids for a conversation:

```powershell
python platform/backend/scripts/eval_retrieval.py dump-corpus <conversation_id> --out platform/backend/evals/corpus.json
```

2. Create an eval file:

```json
{
  "conversation_id": "00000000-0000-0000-0000-000000000000",
  "cases": [
    {
      "id": "q001",
      "query": "Qu'est-ce qu'un systeme de recommandation ?",
      "output_type": "text",
      "relevant_chunk_ids": ["chunk-id-from-corpus-dump"],
      "expected_section_ids": ["section-id-from-corpus-dump"],
      "expected_source_document": "course.pdf",
      "answer_facts": ["A short fact the final answer should contain."]
    }
  ]
}
```

3. Run metrics:

```powershell
python platform/backend/scripts/eval_retrieval.py run platform/backend/evals/my_course.json --k-values 1,3,5,10,20 --out platform/backend/evals/results.json
```

Tracked metrics:

- `hit_rate@K`
- `precision@K`
- `recall@K`
- `mrr@K`
- `ndcg@K`
- `section_recall@K`
- `citation_precision`
- `source_document_hit`
- `latency_ms`

Inspect broad generator context policies:

```powershell
python platform/backend/scripts/eval_course_context.py <conversation_id> --topic "SVD"
```

Benchmark fastembed-compatible model candidates:

```powershell
python platform/backend/scripts/benchmark_embeddings.py platform/backend/evals/corpus.json
```

Use this before changing embedding models, chunking, candidate pools, or reranking.

## What to Benchmark

Build at least 100 course-specific cases before claiming a RAG improvement:

- direct definition questions
- equation or formula lookup questions
- comparison questions
- process / sequence questions
- French queries against French chunks
- broad generator queries such as presentation/report topics
- hard negatives where the answer is not in the course

For each experiment, change one variable at a time:

- `EMBEDDING_MODEL` and `EMBEDDING_DIM`
- `CHUNK_MAX_TOKENS` and `CHUNK_OVERLAP_TOKENS`
- `RETRIEVAL_DENSE_CANDIDATE_K`
- `RETRIEVAL_SPARSE_CANDIDATE_K`
- `RETRIEVAL_CONTEXT_EXPANSION_ENABLED`
- `RETRIEVAL_RERANK_ENABLED`
- `RETRIEVAL_RERANKER_MODEL`

When changing the embedding model or chunking settings, re-ingest the course
before running the benchmark, otherwise Qdrant still contains old vectors/chunks.
