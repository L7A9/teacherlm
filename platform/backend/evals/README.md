# Backend Evaluations

This directory contains small evaluation datasets for retrieval and course-context behavior. The scripts live in `platform/backend/scripts`.

## Retrieval Eval Cases

Retrieval eval files are JSON arrays. Each item should include a query and expected source evidence.

Typical fields:

```json
[
  {
    "query": "What is normalization?",
    "expected_sources": ["chapter_2.pdf"],
    "expected_terms": ["normalization", "relation", "dependency"],
    "mode": "semantic_topk"
  }
]
```

Keep eval cases grounded in uploaded course content. The scripts are designed to help compare retrieval settings, embedding models, reranker behavior, and course-context policies.

## Commands

Run retrieval evaluation:

```bash
cd platform/backend
python scripts/eval_retrieval.py run --conversation-id <conversation-id> --cases evals/<file>.json
```

Dump a conversation corpus for eval authoring:

```bash
cd platform/backend
python scripts/eval_retrieval.py dump-corpus --conversation-id <conversation-id> --output evals/corpus.json
```

Inspect output-specific course context:

```bash
cd platform/backend
python scripts/eval_course_context.py --conversation-id <conversation-id> --output-type quiz
```

Benchmark embedding models:

```bash
cd platform/backend
python scripts/benchmark_embeddings.py
```

Reindex from already parsed markdown:

```bash
cd platform/backend
python scripts/reindex_from_parsed.py --conversation-id <conversation-id>
```

## What To Check

When reviewing an eval run, look at:

- Whether expected files appear in the retrieved sources.
- Whether expected terms appear in the returned chunks.
- Reranker changes before and after ranking.
- Whether source-file filtering changes results correctly.
- Whether output-specific context policies produce the right breadth or focus.
- Whether broad generators, especially quiz and podcast, cover the whole selected material rather than only the first matching chunk.

## Retrieval Defaults

The current platform defaults are:

| Setting | Default |
| --- | ---: |
| `RETRIEVAL_TOP_K` | 16 |
| `RETRIEVAL_RERANK_TOP_K` | 16 |
| `RETRIEVAL_DENSE_CANDIDATE_K` | 80 |
| `RETRIEVAL_SPARSE_CANDIDATE_K` | 80 |
| `RETRIEVAL_RERANK_CANDIDATE_K` | 50 |

Output mode mapping:

| Output type | Mode |
| --- | --- |
| `text` | `semantic_topk` |
| `quiz` | `coverage_broad` |
| `podcast` | `narrative_arc` |
| `mindmap` | `topic_clusters` |
| `report` | `topic_clusters` |
| `presentation` | `topic_clusters` |
| `chart` / `diagram` | `relationship_dense` |

CourseBuilder is intentionally outside source-file selection and should be evaluated as a full-course flow.
