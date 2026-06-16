# Backend Evaluations

This directory contains small evaluation datasets and reports for retrieval and course-context behavior. The scripts live in `platform/backend/scripts`.

Use these evaluations to inspect the RAG pipeline, not the generators themselves. They help answer questions such as:

- Did dense semantic search retrieve the right area of the course?
- Did BM25 recover exact acronyms, formulas, or course labels?
- Did RRF fusion improve candidate recall?
- Did reranking improve final top-k order?
- Did graph search add related prerequisite/example/formula evidence?
- Did source-file filtering restrict results correctly?
- Did broad outputs such as quizzes and mind maps get course coverage rather than one narrow snippet?

## Directory Contents

Typical files in this directory:

| File or folder | Purpose |
| --- | --- |
| `example_retrieval_eval.json` | Small example gold retrieval file. |
| `sample_retrieval_eval.json` | Additional sample retrieval cases. |
| `practical_student_questions.json` | Student-style practical question catalog. |
| `current_mobile_rag_eval.json` | Current mixed retrieval benchmark input. |
| `current_mobile_exact_rag_eval.json` | Current exact-term retrieval benchmark input. |
| `retrieval_variant_comparison.*` | Semantic-only vs BM25-only vs hybrid RRF reports for mixed cases. |
| `retrieval_variant_exact_comparison.*` | Variant comparison reports for exact-term cases. |
| `retrieval_variant_chart.mmd` | Mermaid chart for mixed variant comparison. |
| `retrieval_variant_exact_chart.mmd` | Mermaid chart for exact-term variant comparison. |
| `teacherlm_report_charts.ipynb` | Generated notebook used to create report charts. |
| `build_teacherlm_report_charts_notebook.py` | Notebook generator for chart/report artifacts. |
| `report_charts/` | Generated SVG/PNG charts for reports. |
| `course_pdf_text_extracts/` | Extracted course text used as eval/reference material. |

## Retrieval Eval File Shape

`eval_retrieval.py run` expects a JSON object with a conversation ID and a non-empty `cases` array.

```json
{
  "conversation_id": "00000000-0000-0000-0000-000000000000",
  "cases": [
    {
      "id": "svd_definition",
      "query": "What is SVD?",
      "relevant_chunk_ids": ["chunk-id-1"],
      "expected_section_ids": ["section-id-1"],
      "expected_source_document": "Lecture_04.pdf",
      "answer_facts": ["SVD factorizes a matrix into singular vectors and singular values."],
      "output_type": "text"
    }
  ]
}
```

Accepted evidence fields:

- `relevant_chunk_ids` or `gold_chunk_ids`
- `expected_section_ids` or `relevant_section_ids`
- `expected_source_document`
- `relevant_source_contains`
- `answer_facts`

At least one evidence field is required per case.

Routing fields:

- `mode`: directly evaluate a retrieval mode such as `semantic_topk` or `coverage_broad`.
- `output_type`: evaluate the full output-type policy through `RetrievalOrchestrator.retrieve_for()`.

If neither is supplied, the script uses `--output-type`, defaulting to `text`.

## Commands

Run retrieval evaluation:

```bash
cd platform/backend
python scripts/eval_retrieval.py run evals/example_retrieval_eval.json --k-values 5,10,20
```

Write the report to disk:

```bash
cd platform/backend
python scripts/eval_retrieval.py run evals/example_retrieval_eval.json --out evals/latest_retrieval_report.json
```

Dump chunk IDs and previews for eval authoring:

```bash
cd platform/backend
python scripts/eval_retrieval.py dump-corpus <conversation-id> --out evals/corpus_dump.json
```

Inspect output-specific course context:

```bash
cd platform/backend
python scripts/eval_course_context.py <conversation-id> --output-types text,quiz,podcast,mindmap --query "matrix factorization"
```

Inspect a topic-focused policy:

```bash
cd platform/backend
python scripts/eval_course_context.py <conversation-id> --output-types quiz,podcast --topic "matrix factorization"
```

Benchmark embedding models over a dumped corpus:

```bash
cd platform/backend
python scripts/benchmark_embeddings.py evals/corpus_dump.json --out evals/embedding_benchmark.json
```

Compare dense-only, BM25-only, and hybrid RRF retrieval:

```bash
cd platform/backend
python scripts/compare_retrieval_variants.py evals/current_mobile_rag_eval.json --k-values 5 --out evals/retrieval_variant_comparison.json --csv-out evals/retrieval_variant_comparison.csv --mermaid-out evals/retrieval_variant_chart.mmd
```

Compare exact-term cases:

```bash
cd platform/backend
python scripts/compare_retrieval_variants.py evals/current_mobile_exact_rag_eval.json --k-values 5 --chart-title "TeacherLM Exact-Term Retrieval Comparison" --out evals/retrieval_variant_exact_comparison.json --csv-out evals/retrieval_variant_exact_comparison.csv --mermaid-out evals/retrieval_variant_exact_chart.mmd
```

Run the root helper that starts needed services, runs both comparisons, rebuilds the report notebook, and regenerates SVG/PNG chart files:

```bash
./run.sh report-charts
```

Reindex from already parsed markdown:

```bash
cd platform/backend
python scripts/reindex_from_parsed.py <uploaded-file-id>
```

Reindex all files that have parsed markdown:

```bash
cd platform/backend
python scripts/reindex_from_parsed.py --all
```

## Metrics

Retrieval reports include:

| Metric | Meaning |
| --- | --- |
| `hit_rate@K` | Whether at least one relevant chunk appeared in the top K |
| `precision@K` | Fraction of top K chunks that were relevant |
| `recall@K` | Fraction of known relevant chunks recovered by top K |
| `mrr@K` | Reciprocal rank of the first relevant chunk |
| `ndcg@K` | Ranking quality with higher credit for relevant chunks near the top |
| `section_recall@K` | Fraction of expected sections represented in top K |
| `citation_precision` | Fraction of cited IDs that are relevant when citations are provided |
| `source_document_hit` | Whether expected source document appeared in retrieved sources |
| `latency_ms` | Retrieval latency for the case |

## Variant Comparison Reports

`compare_retrieval_variants.py` evaluates the same cases through:

| Variant | Meaning |
| --- | --- |
| `semantic_only` | Dense Qdrant vector search with the configured embedding model. |
| `bm25_only` | Lexical BM25 over the same PostgreSQL chunk corpus. |
| `hybrid_rrf` | Dense and BM25 candidates fused with reciprocal rank fusion. |

Outputs can include:

- JSON report with case-level and summary metrics,
- CSV summary rows,
- Mermaid `xychart-beta` chart code.

Use these reports to show why TeacherLM keeps both dense and lexical retrieval instead of relying on one method.

## What To Check By Technique

### Dense Semantic Search

Look for cases where the correct section is found even when the query uses different wording from the source. Dense retrieval should be strong for paraphrases, broad conceptual questions, and multilingual wording.

If dense retrieval seems weak, compare embedding models with `benchmark_embeddings.py` and check whether the query needs exact terms better handled by BM25.

### BM25 Keyword Search

Use cases with acronyms, formulas, section titles, named methods, variables, or generated student-style questions. BM25 should recover chunks that contain exact terminology, especially when dense search over-generalizes.

If BM25 looks weak, inspect chunk metadata for:

- `heading_path`,
- `section_title`,
- `key_concepts`,
- `generated_questions`.

### RRF Fusion

RRF should improve recall when dense and BM25 each find different useful candidates. In reports, compare retrieved IDs and ranking movement when toggling retrieval settings locally.

Good signs:

- exact-term chunks and semantic chunks both appear,
- candidates present in both dense and BM25 rise toward the top,
- raw score scale differences do not dominate ranking.

### Reranking

Reranking should improve top-k precision and MRR. It is especially useful when the fused candidate set has the right chunk somewhere in the top 50 but not near the top.

Check:

- `mrr@10`,
- `ndcg@10`,
- first relevant chunk position,
- whether comparison queries still include both terms after reranking.

### Graph Search

Graph search should help when the query names a concept but the best evidence is connected through objectives, examples, formulas, prerequisites, or related chunks.

Look for chunks tagged in metadata with:

```json
{"retrieval_via": "knowledge_graph"}
```

Graph context expansion may also add chunks with:

```json
{"context_type": "knowledge_graph_neighbor"}
```

### Source-File Filtering

Run the same query with different frontend source selections, or add temporary script instrumentation if needed. The expected behavior is that documents, sections, chunks, graph candidates, equations, tables, and context expansion all stay inside selected ready source files.

CourseBuilder is intentionally outside source-file selection and should be evaluated as a full-course flow.

### Output-Specific Context

Use `eval_course_context.py` to compare context shape:

- `text`: focused chunks and local expansion.
- `quiz`: broad outline/representative sections/equations/tables when no topic is supplied.
- `podcast`: outline, narrative flow, representative sections.
- `mindmap`: course outline and module packs.
- `presentation`: reserved disabled policy, topic clusters plus equations/tables.
- `chart`: reserved disabled policy, relationship-dense context.

The goal is not one universal top-k. Each output needs a different evidence shape.

## Retrieval Defaults

Current platform defaults:

| Setting | Default |
| --- | ---: |
| `RETRIEVAL_TOP_K` | 16 |
| `RETRIEVAL_RERANK_TOP_K` | 16 |
| `RETRIEVAL_DENSE_CANDIDATE_K` | 80 |
| `RETRIEVAL_SPARSE_CANDIDATE_K` | 80 |
| `RETRIEVAL_RERANK_CANDIDATE_K` | 50 |
| `RETRIEVAL_RERANKER_MODEL` | `BAAI/bge-reranker-base` |

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

`report`, `presentation`, and `chart` are disabled registry entries in the current stack. Their retrieval policies are documented here because the backend has reserved output-type mappings for them.
