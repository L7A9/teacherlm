# Tests, Evaluations, Reports, And Generated Artifacts

This file documents the non-runtime support material: test suites, retrieval evaluations, report charts, notebooks, generated presentations, and top-level report files.

## Root Pytest Configuration

`pytest.ini` defines the root test discovery paths:

- `packages/teacherlm_core/tests`
- `generators/teacher_gen/tests`
- `generators/mindmap_gen/tests`
- `platform/backend/tests`

Important note: this checkout also contains tests under `generators/quiz_gen/tests` and `generators/podcast_gen/tests`, but those directories are not listed in the current root `pytest.ini`.

## Shared Core Tests

| Path | Purpose |
| --- | --- |
| `packages/teacherlm_core/tests/test_bm25_generated_questions.py` | Verifies BM25 indexing/search can use generated question metadata. |
| `packages/teacherlm_core/tests/test_hybrid_retriever.py` | Verifies hybrid retriever behavior and dense/sparse fusion logic. |
| `packages/teacherlm_core/tests/test_llm_runtime.py` | Verifies per-request LLM runtime option resolution. |
| `packages/teacherlm_core/tests/test_retrieval_evaluation.py` | Verifies retrieval metrics and summary helpers. |

## Backend Tests

Backend tests live in `platform/backend/tests/`.

| Test file | Area |
| --- | --- |
| `test_app_imports_without_optional_services.py` | App import safety when optional services/models are absent. |
| `test_cors_config.py` | CORS settings behavior. |
| `test_file_schema.py` | Uploaded file schema behavior. |
| `test_file_retry.py` | Retry behavior for failed files. |
| `test_document_cleaning_service.py` | Parser markdown cleaning. |
| `test_chunking_service.py` | Chunk creation, metadata, and chunk boundaries. |
| `test_chunk_question_generator.py` | Generated question annotations for chunks. |
| `test_course_structure_service.py` | Course document/section extraction. |
| `test_course_intake_normalizer.py` | Course intake/source metadata normalization. |
| `test_retrieval_orchestrator_config.py` | Retrieval orchestrator configuration and optional components. |
| `test_comparison_retrieval_terms.py` | Comparison query term handling. |
| `test_course_context_policy.py` | Course context selection policy. |
| `test_mindmap_context_policy.py` | Mindmap context policy. |
| `test_concept_inventory_service.py` | Concept extraction, fallback, and persistence behavior. |
| `test_learning_map_service.py` | Learning phase/objective generation. |
| `test_knowledge_graph_service.py` | Knowledge graph generation and remediation behavior. |
| `test_knowledge_assessment_service.py` | Knowledge check generation/grading behavior. |
| `test_learner_tracker_canonical.py` | Canonical learner state updates. |
| `test_review_test_service.py` | Review window and review test behavior. |
| `test_course_player_service.py` | Older course player behavior. |
| `test_coursebuilder_service.py` | Course builder service behavior. |
| `test_coursebuilder_router.py` | Course builder route behavior. |
| `test_practical_eval_catalog.py` | Practical evaluation catalog validity. |
| `test_runtime_settings_service.py` | Runtime settings storage, masking, and resolution. |

## Generator Tests

### Teacher

| Path | Purpose |
| --- | --- |
| `generators/teacher_gen/tests/test_course_overview_detection.py` | Verifies course overview detection logic. |
| `generators/teacher_gen/tests/test_llm_fallback.py` | Verifies fallback from cloud LLM errors to local Ollama. |
| `generators/teacher_gen/tests/test_reranking_config.py` | Verifies teacher generator reranking/config behavior. |

### Quiz

| Path | Purpose |
| --- | --- |
| `generators/quiz_gen/tests/test_concept_extractor_fallback.py` | Verifies quiz concept fallback extraction. |
| `generators/quiz_gen/tests/test_quiz_type_options.py` | Verifies quiz question type option handling. |

These tests exist but are not included in the root `pytest.ini` discovery list at the time of this documentation.

### Podcast

| Path | Purpose |
| --- | --- |
| `generators/podcast_gen/tests/test_grounding_guards.py` | Verifies no-material and grounding guard behavior. |

This test exists but is not included in the root `pytest.ini` discovery list at the time of this documentation.

### Mindmap

| Path | Purpose |
| --- | --- |
| `generators/mindmap_gen/tests/test_module_batches.py` | Verifies module-pack and batch mindmap behavior. |

## Retrieval Evaluation Assets

Retrieval evaluation assets live under `platform/backend/evals/`.

Important files:

| Path | Purpose |
| --- | --- |
| `README.md` | Existing guide for retrieval eval format, commands, metrics, and interpretation. |
| `example_retrieval_eval.json` | Example retrieval evaluation data. |
| `sample_retrieval_eval.json` | Sample eval dataset. |
| `practical_student_questions.json` | Practical student-style question catalog. |
| `recsys_course_baseline.json` | Baseline course evaluation data. |
| `user_course_retrieval_eval.json` | User-course retrieval eval dataset/output. |
| `user_course_partial_rag_report.json` | Partial RAG report data. |
| `user_course_eval_summary.md` | Human-readable summary for user-course evals. |
| `current_mobile_rag_eval.json` | Mobile course RAG eval result/data. |
| `current_mobile_exact_rag_eval.json` | Exact mobile course RAG eval result/data. |

Course text extracts:

- `course_pdf_text_extracts/Guide_for_Students.txt`
- `course_pdf_text_extracts/Lecture_01_organized.txt`
- `course_pdf_text_extracts/Lecture_02.txt`
- `course_pdf_text_extracts/Lecture_03_V2_organized.txt`
- `course_pdf_text_extracts/Lecture_04_V2.txt`
- `course_pdf_text_extracts/Lecture_05.txt`

These extracted texts are used to build or validate practical retrieval cases.

## Retrieval Evaluation Scripts

Scripts live under `platform/backend/scripts/` and `platform/backend/evals/`.

| Path | Purpose |
| --- | --- |
| `scripts/eval_retrieval.py` | Retrieval evaluation runner. |
| `scripts/evaluate_retrieval.py` | Additional retrieval evaluation entry point. |
| `scripts/compare_retrieval_variants.py` | Compares retrieval variants and writes CSV/JSON/chart data. |
| `scripts/eval_course_context.py` | Evaluates course context behavior. |
| `scripts/benchmark_embeddings.py` | Embedding benchmark helper. |
| `scripts/reindex_from_parsed.py` | Reindex helper from parsed content. |
| `evals/build_teacherlm_report_charts_notebook.py` | Builds or executes chart notebook assets for reports. |

`run.sh report-charts` wraps the compare and chart-generation flow inside the backend container.

## Retrieval Evaluation Outputs

`platform/backend/evals/` contains generated output files:

| File | Meaning |
| --- | --- |
| `retrieval_variant_comparison.csv` | Comparison table for retrieval variants. |
| `retrieval_variant_comparison.json` | JSON version of comparison results. |
| `retrieval_variant_bar_chart.png` | Bar chart for retrieval variant comparison. |
| `retrieval_variant_chart.mmd` | Mermaid source for retrieval variant chart. |
| `retrieval_variant_exact_comparison.csv` | Exact-match comparison table. |
| `retrieval_variant_exact_comparison.json` | JSON exact-match comparison results. |
| `retrieval_variant_exact_bar_chart.png` | Exact-match bar chart. |
| `retrieval_variant_exact_chart.mmd` | Mermaid source for exact-match chart. |
| `teacherlm_report_charts.ipynb` | Notebook for report chart generation. |

## Report Chart Assets

`platform/backend/evals/report_charts/` contains report-ready `.png` and `.svg` files:

- `01_exact_retrieval_benchmark`
- `02_coursebuilder_timeline`
- `03_ingestion_pipeline_estimated`
- `04_generator_quality_rubric`
- `05_coursebuilder_file_coverage`
- `06_formula_acronym_topk`
- `07_source_selection_distribution`
- `database_architecture`

These assets support written reports and presentations about the system.

## Root Presentation Artifacts

`artifacts/` contains generated presentation/deck assets.

Common filename patterns:

- `presentation_<id>.json`
- `presentation_<id>.html`
- `presentation_<id>.md`
- `presentation_<id>.pptx`

`artifacts/deck_compare/` contains:

- Individual slide PNGs named like `notebook_slide_01.png`.
- A contact sheet image named `notebook_contact_sheet.jpg`.

These are not part of the active generator runtime. They are generated deliverables or comparison assets.

## Top-Level LaTeX Reports And Presentations

The root has LaTeX files:

- `final_report.tex`
- `presentation.tex`
- `presentation5.tex`
- `presentation_4.tex`
- `repport.tex`
- `teacherlm_report.tex`
- `teacherlm_report_merge.tex`
- `teacherlm_report_part1.tex`
- `teacherlm_report_part2.tex`
- `teacherlm_report_plan.tex`

They appear to document or present the TeacherLM project. They should be treated as report artifacts unless a future task explicitly asks to compile or edit them.

## Existing Project Documentation

Existing docs that should be read before deep changes:

| Path | Covers |
| --- | --- |
| `README.md` | Whole-project overview and main development commands. |
| `platform/README.md` | Platform architecture, API, frontend/backend responsibilities, and operational notes. |
| `platform/backend/docs/rag_course_content.md` | Detailed RAG/course-content architecture. |
| `platform/backend/evals/README.md` | Retrieval evaluation format and workflow. |
| `packages/teacherlm_core/README.md` | Shared core package overview. |
| `generators/teacher_gen/README.md` | Teacher generator behavior. |
| `generators/quiz_gen/README.md` | Quiz generator behavior. |
| `generators/podcast_gen/README.md` | Podcast generator behavior. |
| `generators/mindmap_gen/README.md` | Mindmap generator behavior. |

## Recommended Test Commands

From the repository root:

```bash
pytest
```

To include generator tests not listed in root `pytest.ini`, run them directly:

```bash
pytest generators/quiz_gen/tests generators/podcast_gen/tests
```

Backend-only:

```bash
pytest platform/backend/tests
```

Core-only:

```bash
pytest packages/teacherlm_core/tests
```

Frontend checks depend on installed Node modules:

```bash
cd platform/frontend
npm run lint
npm run build
```

The exact frontend scripts should be confirmed in `platform/frontend/package.json` before running because scripts can change.

## Evaluation Mental Model

The normal test suite checks code behavior. The retrieval eval assets check answer support quality.

For retrieval changes, useful questions are:

- Did the expected source document appear?
- Did the expected chunk appear near the top?
- Did reranking improve or harm exact source recovery?
- Did source-file filtering behave correctly?
- Did formula/acronym queries retrieve the right chunks?
- Did broad modes preserve enough document coverage?
- Did latency remain acceptable?

For generator changes, useful questions are:

- Did the output stay grounded?
- Did sources match the generated claims?
- Did the artifact render in the frontend?
- Did learner updates make sense?
- Did the generator fail gracefully when context was insufficient?

## What Not To Confuse

- `artifacts/` at the root is generated presentation material, not the MinIO artifact store.
- `platform/backend/evals/` contains evaluation and report assets, not production runtime data.
- Disabled registry entries are product concepts, not implemented services in this checkout.
- Frontend optimistic learner progress is not the canonical learner state.
- Qdrant vectors are derived data. PostgreSQL chunks and MinIO parsed/cleaned text are the main rebuild sources.
