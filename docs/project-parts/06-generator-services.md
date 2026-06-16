# Generator Services

Generator services are independent FastAPI applications under `generators/`. The backend retrieves context, builds a `GeneratorInput`, sends it to a generator `/run` endpoint, and streams the generator's response back to the frontend.

The enabled generators in this checkout are:

- `teacher_gen`
- `quiz_gen`
- `podcast_gen`
- `mindmap_gen`

The registry also names report, presentation, and chart generators, but those are disabled and their directories are not present in this checkout.

## Shared Generator Shape

Common files in each implemented generator:

| File | Purpose |
| --- | --- |
| `app.py` | FastAPI app. Usually exposes `/health`, `/info`, and `/run`. |
| `config.py` | Pydantic settings for the service. |
| `pipeline.py` | Main generation pipeline. |
| `schemas.py` | Generator-specific structured output models. |
| `services/` | Supporting service modules. |
| `prompts/` | Prompt files used by the pipeline. |
| `requirements.txt` | Generator dependencies. |
| `Dockerfile` | Container image for the generator. |
| `README.md` | Generator-specific documentation. |
| `tests/` | Generator-specific tests where present. |

Common endpoint behavior:

- `/health`: health check.
- `/info`: generator metadata.
- `/run`: accepts `GeneratorInput` and returns streamed SSE events.

Common SSE event types:

- `progress`: status updates for long operations.
- `token` or `chunk`: streamed natural language output.
- `sources`: source chunks used.
- `artifact`: generated artifact metadata.
- `done`: final `GeneratorOutput` or completion metadata.
- `error`: failure event.

All generators should preserve source grounding and avoid inventing facts outside `context_chunks`.

## Teacher Generator

Location: `generators/teacher_gen`

Registry:

- ID: `teacher_gen`
- Output type: `text`
- Port: 8001
- Retrieval mode: `semantic_topk`
- Role: default chat tutor

### Files

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI endpoints for health, info, and run. |
| `config.py` | `TEACHER_GEN_` settings, model choices, temperatures, relevance threshold, confusion threshold, and stuck-turn threshold. |
| `pipeline.py` | Main chat tutor pipeline. Analyzes query, picks response mode, handles formula/course-overview deterministic paths, streams answer, scores confidence, extracts learner updates, and emits final output. |
| `schemas.py` | Query analysis, concept extraction, and response mode schemas. |
| `services/query_analyzer.py` | Structured LLM query analysis with heuristic fallback. Detects intent, confusion, target concept, and direct-answer need. |
| `services/response_mode.py` | Chooses explain, guide, quiz-back, or affirm mode from query analysis and learner state. |
| `services/llm_service.py` | Loads shared and local prompts, builds system prompts, creates role-specific LLM clients, and falls back to local Ollama on cloud rate-limit errors. |
| `services/confidence_scorer.py` | Combines groundedness and coverage into overall confidence labels. |
| `services/learner_analyzer.py` | Extracts covered, demonstrated, and confused concepts from the conversation turn. |
| `prompts/mode_explain.txt` | Grounded explanatory tutor mode. |
| `prompts/mode_guide.txt` | Socratic guidance mode. |
| `prompts/mode_quiz_back.txt` | Checks understanding with a question. |
| `prompts/mode_affirm.txt` | Confirms correct understanding and nudges forward. |
| `prompts/query_analysis.txt` | Structured query analysis prompt. |
| `prompts/learner_update_extraction.txt` | Learner update extraction prompt. |
| `tests/test_course_overview_detection.py` | Course overview heuristic tests. |
| `tests/test_llm_fallback.py` | Cloud-to-local fallback tests. |
| `tests/test_reranking_config.py` | Reranking/config behavior tests. |

### Pipeline Details

1. Receives backend-selected context chunks.
2. Filters or formats context.
3. Detects formula-specific questions.
4. Detects course-overview questions.
5. Runs query analysis.
6. Chooses response mode.
7. If no evidence exists, gives a grounded refusal or asks for relevant material.
8. Streams LLM answer or deterministic formula/overview answer.
9. Emits sources.
10. Scores confidence:
    - groundedness weight: 0.7
    - coverage weight: 0.3
11. Extracts learner updates.
12. Emits final metadata including mode, analysis, confidence, context/ranking metadata, and fallback indicators.

### Response Modes

| Mode | Use |
| --- | --- |
| `explain` | Student needs a direct grounded explanation. |
| `guide` | Student appears confused and benefits from a guided path. |
| `quiz_back` | Student may need a quick check for understanding. |
| `affirm` | Student demonstrates understanding and needs confirmation or extension. |

## Quiz Generator

Location: `generators/quiz_gen`

Registry:

- ID: `quiz_gen`
- Output type: `quiz`
- Port: 8002
- Retrieval mode: `coverage_broad`
- Role: grounded quiz artifact generator

### Files

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI endpoints. |
| `config.py` | `QUIZ_GEN_` settings, model choices, question count limits, difficulty mix ratios, distractor-engine settings, embedding model, and MinIO config. |
| `pipeline.py` | Main quiz generation pipeline. Resolves options, extracts concepts, plans slots, generates questions, validates, tops up, stores artifact, and emits final output. |
| `schemas.py` | Bloom levels, question kinds, MCQ/true-false/fill-blank schemas, quiz output, concept cards, question slots, and quiz plan. |
| `services/concept_extractor.py` | Extracts quiz concepts from chunks through LLM structured output and deterministic fallback. |
| `services/difficulty_adapter.py` | Builds adaptive question plan using learner state, struggling concepts, understood concepts, coverage, and stretch targets. |
| `services/question_generator.py` | Generates each question slot from a selected source chunk and normalizes result metadata. |
| `services/quality_validator.py` | Validates questions, removes weak/generic/ambiguous/duplicate items, and computes Bloom distribution. |
| `services/distractor_engine.py` | Optional fastembed hard-negative distractor selection for MCQs. |
| `services/llm_service.py` | LLM clients for chat, extraction, and generation roles. |
| `services/artifact_store.py` | Stores quiz JSON in MinIO and returns artifact metadata. |
| `prompts/concept_extraction.txt` | Concept extraction prompt. |
| `prompts/mcq_generation.txt` | Multiple-choice generation prompt. |
| `prompts/true_false_generation.txt` | True/false generation prompt. |
| `prompts/fill_blank_generation.txt` | Fill-blank generation prompt. |
| `prompts/adaptive_guidance.txt` | Adaptive quiz guidance prompt. |
| `tests/test_concept_extractor_fallback.py` | Fallback concept extraction tests. |
| `tests/test_quiz_type_options.py` | Quiz question type option tests. |

### Pipeline Details

1. Reads requested question count, difficulty, and question types.
2. Normalizes question type aliases:
   - `multiple_choice` to `mcq`
   - true/false aliases to `true_false`
3. Extracts candidate concepts from context chunks.
4. Builds a concept-to-source-chunk map.
5. Plans question slots based on difficulty and learner state.
6. Generates questions slot by slot.
7. Optionally enhances MCQ distractors.
8. Repairs labels, concept IDs, and source IDs.
9. Validates questions for grounding and quality.
10. Deduplicates weak repeats.
11. Tops up with fallback grounded questions if needed.
12. Saves quiz JSON artifact to MinIO.
13. Emits a response with artifact link and quiz metadata.

### Adaptive Mix

The default mix is configured around:

- Struggling concepts.
- Broad course coverage.
- Stretch material.

The goal is not just random testing. It should help the learner practice weak areas while still covering the course.

## Podcast Generator

Location: `generators/podcast_gen`

Registry:

- ID: `podcast_gen`
- Output type: `podcast`
- Port: 8007
- Retrieval mode: `narrative_arc`
- Role: grounded podcast audio/transcript generator

### Files

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI endpoints. |
| `config.py` | `PODCAST_GEN_` settings, duration targets, key point counts, TTS paths, voices, language voice mappings, host names, silence, bitrate, and MinIO settings. |
| `pipeline.py` | Main podcast pipeline. Resolves options, filters context, extracts narrative arc, generates script, synthesizes TTS, composes audio, uploads artifacts, and emits metadata. |
| `schemas.py` | Narrative arc, segments, script, TTS result, and podcast bundle schemas. |
| `services/grounding_guard.py` | Filters usable chunks and detects no-material scripts. |
| `services/narrative_extractor.py` | Builds a narrative arc through LLM structured output and fallback. |
| `services/script_generator.py` | Generates educational podcast script sections and fallback scripts. |
| `services/tts_service.py` | Chooses and runs Piper, Kokoro, or pyttsx3 TTS. Handles voice plans, downloads, and fallback. |
| `services/audio_composer.py` | Builds transcript, inserts silence, normalizes audio, and exports MP3 bytes. |
| `services/text_sanitizer.py` | Removes source markers, chunk IDs, and other non-spoken text from narration. |
| `services/artifact_store.py` | Stores audio and transcript artifacts in MinIO. |
| `services/llm_service.py` | Podcast LLM client wrapper. |
| `prompts/narrative_arc.txt` | Narrative arc extraction prompt. |
| `prompts/script_educational.txt` | Full educational podcast script prompt. |
| `prompts/script_section.txt` | Section-level script prompt. |
| `tests/test_grounding_guards.py` | Grounding/no-material guard tests. |

### Pipeline Details

1. Resolve duration, topic, language, host names, and voice plan.
2. Filter unusable context chunks.
3. Extract narrative arc:
   - hook/introduction
   - key points
   - transitions
   - conclusion
4. Generate script.
5. Retry when a model incorrectly claims there is no material despite context.
6. Enforce requested language where possible.
7. Sanitize spoken text.
8. Build transcript.
9. Synthesize audio.
10. Compose MP3.
11. Upload MP3 and transcript to MinIO.
12. Emit artifacts and metadata.

### TTS Backends

Priority:

1. Piper
2. Kokoro
3. pyttsx3

The transcript can still be produced if audio synthesis fails, as long as artifact storage succeeds.

## Mindmap Generator

Location: `generators/mindmap_gen`

Registry:

- ID: `mindmap_gen`
- Output type: `mindmap`
- Port: 8008
- Retrieval mode: `topic_clusters`
- Role: visual study map generator

### Files

| Path | Purpose |
| --- | --- |
| `app.py` | FastAPI endpoints and static artifact serving under `/artifacts`. |
| `config.py` | Ollama/model settings, artifact directory, public base URL, default size, max node count, LLM timeout, and keepalive settings. |
| `pipeline.py` | Main mindmap pipeline. Selects size/layout, uses module packs or LLM batch outlines, synthesizes hierarchy, refines, balances, compiles markdown, renders JSON/HTML artifacts, and streams progress. |
| `schemas.py` | Recursive mind map node, complete mind map, theme list, course outline, and subtopic expansion schemas. |
| `services/theme_extractor.py` | Extracts main themes from chunks. |
| `services/hierarchy_builder.py` | Selects theme-overlap chunks and expands branches. |
| `services/course_structure.py` | Builds maps from parsed course headings and module packs. |
| `services/llm_service.py` | Structured LLM calls for theme extraction, course outline, batch outline, synthesis, subtopic expansion, central topic inference, and label translation. |
| `services/balancer.py` | Promotes singleton chains, caps children, trims leaves, and enforces max node counts. |
| `services/markdown_compiler.py` | Converts hierarchy into Markmap-compatible markdown. |
| `services/mermaid_compiler.py` | Converts hierarchy into Mermaid mindmap syntax. |
| `services/html_renderer.py` | Writes JSON payload and standalone HTML using Jinja2. |
| `templates/mindmap.html.jinja` | Standalone mindmap artifact template. |
| `prompts/theme_extraction.txt` | Main theme extraction prompt. |
| `prompts/course_outline.txt` | Source-structured course outline prompt. |
| `prompts/batch_outline.txt` | Batch outline prompt for many chunks. |
| `prompts/course_synthesis.txt` | Course synthesis prompt. |
| `prompts/subtopic_expansion.txt` | Subtopic expansion prompt. |
| `tests/test_module_batches.py` | Module-pack/batch behavior tests. |

### Pipeline Details

1. Resolve map size:
   - concise
   - standard
   - comprehensive
2. Choose a layout style.
3. If backend supplied module packs and no forced refinement is requested, use the fast module-pack path.
4. Otherwise extract themes and/or course outline with LLM.
5. Expand branches in batches.
6. Synthesize a course-level hierarchy.
7. Fall back to parsed course structure when LLM is unavailable.
8. Optionally adapt labels to requested language.
9. Balance the tree and trim to configured node count.
10. Compile Markmap markdown.
11. Render JSON and standalone HTML artifacts.
12. Emit artifact metadata and learner updates from node labels.

### Artifact Serving

Mindmap artifacts are served by the generator itself, not necessarily by MinIO.

The service mounts a static artifacts directory. The frontend can open the generated URL and render it in `MindmapRenderer`.

## Disabled Registry Entries

The registry includes these disabled output types:

| ID | Output type | State |
| --- | --- | --- |
| `report_gen` | `report` | Disabled. No directory in this checkout. |
| `presentation_gen` | `presentation` | Disabled. No directory in this checkout. |
| `chart_gen` | `chart` | Disabled. No directory in this checkout. |

The frontend should not offer disabled entries as usable actions. The backend registry can list them only when explicitly requested with disabled entries included.

## Generator Boundaries

Generators should:

- Accept canonical `GeneratorInput`.
- Return canonical `GeneratorOutput`.
- Stream useful progress for long-running work.
- Include source information.
- Return learner updates.
- Store artifacts where appropriate.
- Use shared core schemas and LLM utilities.
- Stay grounded in provided chunks.

Generators should not:

- Query the database directly.
- Decide which uploaded files are selected.
- Read Qdrant directly for normal generation.
- Mutate learner state directly.
- Invent unsupported course facts.
- Expose user-facing terminology that conflicts with the product language.
