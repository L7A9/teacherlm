# podcast_gen

`podcast_gen` turns retrieved course context into a two-host educational podcast. It builds a narrative arc, writes a grounded transcript, optionally synthesizes audio, composes an MP3, and always returns a transcript artifact when storage is available.

The result is designed as a listen-along lesson: one host asks student-style questions, the other explains from the uploaded material.

## Service

Default port: `8007`

Endpoints:

- `GET /health`
- `GET /info`
- `POST /run`

`/run` streams server-sent events.

Common events:

- `progress`: planning, scripting, language check, TTS, composition, upload stages.
- `token`: final short response text.
- `done`: final `GeneratorOutput`.
- `error`: failure details from the shared safe SSE wrapper.

Audio and transcript artifact metadata are included in the final `done` payload when storage succeeds.

## Generator Info

| Field | Value |
| --- | --- |
| Generator id | `podcast_gen` |
| Output type | `podcast` |
| Retrieval mode | `narrative_arc` |
| Exports | MP3 when TTS succeeds, transcript text always when artifact storage succeeds |
| TTS backends | Piper, Kokoro, pyttsx3 |

The backend retrieves and filters context before calling this generator. Source-file selection is applied by the backend.

`GET /info` advertises:

- supported duration presets,
- default duration,
- artifact exports,
- supported language tags,
- per-language Piper/Kokoro voice plans,
- preferred backend per language,
- TTS backends,
- chat, extraction, and generation model names.

## Platform Connection

`podcast_gen` connects through the generic generation route:

1. The frontend requests `output_type: "podcast"` through `/api/conversations/{conversation_id}/generate`.
2. The backend resolves the enabled `podcast_gen` registry entry.
3. Source-file selection is applied before retrieval.
4. For no-topic UI requests, the backend sends course outline, narrative-arc retrieval, and representative sections.
5. For API callers that provide a topic, the backend sends topic sections plus focused hits.
6. The backend builds `GeneratorInput` and posts it to `POST /run`.
7. `podcast_gen` streams progress for planning, scripting, language checking, TTS, composition, upload, and final output.
8. The backend persists transcript/audio artifacts, sources, and `concepts_covered`.

The generator never reads uploads directly and never queries Qdrant. It only consumes prepared `context_chunks`.

## Why `narrative_arc`

Audio learning needs a teaching flow. A pure nearest-neighbor context list can produce a jumpy script.

`narrative_arc` fits because the backend can supply:

- introduction-like context,
- key middle concepts,
- conclusion-like context,
- course outline and representative sections for no-topic podcasts,
- focused topic sections when a topic is supplied.

This gives the podcast generator enough structure to sound like a lesson rather than a list of disconnected facts.

## Options

Supported request options include:

| Option | Notes |
| --- | --- |
| `duration` | `short`, `medium`, or `long`; default `medium` |
| `topic` | Optional focus topic, normally supplied by backend request body |
| `language` | Optional output language |
| `host_a_name` | Optional first host name |
| `host_b_name` | Optional second host name |
| `voice_host_a` | Optional voice override |
| `voice_host_b` | Optional voice override |

Duration presets:

| Duration | Target words | Approximate length |
| --- | ---: | ---: |
| `short` | 600 | 4 minutes |
| `medium` | 1400 | 9 minutes |
| `long` | 2500 | 16 minutes |

The frontend currently hides podcast-specific fields in the dialog. UI-triggered podcasts therefore use:

- selected ready source files,
- generator default duration (`medium`),
- default host names (`None`, so hosts do not claim human names),
- default voice plan for the selected or default language,
- forced language from Settings when configured.

The options above remain available to API callers.

## Pipeline

### 1. Receive `GeneratorInput`

The platform sends:

- `context_chunks` from `narrative_arc` policy,
- current learner state,
- chat history,
- runtime LLM/provider options,
- podcast options such as duration, language, host names, and voices.

The generator applies current LLM options and language through `teacherlm_core` helpers.

### 2. Resolve Duration, Language, Hosts, And Voice Plan

`pipeline.py` resolves:

- duration preset,
- topic focus,
- language,
- optional host names,
- voice plan.

`services/tts_service.py` chooses a voice backend and host voices before generation starts, so the first `progress` event can report backend, voices, and whether the selected language uses one shared voice.

Why this exists:

- the script target length depends on duration,
- voice availability differs by language,
- the UI can show meaningful progress before long TTS work begins.

Supported language tags are drawn from the Piper and Kokoro voice maps:

- `en-us`,
- `en-gb`,
- `fr-fr`,
- `es`,
- `it`,
- `pt-br`,
- `de` through Piper,
- `ja`,
- `cmn`,
- `hi`.

### 3. Filter Usable Context

`services/grounding_guard.py` removes unusable or empty context chunks.

If no usable context remains, the generator returns a grounded empty response instead of inventing a podcast.

Why this exists:

- TeacherLM must not generate from missing course evidence,
- parser noise or empty retrieval should be handled honestly.

### 4. Extract Narrative Arc

`services/narrative_extractor.py` produces a structured `NarrativeArc`.

The arc contains:

- title,
- intro,
- key points,
- conclusion,
- source-aware teaching focus.

The extractor receives formatted context for speech and a language hint. If the LLM returns a weak or "no materials" arc despite usable chunks, the service falls back to an arc built directly from source chunks.

Why this step exists:

- scripts need a teachable order,
- a bounded arc gives later script generation a clear plan,
- fallback keeps the podcast grounded even when local models underperform.

### 5. Generate Script Section By Section

`services/script_generator.py` writes a structured two-host `PodcastScript`.

It generates sections independently rather than asking for the whole podcast in one call.

Why section-by-section generation:

- small local models handle bounded word counts better,
- failures are isolated,
- each section can focus on one arc point,
- long podcast scripts are less likely to collapse into empty JSON.

The script style:

- Host A asks learner-style questions,
- Host B explains clearly,
- both hosts stay grounded in uploaded material,
- self-introductions are skipped unless names are supplied,
- each segment is sanitized for speech.

### 6. Grounding Guard Retry

After script generation, `script_claims_no_materials()` checks for the failure mode where the model says no course material was provided even though chunks exist.

If that happens:

1. the generator retries narrative extraction without topic focus,
2. generates the script again,
3. falls back to a deterministic script from the arc if the problem persists.

Why this exists:

- local models can incorrectly refuse inside structured generation,
- the system should use the retrieved evidence it actually has,
- deterministic fallback preserves correctness over style.

### 7. Enforce Language

`enforce_language()` uses `langdetect` when available to find segments that drift from the requested language.

Drifting segments are rewritten by the LLM into the target language and sanitized again.

Why this exists:

- multilingual source material and local models can mix languages,
- TTS output should match the student's selected language,
- language drift is easier to fix before synthesis.

### 8. Build Transcript

`services/audio_composer.py` builds a plain text transcript from the final script.

Transcript generation happens before audio synthesis. The transcript is always uploaded after the TTS block, even when audio is skipped.

Why this exists:

- transcript-only output is still valuable,
- students can skim or search the episode,
- failures in TTS should not erase the teaching content.

### 9. Synthesize Speech

`services/tts_service.py` probes TTS backends in this order:

1. Piper
2. Kokoro
3. pyttsx3

Piper is preferred because supported languages are configured with two distinct neural voices where possible. Kokoro is a strong fallback for languages where configured Piper voices are not available. pyttsx3 is the final offline fallback.

For single-voice languages, the voice plan applies small speed and pitch differences so Host A and Host B remain distinguishable.

Piper voice files can be auto-downloaded from Hugging Face when `PODCAST_GEN_PIPER_AUTO_DOWNLOAD=true`. Kokoro requires local ONNX model and voices files. pyttsx3 depends on system speech support and is treated as a last-resort fallback.

If no backend works, the generator emits a `tts_skipped` progress event and continues with transcript-only output.

### 10. Compose MP3

`services/audio_composer.py` uses pydub and ffmpeg-compatible export to:

- add short silences between segments,
- normalize segment audio,
- concatenate segments,
- export MP3 bytes.

The MP3 is uploaded through MinIO artifact storage as an `audio` artifact.

### 11. Upload Transcript

The transcript is uploaded as a `transcript` artifact:

```json
{
  "type": "transcript",
  "url": "...",
  "filename": "episode-title.txt",
  "key": "conversations/.../artifacts/..."
}
```

### 12. Return Learner Updates

The generator reports `concepts_covered` from narrative arc key points. The backend merges those into learner state after `done`.

## Output

The final response includes:

- short markdown response text,
- audio artifact when synthesis succeeds,
- transcript artifact,
- source chunks used,
- learner updates,
- `metadata.podcast`,
- `metadata.narrative_arc`,
- duration choice,
- language,
- voice/backend metadata,
- host-name metadata.

Metadata flags:

- `used_fallback_tts`: true when pyttsx3 fallback was used,
- `tts_skipped`: true when no TTS backend was available,
- `single_voice`: true when both hosts use one base voice with variation.

## Technology Choices

| Technology | Why it is used |
| --- | --- |
| FastAPI | Independent generator service with health/info/run endpoints |
| SSE | Long-running script/TTS work needs visible progress |
| Pydantic V2 | Validates narrative arc, script, segments, and final metadata |
| Ollama native `format=` | Structured local extraction and script generation |
| `teacherlm_core` | Shared generator contract, prompts, LLM runtime |
| Piper | Preferred multilingual neural TTS with distinct host voices where configured |
| Kokoro | Neural fallback for languages/voices not covered well by Piper |
| pyttsx3 | Last-resort offline TTS fallback |
| langdetect | Detects language drift before synthesis |
| pydub | Normalizes, spaces, concatenates, and exports audio |
| MinIO | Stores MP3 and transcript artifacts |
| Backend-owned `narrative_arc` retrieval | Supplies lesson-like flow and source-file filtering |
| Section-by-section scripting | Keeps local structured generation reliable for long episodes |
| Speech text sanitizer | Removes chunk IDs, source-number labels, and placeholder host names before TTS |
| Single-voice pitch/speed variation | Makes hosts distinguishable when a language has only one usable voice |

## Environment

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `PODCAST_GEN_OLLAMA_HOST` | Ollama base URL override |
| `PODCAST_GEN_CHAT_MODEL` | chat/support model |
| `PODCAST_GEN_EXTRACTION_MODEL` | narrative arc model |
| `PODCAST_GEN_GENERATION_MODEL` | script generation model |
| `PODCAST_GEN_CHAT_TEMPERATURE` | chat/support model temperature |
| `PODCAST_GEN_EXTRACTION_TEMPERATURE` | narrative extraction temperature |
| `PODCAST_GEN_GENERATION_TEMPERATURE` | script generation temperature |
| `PODCAST_GEN_DURATION_WORD_TARGETS` | duration preset to word-count mapping |
| `PODCAST_GEN_DEFAULT_DURATION` | default duration preset |
| `PODCAST_GEN_MIN_KEY_POINTS` | minimum narrative key points |
| `PODCAST_GEN_MAX_KEY_POINTS` | maximum narrative key points |
| `PODCAST_GEN_KOKORO_MODEL_PATH` | Kokoro ONNX model path |
| `PODCAST_GEN_KOKORO_VOICES_PATH` | Kokoro voices file path |
| `PODCAST_GEN_PIPER_MODELS_DIR` | Piper voice model directory |
| `PODCAST_GEN_PIPER_VOICE_URL_BASE` | Hugging Face Piper voice asset base URL |
| `PODCAST_GEN_PIPER_AUTO_DOWNLOAD` | allow first-use Piper voice download |
| `PODCAST_GEN_PIPER_LENGTH_SCALE` | Piper speech speed control |
| `PODCAST_GEN_VOICE_HOST_A` | default Host A voice |
| `PODCAST_GEN_VOICE_HOST_B` | default Host B voice |
| `PODCAST_GEN_TTS_SAMPLE_RATE` | target sample rate for TTS-related output |
| `PODCAST_GEN_TTS_SPEED` | base speech speed |
| `PODCAST_GEN_SINGLE_VOICE_SPEED_DELTA` | speed offset when both hosts share one voice |
| `PODCAST_GEN_SINGLE_VOICE_PITCH_A_SEMITONES` | Host A pitch shift for single-voice languages |
| `PODCAST_GEN_SINGLE_VOICE_PITCH_B_SEMITONES` | Host B pitch shift for single-voice languages |
| `PODCAST_GEN_INTER_SEGMENT_SILENCE_MS` | silence between speech segments |
| `PODCAST_GEN_INTRO_OUTRO_SILENCE_MS` | silence around intro/outro |
| `PODCAST_GEN_MP3_BITRATE` | MP3 export bitrate |
| `PODCAST_GEN_ARTIFACTS_DIR` | local artifact directory fallback |
| `PODCAST_GEN_HOST_A_NAME` | optional default Host A name |
| `PODCAST_GEN_HOST_B_NAME` | optional default Host B name |
| `PODCAST_GEN_MINIO_*` | artifact storage configuration |
| `PODCAST_GEN_ARTIFACT_URL_TTL_S` | presigned artifact URL lifetime |
| `PODCAST_GEN_REQUEST_TIMEOUT_S` | request timeout setting |
| `OLLAMA_HOST` | shared Ollama fallback host |
| `OLLAMA_CHAT_MODEL` | shared model fallback |

## Docker Notes

The Dockerfile:

- builds from the repository root,
- installs Python `3.14-slim`,
- installs `ffmpeg`, `espeak-ng`, and `libespeak1` for pydub, pyttsx3, and Piper phonemization,
- installs `teacherlm_core` without heavy optional deps,
- installs Piper, Kokoro, ONNX Runtime, pyttsx3, pydub, audioop-lts, soundfile, NumPy, NLTK, langdetect, and MinIO,
- pre-downloads NLTK `punkt` and `punkt_tab`,
- creates artifact and model directories,
- exposes port `8007`,
- adds a `/health` healthcheck.

## Local Run

From this directory:

```bash
pip install -e ../../packages/teacherlm_core
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8007
```

Through Docker Compose:

```bash
cd ../../platform
docker compose up -d podcast_gen
```

## Tests

From the repository root:

```bash
pytest generators/podcast_gen/tests
```

## Notes

- TTS requires local model files or backend availability. Without TTS, transcript generation still works.
- Piper may auto-download configured voices unless disabled.
- pydub MP3 export requires an ffmpeg-compatible binary in the environment.
