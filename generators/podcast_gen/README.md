# podcast_gen

`podcast_gen` turns course context into a two-host educational podcast. It builds a narrative arc from the retrieved material, writes a transcript, and synthesizes audio when a local TTS backend is available.

## Service

Default port: `8007`

Endpoints:

- `GET /health`
- `GET /info`
- `POST /run`

`/run` streams server-sent events.

Common events:

- `progress`: planning, scripting, TTS, and upload stages.
- `artifact`: audio or transcript artifact metadata.
- `done`: final generator output.
- `error`: failure details.

## Generator Info

| Field | Value |
| --- | --- |
| Generator id | `podcast_gen` |
| Output type | `podcast` |
| Retrieval mode | `narrative_arc` |

The backend retrieves intro, key-point, and concluding context before calling this generator. Source-file selection is applied by the backend before context reaches the generator.

## Options

Supported request options include:

| Option | Notes |
| --- | --- |
| `duration` | `short`, `medium`, or `long`; default `medium` |
| `topic` | Optional focus topic |
| `language` | Optional output language |
| `host_a_name` | Optional first host name |
| `host_b_name` | Optional second host name |
| voice options | Optional backend-specific voice overrides |

Duration presets:

| Duration | Target words | Approximate length |
| --- | ---: | ---: |
| `short` | 600 | 4 minutes |
| `medium` | 1400 | 9 minutes |
| `long` | 2500 | 16 minutes |

## TTS Backends

The generator probes local TTS backends in this order:

1. Piper
2. Kokoro
3. pyttsx3

If no backend is available, the generator still returns a transcript artifact and marks audio synthesis as unavailable in metadata.

## Output

The final output can include:

- An MP3 audio artifact when synthesis succeeds.
- A transcript text artifact.
- Narrative arc metadata.
- Duration, language, voice, and host metadata.
- Learner updates for covered concepts.

Artifacts are uploaded through MinIO when storage configuration is available.

## Environment

Common environment variables:

| Variable | Purpose |
| --- | --- |
| `PODCAST_GEN_OLLAMA_URL` | Ollama base URL override |
| `PODCAST_GEN_MODEL` | script-generation model |
| `PODCAST_GEN_DEFAULT_DURATION` | default duration preset |
| `PODCAST_GEN_ARTIFACTS_DIR` | local artifact directory |
| `PODCAST_GEN_PUBLIC_BASE_URL` | public artifact base URL |
| `PODCAST_GEN_TTS_BACKEND` | force or prefer a TTS backend |
| `PODCAST_GEN_PIPER_MODELS_DIR` | Piper model directory |
| `PODCAST_GEN_DISABLE_MODEL_DOWNLOAD` | disable automatic Piper model downloads |
| `MINIO_ENDPOINT` | artifact storage endpoint |
| `MINIO_ACCESS_KEY` | artifact storage access key |
| `MINIO_SECRET_KEY` | artifact storage secret key |
| `MINIO_BUCKET` | artifact bucket |

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
