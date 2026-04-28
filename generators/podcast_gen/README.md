# podcast_gen

Two-host educational podcast generator. Port **8007** • Python 3.14+.

## What it does
1. **Narrative arc** — extract a teaching-order arc (intro → key points → conclusion) from retrieved chunks.
2. **Script** — generate a two-host script via `ollama format=PodcastScript`:
   - **Host A** is the curious student-style host (asks, echoes, relates to analogies).
   - **Host B** is the teacher (explains, refers to source material naturally).
   - Word target by `options.duration`: `short=600`, `medium=1400`, `long=2500`.
3. **TTS** — synthesise each segment via the first backend that's available:
   - **Piper** (preferred) — two distinct neural voices per supported
     language, including French, Spanish, Italian, German, Portuguese.
     Voice files (~25–60 MB each) are auto-downloaded on first use.
   - **Kokoro** — fallback for languages Piper doesn't cover (ja, cmn, hi).
   - **pyttsx3** — last-resort offline single voice via espeak-ng.
   Falls back to a transcript-only artifact if all three are unavailable.
   After the LLM emits the script, a langdetect post-pass re-translates
   any segment that drifted from the user-selected language.
4. **Compose** — `pydub` concatenates segments with short inter-segment
   silence, normalises levels, and exports `MP3 @ 128kbps` plus a transcript
   `.txt`.

## Endpoints
- `GET /health` — liveness
- `GET /info` — capabilities (durations, languages, voices, models)
- `POST /run` — accepts `GeneratorInput`, returns SSE stream (`progress`,
  `token`, `done`)

## Options (`GeneratorInput.options`)
| key | type | default | meaning |
|-----|------|---------|---------|
| `duration` | `"short" \| "medium" \| "long"` | `"medium"` | target script length |
| `language` | language code | `"en-us"` | one of the supported languages below |
| `topic` | `str` | `""` | narrows the narrative arc to a focus area |
| `voice_host_a` | kokoro voice id | from language | override Host A voice |
| `voice_host_b` | kokoro voice id | from language | override Host B voice |

### Supported languages

Piper (preferred — two distinct voices per language):

| code | host_a (Piper) | host_b (Piper) |
|---|---|---|
| `en-us` | en_US-amy-medium | en_US-ryan-high |
| `en-gb` | en_GB-alba-medium | en_GB-alan-medium |
| `fr-fr` | fr_FR-siwis-medium | fr_FR-gilles-low |
| `es` | es_ES-davefx-medium | es_ES-sharvard-medium |
| `it` | it_IT-paola-medium | it_IT-riccardo-x_low |
| `pt-br` | pt_BR-faber-medium | pt_BR-edresson-low |
| `de` | de_DE-thorsten-medium | de_DE-karlsson-low |

Kokoro fallback (used when Piper has no voices for the language):

| code | host_a (Kokoro) | host_b (Kokoro) |
|---|---|---|
| `ja` | jf_alpha | jm_kumo |
| `cmn` | zf_xiaoxiao | zm_yunjian |
| `hi` | hf_alpha | hm_omega |

## Output artifacts
- `audio/mpeg` — `<title>.mp3` (skipped if both TTS backends fail)
- `transcript` — `<title>.txt` with speaker labels, always emitted

## Voice model files

### Piper (preferred, ~25–60 MB per voice)
Voices auto-download from HuggingFace (`rhasspy/piper-voices`) on first
use into `generators/podcast_gen/models/piper/`. Set
`PODCAST_GEN_PIPER_AUTO_DOWNLOAD=false` to disable auto-download — then
drop the `.onnx` + `.onnx.json` files in manually:

```
generators/podcast_gen/models/piper/
├── fr_FR-siwis-medium.onnx
├── fr_FR-siwis-medium.onnx.json
├── fr_FR-gilles-low.onnx
└── fr_FR-gilles-low.onnx.json
```

### Kokoro fallback (~338 MB total)
The kokoro weights are **not** committed to git — `models/*.onnx` and
`*.bin` are in `.gitignore`. Drop them into `generators/podcast_gen/models/`:

```
generators/podcast_gen/models/
├── kokoro-v1.0.onnx        # ~310 MB
└── voices-v1.0.bin         # ~27 MB
```

Quick download (any platform with curl):

```bash
mkdir -p generators/podcast_gen/models && cd generators/podcast_gen/models
curl -fL -o kokoro-v1.0.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx
curl -fL -o voices-v1.0.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin
```

`config.py` resolves model paths relative to the package by default, so the
files work whether you run from the repo root, from this directory, or via
Docker. Without them the generator transparently falls back to pyttsx3
(single voice via espeak-ng) or to transcript-only delivery.

## Run with Docker (recommended)
The compose file bind-mounts `generators/podcast_gen/models/` into the
container, so once the model files are on the host they work in both modes.

```bash
cd platform
docker compose up -d podcast_gen
```

Build only:
```bash
docker build -f generators/podcast_gen/Dockerfile -t teacherlm/podcast_gen:latest .
```

## Run without Docker
Requirements: Python 3.14+, ffmpeg, espeak-ng (only needed if you want the
pyttsx3 fallback to work — kokoro doesn't need it).

```bash
# 1. System deps (per-OS)
#    macOS:    brew install ffmpeg espeak-ng
#    Debian:   sudo apt install ffmpeg espeak-ng libespeak1
#    Windows:  choco install ffmpeg ; pip install pipwin && pipwin install pyaudio
#              (espeak-ng optional — only needed for pyttsx3 fallback)

# 2. Install dependencies (uses local teacherlm_core editable install)
cd <repo-root>
python -m venv .venv && source .venv/bin/activate     # or .venv\Scripts\activate on Windows
pip install -e packages/teacherlm_core
pip install -r generators/podcast_gen/requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"

# 3. Make sure Ollama is running on the host
#    ollama serve  (in another terminal)
#    ollama pull llama3.1:8b-instruct-q4_K_M

# 4. (Optional but recommended) MinIO for artifact uploads, or set
#    PODCAST_GEN_MINIO_* env vars to point at any S3-compatible store.

# 5. Run the service
PYTHONPATH=generators uvicorn podcast_gen.app:app \
    --host 0.0.0.0 --port 8007
```

The model paths default to `generators/podcast_gen/models/{kokoro-v1.0.onnx,voices-v1.0.bin}`.
Override via `PODCAST_GEN_KOKORO_MODEL_PATH` / `PODCAST_GEN_KOKORO_VOICES_PATH`
if you keep them somewhere else.

## Retrieval mode
Expects the platform to use the `narrative_arc` retrieval mode (intro +
key points + conclusion sampling) — see root `CLAUDE.md`.
