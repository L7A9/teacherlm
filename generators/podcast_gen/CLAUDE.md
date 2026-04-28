# Podcast Generator — CLAUDE.md

## Port: 8007 | Python 3.14+
## Purpose: Two-host educational podcasts

## Stack
- FastAPI, ollama, teacherlm_core
- TTS probe order: piper-tts → kokoro-onnx → pyttsx3 (all 3.14-safe).
  Piper is preferred because every supported language ships TWO distinct
  neural voices (host_a + host_b sound clearly different in French,
  Spanish, German, etc — no pitch-shift hack needed).
- pydub + ffmpeg (audio manipulation)
- nltk (sentence tokenization)
- langdetect (post-script language drift check + retranslate)

## Module Map
podcast_gen/
├── CLAUDE.md, app.py, config.py, schemas.py, pipeline.py
├── services/
│   ├── narrative_extractor.py
│   ├── script_generator.py        # teacher-style two-host script
│   ├── tts_service.py
│   ├── audio_composer.py
│   └── llm_service.py
├── prompts/
│   ├── narrative_arc.txt
│   └── script_educational.txt     # only style needed for students
├── models/                         # kokoro .onnx cached here
│   └── piper/                      # piper voices (auto-downloaded on first use)
├── artifacts/
├── requirements.txt, Dockerfile, README.md
## Educational-only style
Two hosts with teacher energy: one asks student-style questions,
the other explains clearly. Pulls directly from uploaded content.