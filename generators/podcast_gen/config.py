from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


# Resolve to <repo>/generators/podcast_gen/models so default paths work from
# any cwd — repo root, the package dir, or the Docker WORKDIR.
_PACKAGE_DIR = Path(__file__).resolve().parent
_DEFAULT_MODELS_DIR = _PACKAGE_DIR / "models"
_DEFAULT_PIPER_DIR = _DEFAULT_MODELS_DIR / "piper"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PODCAST_GEN_",
        env_file=".env",
        extra="ignore",
    )

    generator_id: str = "podcast_gen"
    output_type: str = "podcast"
    version: str = "0.1.0"

    host: str = "0.0.0.0"
    port: int = 8007

    ollama_host: str = "http://localhost:11434"
    chat_model: str = "llama3.1:8b-instruct-q4_K_M"
    extraction_model: str = "llama3.1:8b-instruct-q4_K_M"
    generation_model: str = "llama3.1:8b-instruct-q4_K_M"

    extraction_temperature: float = 0.2
    generation_temperature: float = 0.6
    chat_temperature: float = 0.4

    # Duration presets — student-style hosts read ~150 wpm, so a 4-minute
    # podcast is ~600 words. Map duration option → target script word count.
    duration_word_targets: dict[str, int] = {
        "short": 600,    # ~4 min
        "medium": 1400,  # ~9 min
        "long": 2500,    # ~16 min
    }
    default_duration: str = "medium"

    # How many narrative key points to extract before scripting.
    min_key_points: int = 3
    max_key_points: int = 8

    # ---------- TTS ----------
    # Backend probe order (preferred → fallback): piper → kokoro → pyttsx3.
    # Piper is preferred because every supported language ships TWO distinct
    # voices, so French (and other non-English) podcasts get genuinely
    # different host_a and host_b voices instead of a single voice with a
    # pitch-shift hack.
    #
    # kokoro-onnx model + voices. Defaults resolve to the package's models/
    # directory so it works whether you run from the repo root, from this
    # package, or inside Docker (the compose file mounts a volume here too).
    kokoro_model_path: str = str(_DEFAULT_MODELS_DIR / "kokoro-v1.0.onnx")
    kokoro_voices_path: str = str(_DEFAULT_MODELS_DIR / "voices-v1.0.bin")

    # Piper voice files live as <models_dir>/<voice_id>.onnx (+ .onnx.json).
    # Missing files are auto-downloaded from HuggingFace on first use unless
    # piper_auto_download is False.
    piper_models_dir: str = str(_DEFAULT_PIPER_DIR)
    piper_voice_url_base: str = (
        "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
    )
    piper_auto_download: bool = True
    # length_scale=1.0 is normal; <1 faster, >1 slower.
    piper_length_scale: float = 1.0
    # Default voices (used when language='en-us'). language_voices below
    # overrides per-language. Two distinct voices: A = curious student-style
    # host, B = teacher.
    voice_host_a: str = "af_heart"     # warmer, higher — student
    voice_host_b: str = "am_michael"   # calmer, lower — teacher
    tts_sample_rate: int = 24000
    tts_speed: float = 1.0
    # When the chosen language only ships one voice (e.g. French has only
    # ff_siwis), we use it for both hosts and apply BOTH a small speed delta
    # AND a pitch shift so the two turns sound clearly distinct. Pitch
    # shifting is done by resampling, which changes duration too — small
    # values (a few semitones) keep things natural-sounding.
    single_voice_speed_delta: float = 0.08
    single_voice_pitch_a_semitones: float = 1.5    # student: slightly higher
    single_voice_pitch_b_semitones: float = -3.0   # teacher: deeper

    # ---------- Hosts ----------
    # AI hosts shouldn't claim made-up human names (the LLM otherwise drops
    # `[prénom]` placeholders into the script). When both are None, the prompt
    # tells the model to skip self-introductions entirely. Override per-call
    # via options.host_a_name / options.host_b_name, or set defaults here.
    host_a_name: str | None = None
    host_b_name: str | None = None

    # Language → (kokoro lang code, host_a voice, host_b voice).
    # Where a language only ships a single voice in voices-v1.0.bin we
    # reuse it for both hosts and lean on single_voice_speed_delta.
    default_language: str = "en-us"
    language_voices: dict[str, dict[str, str]] = {
        "en-us": {"lang": "en-us", "host_a": "af_heart",   "host_b": "am_michael"},
        "en-gb": {"lang": "en-gb", "host_a": "bf_emma",    "host_b": "bm_george"},
        "fr-fr": {"lang": "fr-fr", "host_a": "ff_siwis",   "host_b": "ff_siwis"},
        "es":    {"lang": "es",    "host_a": "ef_dora",    "host_b": "em_alex"},
        "it":    {"lang": "it",    "host_a": "if_sara",    "host_b": "im_nicola"},
        "pt-br": {"lang": "pt-br", "host_a": "pf_dora",    "host_b": "pm_alex"},
        "ja":    {"lang": "ja",    "host_a": "jf_alpha",   "host_b": "jm_kumo"},
        "cmn":   {"lang": "cmn",   "host_a": "zf_xiaoxiao","host_b": "zm_yunjian"},
        "hi":    {"lang": "hi",    "host_a": "hf_alpha",   "host_b": "hm_omega"},
    }

    # Piper voice catalog. Two distinct neural voices per language — host_a
    # is the curious learner (typically female), host_b is the teacher
    # (typically male). Voice IDs follow Piper's naming on HuggingFace
    # (rhasspy/piper-voices). Languages without two strong voices in Piper
    # (ja, cmn, hi) are intentionally omitted — they fall back to Kokoro.
    piper_language_voices: dict[str, dict[str, str]] = {
        "en-us": {"host_a": "en_US-amy-medium",      "host_b": "en_US-ryan-high"},
        "en-gb": {"host_a": "en_GB-alba-medium",     "host_b": "en_GB-alan-medium"},
        "fr-fr": {"host_a": "fr_FR-siwis-medium",    "host_b": "fr_FR-gilles-low"},
        "es":    {"host_a": "es_ES-davefx-medium",   "host_b": "es_ES-sharvard-medium"},
        "it":    {"host_a": "it_IT-paola-medium",    "host_b": "it_IT-riccardo-x_low"},
        "pt-br": {"host_a": "pt_BR-faber-medium",    "host_b": "pt_BR-edresson-low"},
        "de":    {"host_a": "de_DE-thorsten-medium", "host_b": "de_DE-karlsson-low"},
    }

    # Silence between segments (ms).
    inter_segment_silence_ms: int = 280
    intro_outro_silence_ms: int = 600

    # MP3 export
    mp3_bitrate: str = "128k"

    # ---------- Storage ----------
    artifacts_dir: str = "artifacts"
    minio_endpoint: str = "localhost:9000"
    minio_public_endpoint: str | None = None
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "teacherlm"
    minio_secure: bool = False
    artifact_url_ttl_s: int = 3600

    request_timeout_s: float = 600.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
