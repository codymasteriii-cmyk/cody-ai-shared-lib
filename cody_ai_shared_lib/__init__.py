from .auth import AuthError, extract_bearer_token, verify_supabase_jwt
from .email import EmailError, send_email
from .embedding import generate_embedding
from .fetcher import (
    extract_pdf_bytes,
    extract_title_and_markdown,
    fetch_article,
    normalize_url,
)
from .llm import LLMClient
from .supabase_client import _get_supabase_client as get_supabase_client
from .tts import (
    DEFAULT_VOICE,
    POPULAR_VOICES,
    SynthResult,
    TTSCache,
    TTSError,
    UnknownVoiceError,
    VOICES,
    VoiceUnavailableError,
    align_cues,
    clean_text_for_tts,
    default_voice,
    detect_language,
    get_available_voices,
    is_mostly_english,
    synthesize,
    synthesize_chunk_bytes,
    synthesize_speech,
    verify_voices_available,
)
from .url_validator import validate_public_url

__all__ = [
    "generate_embedding",
    "fetch_article",
    "normalize_url",
    "extract_pdf_bytes",
    "extract_title_and_markdown",
    "LLMClient",
    "get_supabase_client",
    "validate_public_url",
    "verify_supabase_jwt",
    "extract_bearer_token",
    "AuthError",
    "send_email",
    "EmailError",
    "synthesize",
    "SynthResult",
    "align_cues",
    "TTSCache",
    "VOICES",
    "default_voice",
    "verify_voices_available",
    "TTSError",
    "UnknownVoiceError",
    "VoiceUnavailableError",
    "synthesize_speech",
    "synthesize_chunk_bytes",
    "clean_text_for_tts",
    "detect_language",
    "is_mostly_english",
    "get_available_voices",
    "POPULAR_VOICES",
    "DEFAULT_VOICE",
]


def __getattr__(name: str):
    """Package-level PEP 562 hook — resolves 'supabase' lazily on first access.

    Preserves `from cody_ai_shared_lib import supabase` without importing
    supabase_client.supabase directly above (that would trigger its own
    __getattr__ eagerly at package-import time, defeating the lazy singleton).
    """
    if name == "supabase":
        return get_supabase_client()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
