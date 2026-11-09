from .auth import AuthError, extract_bearer_token, verify_supabase_jwt
from .email import EmailError, send_email
from .embedding import (
    EmbeddingResult,
    EmbeddingUsage,
    generate_embedding,
    generate_embedding_with_usage,
)
from .fetcher import (
    extract_pdf_text,
    extract_title_and_markdown,
    fetch_article,
    normalize_url,
)
from .jev import JevClient, JevResult, JevUsage, NoulAnswer
from .llm import LLMClient, LLMResult, LLMUsage, ModelCallError
from .pricing import (
    PRICE_TABLE_VERSION,
    clear_price_overrides,
    compute_cost,
    estimate_cost,
    get_model_pricing,
    get_price_table_version,
    set_price_override,
)
from .rerank import RerankClient, RerankItem, RerankResult, RerankUsage
from .supabase_client import _get_supabase_client as get_supabase_client
from .url_validator import validate_public_url

# Graceful optional import for TTS — allows non-audio projects (such as NewsIntelligence)
# to use auth, email, llm, fetcher, etc. without requiring edge-tts to be installed.
try:
    from . import tts
    _tts_available = True
    _tts_public_names = list(tts.__all__)
    for _name in _tts_public_names:
        globals()[_name] = getattr(tts, _name)
except ModuleNotFoundError as _tts_import_error:
    # Only a missing edge-tts means "TTS not installed"; any other import problem is a real bug.
    if _tts_import_error.name != "edge_tts":
        raise
    _tts_available = False
    _tts_public_names = []

__all__ = [
    "generate_embedding",
    "generate_embedding_with_usage",
    "EmbeddingResult",
    "EmbeddingUsage",
    "fetch_article",
    "normalize_url",
    "extract_pdf_text",
    "extract_title_and_markdown",
    "LLMClient",
    "LLMResult",
    "LLMUsage",
    "ModelCallError",
    "JevClient",
    "JevResult",
    "JevUsage",
    "NoulAnswer",
    "RerankClient",
    "RerankResult",
    "RerankUsage",
    "RerankItem",
    "get_supabase_client",
    "validate_public_url",
    "verify_supabase_jwt",
    "extract_bearer_token",
    "AuthError",
    "send_email",
    "EmailError",
    "PRICE_TABLE_VERSION",
    "compute_cost",
    "estimate_cost",
    "get_model_pricing",
    "get_price_table_version",
    "set_price_override",
    "clear_price_overrides",
]

if _tts_available:
    __all__.extend(_tts_public_names)


# Names that need edge-tts (kept in sync with tts.__all__).
_TTS_NAMES = frozenset({
    "DEFAULT_CACHE_DIR", "SynthResult", "TTSCache", "TTSError", "UnknownVoiceError", "VOICES",
    "VOICE_MAP", "align_cues", "clean_text_for_tts", "default_voice", "detect_language",
    "is_mostly_english", "synthesize", "verify_voices_available",
})


def __getattr__(name: str):
    """Package-level PEP 562 hook — resolves 'supabase' lazily on first access,
    and provides a clear error if TTS features are used without edge-tts.

    Preserves `from cody_ai_shared_lib import supabase` without importing
    supabase_client.supabase directly above (that would trigger its own
    __getattr__ eagerly at package-import time, defeating the lazy singleton).
    """
    if name == "supabase":
        return get_supabase_client()
    if not _tts_available and name in _TTS_NAMES:
        raise ImportError(
            f"'{name}' requires edge-tts, which is not installed in the current environment. "
            "Install with `pip install edge-tts` to use TTS features."
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
