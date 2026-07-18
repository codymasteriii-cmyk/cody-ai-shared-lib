from .embedding import generate_embedding
from .fetcher import fetch_article
from .llm import LLMClient
from .supabase_client import _get_supabase_client as get_supabase_client
from .url_validator import validate_public_url


def __getattr__(name: str):
    """Package-level PEP 562 hook — resolves 'supabase' lazily on first access.

    Preserves `from cody_ai_shared_lib import supabase` without importing
    supabase_client.supabase directly above (that would trigger its own
    __getattr__ eagerly at package-import time, defeating the lazy singleton).
    """
    if name == "supabase":
        return get_supabase_client()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
