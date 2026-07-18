"""Supabase Client — Lazy singleton database connection using the service role (secret) key.

Uses SUPABASE_SECRET_API_KEY to bypass Row Level Security (RLS) in backend
services. Frontend clients use the publishable anon key with RLS enforcement instead.

The client is created on first use (lazy init), not at import time. This means:
  - Importing this module never fails due to missing env vars.
  - RuntimeError is raised on the first actual database call if env vars are absent.
  - Unit tests and tools can import the module without a live Supabase connection.

Callers: `from cody_ai_shared_lib.supabase_client import supabase` continues to work
unchanged. Python resolves the name via module __getattr__ on first access.
"""
import os
import logging
from typing import Optional
from supabase import create_client, Client
from dotenv import load_dotenv, find_dotenv

# find_dotenv(usecwd=True) searches upward from the process's working directory,
# not from this file's location in site-packages. Required when the package is
# installed via pip from GitHub rather than run from the project directory.
load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))

logger = logging.getLogger("db-supabase")

# Kept as module-level constants so callers that import SUPABASE_URL / SUPABASE_KEY
# directly continue to work (e.g. app/db/supabase.py in NewsIntelligence). These
# are a convenience snapshot only — _get_supabase_client() re-reads os.getenv()
# itself rather than trusting this snapshot, so tests that set env vars after
# import still get picked up on first real call.
SUPABASE_URL: Optional[str] = os.getenv("SUPABASE_URL")
SUPABASE_KEY: Optional[str] = os.getenv("SUPABASE_SECRET_API_KEY")

# Module-level lazy singleton. Do NOT access directly — use _get_supabase_client().
_supabase_client: Optional[Client] = None


def _get_supabase_client() -> Client:
    """Return the shared Supabase client, creating it on first call.

    Re-reads os.getenv() at call time (not the module-level SUPABASE_URL/KEY
    snapshot above) so env vars set after import — e.g. by a test fixture —
    are picked up. Fails fast with RuntimeError if the required env vars are
    absent, so the error surfaces at the first database call rather than
    silently returning None.
    """
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SECRET_API_KEY")
        if not url or not key:
            raise RuntimeError(
                "Missing required environment variables: SUPABASE_URL and/or "
                "SUPABASE_SECRET_API_KEY. A valid database connection is required."
            )
        _supabase_client = create_client(url, key)
        logger.debug("Supabase client initialised (lazy singleton).")
    return _supabase_client


def __getattr__(name: str):
    """Module-level __getattr__ — resolves 'supabase' lazily on first access.

    Allows `from cody_ai_shared_lib.supabase_client import supabase` to keep
    working at all existing call sites without any code changes, while deferring
    client creation until the name is actually accessed.
    """
    if name == "supabase":
        return _get_supabase_client()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
