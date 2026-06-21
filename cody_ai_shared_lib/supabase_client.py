"""Supabase Client — Singleton database connection using the service role (secret) key.

Uses SUPABASE_SECRET_API_KEY to bypass Row Level Security (RLS) in backend
services. Frontend clients use the publishable anon key with RLS enforcement instead.

Fails fast at import time if required env vars are missing, preventing
silent NoneType crashes during request handling.
"""
import os
import logging
from supabase import create_client, Client
from dotenv import load_dotenv, find_dotenv

# find_dotenv(usecwd=True) searches upward from the process's working directory,
# not from this file's location in site-packages. Required when the package is
# installed via pip from GitHub rather than run from the project directory.
load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SECRET_API_KEY")

logger = logging.getLogger("db-supabase")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "Missing required environment variables: SUPABASE_URL and/or SUPABASE_SECRET_API_KEY. "
        "The server cannot start without a valid database connection."
    )

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
