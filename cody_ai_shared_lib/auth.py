"""Supabase JWT Verification — Authenticate requests using Supabase's JWKS endpoint.

Validates RS256 / ES256 JSON Web Tokens (JWT) issued by Supabase Auth for any
project. This module is schema-agnostic and application-agnostic: it verifies
that a token is authentic, current, and issued by the configured Supabase
instance, but does NOT perform application-level authorization (such as checking
app-specific profile tables or roles).

Usage:
    from cody_ai_shared_lib.auth import verify_supabase_jwt, extract_bearer_token

    # 1. Extract Bearer token from HTTP Authorization header
    token = extract_bearer_token(request.headers.get("Authorization"))

    # 2. Verify signature and claims (returns normalized dict or None)
    claims = verify_supabase_jwt(token)
    if not claims:
        raise HTTPException(status_code=401, detail="Invalid or expired session")

    # 3. Access verified user claims
    user_id = claims["user_id"]  # Normalized from standard 'sub' claim
    email = claims.get("email")
"""

import logging
import os
from typing import Any, Dict, Optional
import jwt
from jwt import PyJWKClient, PyJWKClientError
from dotenv import find_dotenv, load_dotenv

# Search upward for .env if not already loaded
load_dotenv(find_dotenv(usecwd=True, raise_error_if_not_found=False))

logger = logging.getLogger("cody-ai-shared-auth")

# Cache JWKS clients keyed by base Supabase URL to reuse public key in-memory caches
_jwks_clients: Dict[str, PyJWKClient] = {}


class AuthError(Exception):
    """Raised when token validation fails and raise_error=True."""
    pass


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    """Safely extract the bearer token string from an Authorization header.

    Handles case-insensitive 'Bearer ' prefix and whitespace stripping.
    Prevents prefix-stripping bypass attacks by using a single split.
    """
    if not authorization_header or not isinstance(authorization_header, str):
        return None

    parts = authorization_header.strip().split(" ", 1)
    if len(parts) == 2 and parts[0].lower() == "bearer":
        return parts[1].strip() or None
    return None


def _get_jwks_client(supabase_url: str) -> PyJWKClient:
    """Retrieve or create a cached PyJWKClient for the given Supabase URL."""
    clean_url = supabase_url.rstrip("/")
    if clean_url not in _jwks_clients:
        jwks_url = f"{clean_url}/auth/v1/.well-known/jwks.json"
        _jwks_clients[clean_url] = PyJWKClient(jwks_url)
    return _jwks_clients[clean_url]


def verify_supabase_jwt(
    token: Optional[str],
    supabase_url: Optional[str] = None,
    audience: str = "authenticated",
    raise_error: bool = False,
) -> Optional[Dict[str, Any]]:
    """Verify and decode a Supabase session JWT using RS256/ES256 public keys.

    Args:
        token: Raw JWT string or 'Bearer <token>' string.
        supabase_url: Supabase project URL (defaults to os.getenv("SUPABASE_URL")).
        audience: Expected audience claim (defaults to 'authenticated').
        raise_error: If True, raises AuthError on failure. If False, logs and returns None.

    Returns:
        A dictionary of normalized claims on success:
            - user_id: User's UUID (from 'sub' claim)
            - sub: User's UUID
            - email: User's email address (if present)
            - role: User's auth role (e.g. 'authenticated')
            - app_metadata: Dictionary of app metadata
            - user_metadata: Dictionary of user metadata
            - claims: Full raw decoded claims dictionary
        Returns None (or raises AuthError) if verification fails.
    """
    if not token or not isinstance(token, str):
        msg = "Token is missing or empty"
        if raise_error:
            raise AuthError(msg)
        return None

    # Strip 'Bearer ' if caller passed the full header value directly
    if token.strip().lower().startswith("bearer "):
        token = token.strip().split(" ", 1)[1].strip()

    target_url = supabase_url or os.getenv("SUPABASE_URL")
    if not target_url:
        msg = "SUPABASE_URL is not configured and was not provided"
        logger.error(f"[Auth] {msg}")
        if raise_error:
            raise AuthError(msg)
        return None

    target_url = target_url.rstrip("/")
    expected_issuer = f"{target_url}/auth/v1"

    try:
        jwks_client = _get_jwks_client(target_url)
        signing_key = jwks_client.get_signing_key_from_jwt(token)

        payload = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=audience,
            issuer=expected_issuer,
        )

        user_id = payload.get("sub")
        if not user_id:
            msg = "Token missing subject ('sub') claim"
            logger.warning(f"[Auth] {msg}")
            if raise_error:
                raise AuthError(msg)
            return None

        return {
            "user_id": user_id,
            "sub": user_id,
            "email": payload.get("email"),
            "role": payload.get("role"),
            "app_metadata": payload.get("app_metadata", {}),
            "user_metadata": payload.get("user_metadata", {}),
            "claims": payload,
        }

    except jwt.ExpiredSignatureError:
        logger.warning("[Auth] Token expired")
        if raise_error:
            raise AuthError("Token expired")
        return None
    except jwt.InvalidAudienceError:
        logger.warning(f"[Auth] Invalid audience claim (expected '{audience}')")
        if raise_error:
            raise AuthError("Invalid token audience")
        return None
    except jwt.InvalidIssuerError:
        logger.warning(f"[Auth] Invalid issuer claim (expected '{expected_issuer}')")
        if raise_error:
            raise AuthError("Invalid token issuer")
        return None
    except (jwt.InvalidTokenError, PyJWKClientError) as e:
        logger.warning(f"[Auth] Token verification failed: {e}")
        if raise_error:
            raise AuthError(f"Invalid token: {e}")
        return None
    except Exception as e:
        logger.error(f"[Auth] Unexpected verification error: {type(e).__name__} - {e}")
        if raise_error:
            raise AuthError(f"Authentication failed: {e}")
        return None
