"""
auth.py
───────
API-key authentication.

Why an API key (not Azure AD)?
  - Zero Azure cost at any scale
  - Trivial for the support portal to integrate (one header)
  - In production the key lives in Azure Key Vault and is injected as
    an env var at container start — see docs/INTEGRATION_SECURITY.md

Header format:   X-API-Key: <secret>
"""

from __future__ import annotations

import hmac
import logging

from fastapi import Depends, HTTPException, Security, status
from fastapi.security.api_key import APIKeyHeader

from .dependencies import Settings, get_settings

logger = logging.getLogger("api.auth")

# auto_error=False so we can return a JSON 401 instead of FastAPI's default.
_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(
    provided_key: str | None = Security(_API_KEY_HEADER),
    settings: Settings = Depends(get_settings),
) -> None:
    """
    FastAPI dependency that 401s when the X-API-Key header is missing
    or doesn't match the configured key.

    Bypass: setting ALLOW_UNAUTH=1 disables this check (local dev only).
    """
    if not settings.auth_required():
        return  # explicit opt-out

    expected = settings.api_key
    if not expected:
        # Misconfiguration — refuse all traffic rather than silently allow.
        logger.error("API_KEY env var not set; refusing all requests.")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Server auth not configured (API_KEY missing).",
        )

    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    # Constant-time comparison to avoid timing oracles.
    if not hmac.compare_digest(provided_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key.",
        )
