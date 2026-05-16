"""
dependencies.py
───────────────
Shared dependencies for the FastAPI app:
  - Settings (env-driven, no .env file required at minimum)
  - Cached RAGPipeline singleton (loaded once at startup)

The RAGPipeline is heavyweight (loads BGE + connects to Mongo + checks
Ollama). We instantiate it once during the app lifespan and inject it
into every request via Depends().
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from typing import Optional

logger = logging.getLogger("api.deps")


class Settings:
    """Runtime configuration sourced from environment variables."""

    def __init__(self) -> None:
        self.config_path: str = os.environ.get("RAG_CONFIG", "config/config.yaml")
        self.api_key: Optional[str] = os.environ.get("API_KEY") or None
        self.allow_unauth: bool = os.environ.get("ALLOW_UNAUTH", "").lower() in (
            "1", "true", "yes",
        )
        self.cors_origins: list[str] = [
            o.strip() for o in os.environ.get("CORS_ORIGINS", "*").split(",") if o.strip()
        ]
        self.default_top_k: int = int(os.environ.get("DEFAULT_TOP_K", "5"))

    def auth_required(self) -> bool:
        """Auth is required unless explicitly disabled via ALLOW_UNAUTH=1."""
        return not self.allow_unauth


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


# ── RAG pipeline holder ───────────────────────────────────────────────────────
# Stored on app.state.pipeline (set during lifespan), not in a module-level
# global, so tests can swap it out cleanly.

def get_pipeline(request):  # FastAPI Request, untyped to avoid import cycles
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError(
            "RAGPipeline not initialised. The app lifespan should set it."
        )
    return pipeline
