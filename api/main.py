"""
main.py
───────
FastAPI application factory + lifespan.

The RAG pipeline is heavy (loads BGE, opens Mongo, pings Ollama). We
construct it once during the FastAPI lifespan (startup) and reuse the
same instance across every request.

Run locally:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

In production (Azure Container Apps), the Docker image's CMD launches
gunicorn with uvicorn workers — see deploy/Dockerfile.
"""

from __future__ import annotations

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Make sibling 'src' package importable when running `uvicorn api.main:app`
# from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.rag.rag_pipeline import RAGPipeline  # noqa: E402

from .dependencies import get_settings  # noqa: E402
from .routes import router as rag_router  # noqa: E402

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the RAG pipeline once at startup, dispose on shutdown."""
    settings = get_settings()
    logger.info("Booting RAG pipeline (config=%s)...", settings.config_path)
    try:
        app.state.pipeline = RAGPipeline(
            config_path=settings.config_path,
            top_k=settings.default_top_k,
        )
        logger.info("RAG pipeline ready.")
    except Exception as e:
        # Don't crash the process — let /health report 'degraded' so
        # operators can debug instead of looking at a restart loop.
        logger.exception("RAG pipeline failed to initialise: %s", e)
        app.state.pipeline = None

    yield

    logger.info("Shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="Customer Support RAG API",
        description=(
            "REST interface to the BGE + MongoDB + Mistral RAG pipeline. "
            "Send a customer query to `/chat`, get a grounded answer back."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["X-API-Key", "Content-Type"],
    )

    app.include_router(rag_router)
    return app


app = create_app()
