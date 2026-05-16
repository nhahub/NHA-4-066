"""
generator_factory.py
────────────────────
Selects the LLM backend at runtime so the same RAGPipeline can run
either against local Ollama or against HuggingFace Inference API
without code changes.

Selection (env-var driven, no config.yaml change required)
──────────────────────────────────────────────────────────
    RAG_GENERATION_PROVIDER=ollama          (default)
    RAG_GENERATION_PROVIDER=huggingface

The returned object exposes a single public method:
    .generate(query: str, context: str) -> str
and a `.model` attribute (string) for logging / health reporting.

This indirection keeps `src/rag/generator.py` (Ollama, teammate's
file) untouched. We add a sibling class and pick between them here.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)


def make_generator():
    """
    Return a generator instance matching RAG_GENERATION_PROVIDER.
    Falls back to Ollama (the original behaviour) if the env var is
    unset or unrecognised, so local dev keeps working unchanged.
    """
    provider = os.environ.get("RAG_GENERATION_PROVIDER", "ollama").strip().lower()

    if provider in ("hf", "huggingface", "hugging_face"):
        from .hf_generator import HuggingFaceGenerator
        logger.info("Using HuggingFaceGenerator backend.")
        return HuggingFaceGenerator()

    if provider not in ("ollama", "", "local"):
        logger.warning(
            f"Unknown RAG_GENERATION_PROVIDER='{provider}'; "
            f"falling back to Ollama."
        )

    from .generator import Generator
    logger.info("Using OllamaGenerator backend.")
    return Generator()
