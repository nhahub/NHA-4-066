"""
hf_generator.py
───────────────
HuggingFace Inference API generator. Drop-in replacement for the
Ollama-based Generator when running in the cloud where we don't want
to host the LLM ourselves.

Why
───
The local Generator (src/rag/generator.py) talks to Ollama running on
localhost. That's fine for dev, but in Azure Container Apps we'd need
a second container with ~6 vCPU / 8 GiB just for Mistral — too
expensive for an academic deploy.

HF Inference API hosts the model for us, free tier exists, no
infrastructure required. Same `generate(query, context) -> str`
contract as the Ollama Generator so RAGPipeline doesn't care which
one it's holding.

Env vars
────────
    HF_TOKEN           Your HuggingFace API token (required)
    HF_MODEL           Optional override; default 'mistralai/Mistral-7B-Instruct-v0.3'
    HF_API_URL         Optional override; default 'https://api-inference.huggingface.co/models'

Cold-start
──────────
Free-tier models sleep when idle. We send `options.wait_for_model=True`
so the first request blocks (up to ~20 s) until the model is loaded,
instead of getting a 503. The retry loop catches the same case
defensively in case the flag is ignored.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_API_URL = "https://api-inference.huggingface.co/models"
DEFAULT_MODEL   = "mistralai/Mistral-7B-Instruct-v0.3"

SYSTEM_PROMPT = """You are a helpful customer support assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer, say:
"I don't have enough information to answer that."
Be concise and direct. Do not repeat the question."""


class HuggingFaceGenerator:
    """
    Calls the HuggingFace Inference API.

    Usage
    -----
    gen = HuggingFaceGenerator()  # reads HF_TOKEN from env
    answer = gen.generate(
        query="How do I cancel my order?",
        context="[Chunk 1 | ...] Q: ... A: ...",
    )
    """

    # Attribute name matches OllamaGenerator's `.model` so /health and
    # other introspection can use either interchangeably.
    model: str

    def __init__(
        self,
        model:       Optional[str]   = None,
        api_url:     Optional[str]   = None,
        token:       Optional[str]   = None,
        temperature: float           = 0.1,
        max_tokens:  int             = 300,
        timeout:     int             = 60,
        max_retries: int             = 2,
    ):
        self.model       = model     or os.environ.get("HF_MODEL", DEFAULT_MODEL)
        base_url         = api_url   or os.environ.get("HF_API_URL", DEFAULT_API_URL)
        self.endpoint    = f"{base_url.rstrip('/')}/{self.model}"
        self.token       = token     or os.environ.get("HF_TOKEN")
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self.timeout     = timeout
        self.max_retries = max_retries

        if not self.token:
            raise RuntimeError(
                "HF_TOKEN env var not set. Get a free token at "
                "https://huggingface.co/settings/tokens"
            )

        self._headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type":  "application/json",
        }
        logger.info(f"HuggingFaceGenerator ready. Model: {self.model}")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def generate(self, query: str, context: str) -> str:
        """
        Generate an answer grounded in the retrieved context.
        Same signature and return type as OllamaGenerator.generate().
        """
        prompt = (
            f"<s>[INST] {SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query} [/INST]"
        )

        payload = {
            "inputs": prompt,
            "parameters": {
                "temperature":      self.temperature,
                "max_new_tokens":   self.max_tokens,
                "return_full_text": False,
                "do_sample":        self.temperature > 0,
            },
            # Block (instead of 503) while the model wakes from idle.
            "options": {
                "wait_for_model": True,
                "use_cache":      False,
            },
        }

        # One retry on cold-start (503) — HF returns estimated_time in body.
        for attempt in range(self.max_retries + 1):
            try:
                r = requests.post(
                    self.endpoint, headers=self._headers,
                    json=payload, timeout=self.timeout,
                )
            except requests.exceptions.Timeout:
                logger.error("HF Inference request timed out.")
                return "[TIMEOUT]"
            except requests.exceptions.RequestException as e:
                logger.error(f"HF Inference network error: {e}")
                return "[ERROR]"

            # Cold-start: explicit retry path
            if r.status_code == 503 and attempt < self.max_retries:
                try:
                    body = r.json()
                except ValueError:
                    body = {}
                wait = float(body.get("estimated_time", 10)) + 1
                logger.warning(
                    f"HF model warming up; retrying in {wait:.1f}s "
                    f"(attempt {attempt + 1}/{self.max_retries + 1})"
                )
                time.sleep(min(wait, 25))
                continue

            if r.status_code == 401:
                logger.error("HF Inference returned 401 — check HF_TOKEN.")
                return "[AUTH ERROR]"
            if r.status_code == 429:
                logger.error("HF Inference rate limit hit (429).")
                return "[RATE LIMITED]"
            if not r.ok:
                logger.error(
                    f"HF Inference {r.status_code}: {r.text[:200]}"
                )
                return "[ERROR]"

            return self._extract_text(r.json())

        return "[ERROR]"

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_text(payload) -> str:
        """
        HF text-generation endpoints typically return:
            [{"generated_text": "..."}]
        Some return:
            {"generated_text": "..."}
        Be tolerant of both shapes.
        """
        if isinstance(payload, list) and payload:
            first = payload[0]
            if isinstance(first, dict) and "generated_text" in first:
                return str(first["generated_text"]).strip()
        if isinstance(payload, dict) and "generated_text" in payload:
            return str(payload["generated_text"]).strip()
        if isinstance(payload, dict) and "error" in payload:
            logger.error(f"HF Inference error in body: {payload['error']}")
            return "[ERROR]"
        logger.warning(f"Unexpected HF response shape: {str(payload)[:200]}")
        return ""
