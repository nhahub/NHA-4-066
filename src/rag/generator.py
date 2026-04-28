"""
generator.py
────────────
Calls a local Ollama model (Mistral 7B) to generate answers given:
  - A user query
  - Retrieved context chunks from the vector store

Why Ollama?
  - Runs 100% locally — no API cost, no data leaving your machine
  - Mistral 7B is fast and handles customer support Q&A well
  - Simple REST API at localhost:11434 — no special SDK needed

Prompt Design
─────────────
We use a structured system prompt that tells the model:
  1. It is a customer support assistant
  2. It must answer ONLY from the provided context
  3. If context is insufficient, say so — don't hallucinate

This "grounded generation" constraint is critical for RAG quality.
"""

import logging
from typing import Optional
import requests

logger = logging.getLogger(__name__)

OLLAMA_URL   = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "mistral"

SYSTEM_PROMPT = """You are a customer support assistant. Answer EXACTLY as shown in examples.

EXAMPLES:
Q: How do I return an item?
A: Items can be returned within 30 days of purchase in original condition. Go to your Orders, select the item, and click "Return".

Q: What's the refund timeline?
A: Refunds are processed within 5-7 business days after we receive your return.

Q: Can I cancel my order?
A: Orders can be cancelled within 24 hours of purchase before shipping. Contact support immediately to request cancellation.

---
Now answer the following question EXACTLY in the style of the examples above:
- Be concise (1-3 sentences)
- Use terminology from the CONTEXT
- Do not invent information not in CONTEXT
- Match the direct, helpful tone of examples"""


class Generator:
    """
    Wraps the Ollama REST API for answer generation.

    Usage
    -----
    gen = Generator()
    answer = gen.generate(
        query="How do I cancel my order?",
        context="[Chunk 1 | ...] Q: ... A: ..."
    )
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = OLLAMA_URL,
        temperature: float = 0.1,      # low temp → consistent, factual answers
        max_tokens: int = 300,
    ):
        self.model       = model
        self.ollama_url  = ollama_url
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._check_ollama()

    def _check_ollama(self):
        """Verify Ollama is running before we start evaluation."""
        try:
            r = requests.get("http://localhost:11434/api/tags", timeout=3)
            models = [m["name"] for m in r.json().get("models", [])]
            if not any(self.model in m for m in models):
                logger.warning(
                    f"Model '{self.model}' not found in Ollama. "
                    f"Run: ollama pull {self.model}"
                )
            else:
                logger.info(f"Ollama ready. Using model: {self.model}")
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                "Ollama is not running. Start it with: ollama serve"
            )

    def generate(self, query: str, context: str) -> str:
        """
        Generate an answer grounded in the retrieved context.

        Parameters
        ----------
        query   : the user's original question
        context : formatted string from VectorSearcher.format_context()

        Returns
        -------
        Generated answer string
        """
        prompt = (
            f"{SYSTEM_PROMPT}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"QUESTION: {query}\n\n"
            f"ANSWER:"
        )

        payload = {
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature":   self.temperature,
                "num_predict":   self.max_tokens,
            },
        }

        try:
            response = requests.post(
                self.ollama_url, json=payload, timeout=400
            )
            response.raise_for_status()
            answer = response.json().get("response", "").strip()
            return answer
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out.")
            return "[TIMEOUT]"
        except Exception as e:
            logger.error(f"Ollama error: {e}")
            return "[ERROR]"