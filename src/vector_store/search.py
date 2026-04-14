"""
search.py
─────────
Clean retrieval interface that wraps Embedder + MongoVectorStore.

This is the only module the RAG pipeline needs to import.
It hides all the embedding and DB complexity behind a single method:

    searcher = VectorSearcher("config/config.yaml")
    results  = searcher.search("how do I cancel my order?", top_k=5)

Design rationale
────────────────
- Single responsibility: convert a query string into ranked chunk results
- Embedder and MongoVectorStore are injected (or auto-created), making
  this easy to unit-test by mocking either dependency
- Supports optional category/intent filters for guided retrieval
  (e.g., the chatbot already knows this is an ORDER query → pre-filter)
"""

import logging
from typing import Optional

import yaml

from .embedder import Embedder
from .mongo_store import MongoVectorStore

logger = logging.getLogger(__name__)


class VectorSearcher:
    """
    One-stop shop for semantic retrieval.

    Parameters
    ----------
    config_path : path to config/config.yaml
    embedder    : optional pre-loaded Embedder (avoids reloading the model)
    store       : optional pre-loaded MongoVectorStore
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        embedder: Optional[Embedder] = None,
        store: Optional[MongoVectorStore] = None,
    ):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.search_cfg = cfg["search"]
        self.embedder   = embedder or Embedder(config_path)
        self.store      = store or MongoVectorStore(config_path)

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_category: Optional[str] = None,
        filter_intent: Optional[str] = None,
        min_score: Optional[float] = None,
    ) -> list[dict]:
        """
        Retrieve the most relevant chunks for a user query.

        Parameters
        ----------
        query           : raw user question (no prefix needed, handled internally)
        top_k           : override config default
        filter_category : restrict search to a category (e.g. "ORDER")
        filter_intent   : restrict search to an intent (e.g. "cancel_order")
        min_score       : minimum cosine similarity (0–1)

        Returns
        -------
        List of dicts, each with keys:
            chunk_id, text, instruction, response,
            category, intent, source, token_len, score
        """
        top_k     = top_k     or self.search_cfg["top_k"]
        min_score = min_score or self.search_cfg["min_score"]

        # Step 1: embed the query with the BGE retrieval prefix
        query_vector = self.embedder.encode_query(query)

        # Step 2: cosine similarity search in MongoDB
        results = self.store.vector_search(
            query_vector=query_vector,
            top_k=top_k,
            filter_category=filter_category,
            filter_intent=filter_intent,
            min_score=min_score,
        )

        logger.info(
            f"Query: '{query[:60]}...' → {len(results)} results "
            f"(top score: {results[0]['score'] if results else 'N/A'})"
        )
        return results

    def format_context(self, results: list[dict]) -> str:
        """
        Format retrieved chunks into a context string ready for the LLM prompt.

        Each chunk contributes its agent response (the authoritative answer).
        The LLM will use this context to generate its final reply.
        """
        if not results:
            return "No relevant information found in the knowledge base."

        parts = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Chunk {i} | {r['source']} | {r['category']} / {r['intent']} "
                f"| score: {r['score']}]\n"
                f"Q: {r['instruction']}\n"
                f"A: {r['response']}"
            )
        return "\n\n---\n\n".join(parts)

    def search_and_format(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_category: Optional[str] = None,
    ) -> tuple[list[dict], str]:
        """
        Convenience method: retrieve + format in one call.

        Returns
        -------
        (results list, formatted context string)
        """
        results = self.search(query, top_k=top_k, filter_category=filter_category)
        context = self.format_context(results)
        return results, context