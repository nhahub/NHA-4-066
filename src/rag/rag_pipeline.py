"""
rag_pipeline.py
───────────────
Combines VectorSearcher (retrieval) + Generator (generation) into one
clean end-to-end pipeline object.

This is the central object the evaluation scripts — and later the API —
will use. Everything else is just wiring around this.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.vector_store.search import VectorSearcher
from src.rag.generator import Generator

logger = logging.getLogger(__name__)


@dataclass
class RAGResult:
    """
    Holds the full output of one RAG query — useful for evaluation
    because it keeps retrieved chunks alongside the generated answer.
    """
    query:           str
    retrieved_chunks: list[dict]       # raw results from VectorSearcher
    context:         str               # formatted context string fed to LLM
    generated_answer: str              # Mistral's output
    top_intents:     list[str] = field(default_factory=list)   # intents of top-K chunks


class RAGPipeline:
    """
    End-to-end RAG pipeline: query → retrieve → generate → RAGResult

    Usage
    -----
    pipeline = RAGPipeline("config/config.yaml")
    result   = pipeline.run("how do I cancel my order?")
    print(result.generated_answer)
    """

    def __init__(
        self,
        config_path: str = "config/config.yaml",
        top_k: int = 5,
        filter_category: Optional[str] = None,
    ):
        self.config_path     = config_path
        self.top_k           = top_k
        self.filter_category = filter_category

        logger.info("Initialising RAG pipeline...")
        self.searcher  = VectorSearcher(config_path)
        self.generator = Generator()
        logger.info("RAG pipeline ready.")

    def run(self, query: str, top_k: Optional[int] = None) -> RAGResult:
        """
        Run the full RAG pipeline for a single query.

        Steps
        ─────
        1. Embed the query with BGE
        2. Retrieve top-K chunks from MongoDB
        3. Format chunks into a context string
        4. Call Mistral via Ollama with query + context
        5. Return a RAGResult with everything stored

        Parameters
        ----------
        query  : raw user question
        top_k  : override the default top_k
        """
        k = top_k or self.top_k

        # Step 1 + 2: retrieve
        retrieved_chunks, context = self.searcher.search_and_format(
            query,
            top_k=k,
            filter_category=self.filter_category,
        )

        # Step 3: generate
        generated_answer = self.generator.generate(query, context)

        # Extract intents of retrieved chunks (used in retrieval evaluation)
        top_intents = [c["intent"] for c in retrieved_chunks]

        return RAGResult(
            query=query,
            retrieved_chunks=retrieved_chunks,
            context=context,
            generated_answer=generated_answer,
            top_intents=top_intents,
        )