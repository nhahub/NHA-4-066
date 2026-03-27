"""
embedder.py
───────────
Loads the BAAI/bge-base-en-v1.5 model and generates embeddings for
text chunks.

Why BGE?
  - Strong retrieval performance on semantic similarity benchmarks
  - Runs fully locally (no API cost, no data leaving your machine)
  - Uses separate prefixes for queries vs. passages, which improves
    retrieval quality compared to symmetric models like MiniLM

Key design decisions:
  - Batched inference: avoids OOM on large corpora
  - Normalised vectors: cosine similarity becomes a simple dot product,
    which MongoDB's vector search uses internally
  - Separate encode_query() vs encode_passages() so the right prefix
    is always applied automatically
"""

import logging
from typing import Union

import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

logger = logging.getLogger(__name__)


class Embedder:
    """
    Wraps BAAI/bge-base-en-v1.5 for both passage indexing and query encoding.

    Usage
    -----
    embedder = Embedder("config/config.yaml")

    # For indexing chunks (done once, stored in MongoDB)
    vectors = embedder.encode_passages(["text1", "text2", ...])

    # For user queries at inference time
    q_vec = embedder.encode_query("how do I cancel my order?")
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        self.cfg = cfg["embedding"]
        self.model_name   = self.cfg["model_name"]
        self.batch_size   = self.cfg["batch_size"]
        self.dimension    = self.cfg["dimension"]
        self.query_prefix = self.cfg["query_prefix"]

        # Detect best available device: CUDA > MPS (Apple) > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"

        logger.info(f"Loading embedding model '{self.model_name}' on {self.device}...")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        self.model.max_seq_length = self.cfg["max_seq_length"]
        logger.info("Embedding model loaded.")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        """
        Embed a list of knowledge-base passages for indexing.

        Returns
        -------
        np.ndarray of shape (N, 768), float32, L2-normalised
        """
        return self._encode(texts, prefix="")

    def encode_query(self, query: str) -> np.ndarray:
        """
        Embed a single user query with the BGE retrieval prefix.

        Returns
        -------
        np.ndarray of shape (768,), float32, L2-normalised
        """
        return self._encode([self.query_prefix + query], prefix="")[0]

    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """Batch version of encode_query."""
        prefixed = [self.query_prefix + q for q in queries]
        return self._encode(prefixed, prefix="")

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _encode(self, texts: list[str], prefix: str) -> np.ndarray:
        """
        Core encoding loop with batching, progress bar, and normalisation.
        """
        all_embeddings = []

        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Encoding",
            unit="batch",
            disable=len(texts) < self.batch_size,  # skip bar for tiny inputs
        ):
            batch = texts[i : i + self.batch_size]
            if prefix:
                batch = [prefix + t for t in batch]

            embeddings = self.model.encode(
                batch,
                convert_to_numpy=True,
                normalize_embeddings=True,   # L2 norm → cosine sim = dot product
                show_progress_bar=False,
            )
            all_embeddings.append(embeddings.astype(np.float32))

        return np.vstack(all_embeddings)