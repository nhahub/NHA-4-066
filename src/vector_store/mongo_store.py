"""
mongo_store.py
──────────────
Handles all MongoDB interactions:
  1. Connecting to local MongoDB
  2. Upserting chunk documents with their embedding vectors
  3. Creating a vector search index (using mongot / Atlas-compatible format,
     with a cosine-similarity fallback for local MongoDB)
  4. Providing raw vector search via aggregation pipeline

Why store vectors in MongoDB?
  - Your chunks are already structured JSON – Mongo is a natural fit
  - Keeps the entire knowledge base (text + vectors + metadata) in one place
  - Easy to filter by category/intent alongside vector similarity
  - Seamless migration path to MongoDB Atlas Vector Search later

Local MongoDB Note
──────────────────
Local MongoDB (< 7.0) does NOT support the $vectorSearch aggregation stage
(that is Atlas-only). For local dev we implement an approximate cosine search
using $addFields + $project + $sort instead. The interface is identical so
you can swap in Atlas later by just changing the search method.
"""

import logging
from typing import Any

import numpy as np
import pymongo
import yaml
from pymongo import MongoClient, UpdateOne

logger = logging.getLogger(__name__)


class MongoVectorStore:
    """
    Manages the chunk_embeddings collection in local MongoDB.

    Each document schema
    ────────────────────
    {
        "_id":        str,          # chunk_id (unique, used for upserts)
        "text":       str,          # full chunk text fed to the LLM
        "instruction":str,          # original user query
        "response":   str,          # agent response
        "category":   str,          # e.g. "ORDER"
        "intent":     str,          # e.g. "cancel_order"
        "flags":      str,          # quality flags from preprocessing
        "token_len":  int,
        "source":     str,          # "rag_chunks" | "faq_chunks"
        "embedding":  list[float],  # 768-d BGE vector
    }
    """

    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        mongo_cfg  = cfg["mongodb"]
        self.uri   = mongo_cfg["uri"]
        self.db_name      = mongo_cfg["db_name"]
        self.collection_name = mongo_cfg["collections"]["embeddings"]

        self.client     = MongoClient(self.uri)
        self.db         = self.client[self.db_name]
        self.collection = self.db[self.collection_name]

        logger.info(
            f"Connected to MongoDB at {self.uri} → "
            f"db='{self.db_name}', collection='{self.collection_name}'"
        )

    # ------------------------------------------------------------------ #
    #  Indexing                                                            #
    # ------------------------------------------------------------------ #

    def upsert_chunks(self, documents: list[dict]) -> dict:
        """
        Insert or update documents in bulk.

        Uses UpdateOne with upsert=True keyed on _id (chunk_id) so
        re-running the build script is idempotent – existing chunks are
        updated, new ones are inserted.

        Parameters
        ----------
        documents : list of dicts, each must contain 'chunk_id' and 'embedding'

        Returns
        -------
        dict with inserted / modified counts
        """
        if not documents:
            return {"inserted": 0, "modified": 0}

        operations = [
            UpdateOne(
                filter={"_id": doc["chunk_id"]},
                update={"$set": {**doc, "_id": doc["chunk_id"]}},
                upsert=True,
            )
            for doc in documents
        ]

        result = self.collection.bulk_write(operations, ordered=False)
        stats = {
            "inserted": result.upserted_count,
            "modified": result.modified_count,
        }
        logger.info(f"Upsert complete: {stats}")
        return stats

    def create_indexes(self):
        """
        Create standard MongoDB indexes on metadata fields.

        These speed up filtered queries like:
          "find ORDER-category chunks similar to this query"

        Note: $vectorSearch index (Atlas) is NOT created here because local
        MongoDB doesn't support it. We handle vector search manually below.
        """
        self.collection.create_index("category")
        self.collection.create_index("intent")
        self.collection.create_index("source")
        self.collection.create_index([("category", 1), ("intent", 1)])
        logger.info("Metadata indexes created.")

    # ------------------------------------------------------------------ #
    #  Vector Search (local MongoDB fallback)                             #
    # ------------------------------------------------------------------ #

    def vector_search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_category: str | None = None,
        filter_intent: str | None = None,
        min_score: float = 0.0,
    ) -> list[dict]:
        """
        Find the top-k most similar chunks to a query vector.

        Local implementation: load candidate documents, compute cosine
        similarity in Python, sort, and return top-k.

        This is efficient enough for datasets up to ~100k chunks.
        For production scale → migrate to Atlas $vectorSearch.

        Parameters
        ----------
        query_vector     : 1-D numpy array of shape (768,), L2-normalised
        top_k            : number of results to return
        filter_category  : optional metadata pre-filter (e.g. "ORDER")
        filter_intent    : optional metadata pre-filter (e.g. "cancel_order")
        min_score        : minimum cosine similarity to include in results

        Returns
        -------
        list of dicts sorted by score descending, each containing:
            chunk_id, text, instruction, response, category, intent,
            source, score
        """
        # Build optional pre-filter to narrow the candidate set
        mongo_filter: dict[str, Any] = {}
        if filter_category:
            mongo_filter["category"] = filter_category
        if filter_intent:
            mongo_filter["intent"] = filter_intent

        # Fetch candidates (only fields we need – exclude large embedding)
        projection = {
            "_id": 1,
            "text": 1,
            "instruction": 1,
            "response": 1,
            "category": 1,
            "intent": 1,
            "source": 1,
            "token_len": 1,
            "embedding": 1,
        }
        candidates = list(self.collection.find(mongo_filter, projection))

        if not candidates:
            logger.warning("No candidates found for vector search.")
            return []

        # Stack embeddings into a matrix for fast dot-product (= cosine sim
        # because both query and passages are L2-normalised)
        embeddings_matrix = np.array(
            [doc["embedding"] for doc in candidates], dtype=np.float32
        )
        query_vector = query_vector.astype(np.float32)
        scores = embeddings_matrix @ query_vector  # shape: (N,)

        # Get top-k indices sorted by descending score
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score < min_score:
                break
            doc = candidates[idx]
            results.append(
                {
                    "chunk_id":    doc["_id"],
                    "text":        doc["text"],
                    "instruction": doc.get("instruction", ""),
                    "response":    doc.get("response", ""),
                    "category":    doc.get("category", ""),
                    "intent":      doc.get("intent", ""),
                    "source":      doc.get("source", ""),
                    "token_len":   doc.get("token_len", 0),
                    "score":       round(score, 4),
                }
            )

        return results

    # ------------------------------------------------------------------ #
    #  Utilities                                                           #
    # ------------------------------------------------------------------ #

    def count(self) -> int:
        return self.collection.count_documents({})

    def get_stats(self) -> dict:
        """Return a breakdown of stored chunks by source and category."""
        pipeline = [
            {
                "$group": {
                    "_id": {"source": "$source", "category": "$category"},
                    "count": {"$sum": 1},
                }
            },
            {"$sort": {"_id.source": 1, "_id.category": 1}},
        ]
        rows = list(self.collection.aggregate(pipeline))
        return {
            "total": self.count(),
            "breakdown": [
                {
                    "source":   r["_id"]["source"],
                    "category": r["_id"]["category"],
                    "count":    r["count"],
                }
                for r in rows
            ],
        }

    def drop_collection(self):
        """Wipe the collection – useful for a clean rebuild."""
        self.collection.drop()
        logger.warning(f"Collection '{self.collection_name}' dropped.")

    def close(self):
        self.client.close()