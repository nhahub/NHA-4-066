"""
build_vector_store.py
──────────────────────
One-time (or periodic) script that:
  1. Loads rag_chunks.jsonl and faq_chunks.jsonl from disk
  2. Generates BGE embeddings for every chunk
  3. Upserts all documents (text + embedding + metadata) into MongoDB
  4. Creates metadata indexes
  5. Prints a summary of what was stored

Run from the project root:
    python -m src.vector_store.build_vector_store

Re-running is safe (upsert logic won't create duplicates).
"""

import json
import logging
import sys
from pathlib import Path

import yaml

# ── allow running as a script from project root ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.vector_store.embedder import Embedder
from src.vector_store.mongo_store import MongoVectorStore

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("build_vector_store")

CONFIG_PATH = "config/config.yaml"


# ── helpers ───────────────────────────────────────────────────────────────────

def load_jsonl(path: str) -> list[dict]:
    """Read a .jsonl file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info(f"Loaded {len(records):,} records from '{path}'")
    return records


def prepare_documents(chunks: list[dict], source: str) -> tuple[list[dict], list[str]]:
    """
    Extract texts to embed and build document skeletons.

    Returns
    -------
    documents : list of dicts ready for Mongo (missing 'embedding' key)
    texts     : parallel list of strings to feed the embedder
    """
    documents, texts = [], []
    for chunk in chunks:
        doc = {
            "chunk_id":    chunk["chunk_id"],
            "text":        chunk["text"],
            "instruction": chunk.get("instruction", ""),
            "response":    chunk.get("response", ""),
            "category":    chunk.get("category", ""),
            "intent":      chunk.get("intent", ""),
            "flags":       chunk.get("flags", ""),
            "token_len":   chunk.get("token_len", 0),
            "source":      source,
        }
        documents.append(doc)
        # We embed the full text field (Q+A or Customer+Agent exchange)
        # because it gives the model both query signal and answer context
        texts.append(chunk["text"])

    return documents, texts


# ── main pipeline ─────────────────────────────────────────────────────────────

def build(config_path: str = CONFIG_PATH, drop_existing: bool = False):
    """
    Full build pipeline.

    Parameters
    ----------
    config_path   : path to config.yaml
    drop_existing : if True, wipe the collection before rebuilding
                    (useful for a clean re-embed after a model change)
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    rag_path = cfg["data"]["rag_chunks_path"]
    faq_path = cfg["data"]["faq_chunks_path"]

    # ── Step 1: initialise components ─────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 – Initialising Embedder and MongoVectorStore")
    logger.info("=" * 60)

    embedder = Embedder(config_path)
    store    = MongoVectorStore(config_path)

    if drop_existing:
        logger.warning("drop_existing=True → dropping existing collection.")
        store.drop_collection()

    # ── Step 2: load chunks ────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 2 – Loading chunk files")
    logger.info("=" * 60)

    rag_chunks = load_jsonl(rag_path)
    faq_chunks = load_jsonl(faq_path)

    rag_docs, rag_texts = prepare_documents(rag_chunks, source="rag_chunks")
    faq_docs, faq_texts = prepare_documents(faq_chunks, source="faq_chunks")

    all_docs  = rag_docs  + faq_docs
    all_texts = rag_texts + faq_texts

    logger.info(
        f"Total chunks to embed: {len(all_texts):,} "
        f"(RAG: {len(rag_texts):,} | FAQ: {len(faq_texts):,})"
    )

    # ── Step 3: generate embeddings ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 – Generating BGE embeddings")
    logger.info("=" * 60)

    embeddings = embedder.encode_passages(all_texts)
    logger.info(f"Embedding matrix shape: {embeddings.shape}")

    # Attach each embedding to its document
    for doc, vector in zip(all_docs, embeddings):
        doc["embedding"] = vector.tolist()  # MongoDB stores JSON – list of floats

    # ── Step 4: upsert to MongoDB ──────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 – Upserting documents to MongoDB")
    logger.info("=" * 60)

    stats = store.upsert_chunks(all_docs)
    logger.info(f"Upsert stats: {stats}")

    # ── Step 5: create indexes ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 – Creating metadata indexes")
    logger.info("=" * 60)

    store.create_indexes()

    # ── Step 6: summary ────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 6 – Summary")
    logger.info("=" * 60)

    summary = store.get_stats()
    logger.info(f"Total documents in collection: {summary['total']:,}")
    for row in summary["breakdown"]:
        logger.info(f"  {row['source']:<15}  {row['category']:<20}  {row['count']:>5} chunks")

    logger.info("✅ Vector store build complete.")
    store.close()
    return summary


# ── smoke test (quick retrieval check after build) ────────────────────────────

def smoke_test(config_path: str = CONFIG_PATH):
    """
    After building, verify retrieval works end-to-end with two test queries.
    """
    from src.vector_store.search import VectorSearcher

    logger.info("=" * 60)
    logger.info("SMOKE TEST – Verifying retrieval")
    logger.info("=" * 60)

    searcher = VectorSearcher(config_path)
    test_queries = [
        "how do I cancel my order?",
        "I want to change an item in my order",
    ]

    for query in test_queries:
        logger.info(f"\nQuery: '{query}'")
        results = searcher.search(query, top_k=3)
        for r in results:
            logger.info(
                f"  [{r['score']:.4f}] ({r['source']} | {r['intent']}) "
                f"{r['instruction'][:60]}..."
            )

    logger.info("\n✅ Smoke test passed.")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build the RAG vector store")
    parser.add_argument(
        "--config", default=CONFIG_PATH, help="Path to config.yaml"
    )
    parser.add_argument(
        "--drop", action="store_true",
        help="Drop existing collection before rebuilding (clean re-embed)"
    )
    parser.add_argument(
        "--smoke-test", action="store_true",
        help="Run retrieval smoke test after building"
    )
    args = parser.parse_args()

    build(config_path=args.config, drop_existing=args.drop)

    if args.smoke_test:
        smoke_test(config_path=args.config)