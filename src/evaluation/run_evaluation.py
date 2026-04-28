"""
run_evaluation.py
──────────────────
Main evaluation orchestrator.

What it does
────────────
1. Loads corpus_test.parquet (your held-out test set from preprocessing)
2. Samples N queries (default 50)
3. For each query:
   a. Runs the full RAG pipeline (retrieve + generate)
   b. Records retrieved intents for retrieval evaluation
   c. Records generated answer vs ground truth for relevance evaluation
4. Computes all metrics via RetrievalEvaluator + RelevanceEvaluator
5. Saves two output files:
   - reports/eval_results.json  : full per-query details
   - reports/eval_report.json   : aggregated metric summary

Run from project root:
    python -m src.evaluation.run_evaluation
    python -m src.evaluation.run_evaluation --samples 200
    python -m src.evaluation.run_evaluation --no-bertscore   # faster
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.rag.rag_pipeline import RAGPipeline
from src.vector_store.embedder import Embedder
from src.evaluation.retrieval_eval import RetrievalEvaluator
from src.evaluation.relevance_eval import RelevanceEvaluator

# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_evaluation")

CONFIG_PATH  = "config/config.yaml"
REPORTS_DIR  = Path("reports")


# ── helpers ───────────────────────────────────────────────────────────────────

def load_test_set(config_path: str, n_samples: int) -> pd.DataFrame:
    """
    Load corpus_test.parquet and sample n_samples rows.

    We stratify by intent so the sample covers all support categories,
    not just the most common ones.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # corpus_test lives next to the other preprocessed files
    test_path = Path(cfg["data"]["rag_chunks_path"]).parent / "corpus_test.parquet"

    df = pd.read_parquet(test_path)
    logger.info(f"Loaded test set: {len(df):,} rows, columns: {list(df.columns)}")

    # Stratified sample — proportional per intent
    intents = df["intent"].value_counts(normalize=True)
    sampled_parts = []
    for intent, proportion in intents.items():
        intent_df = df[df["intent"] == intent]
        n = max(1, round(proportion * n_samples))
        sampled_parts.append(intent_df.sample(min(n, len(intent_df)), random_state=42))

    sampled = pd.concat(sampled_parts).sample(
        min(n_samples, sum(len(p) for p in sampled_parts)),
        random_state=42,
    ).reset_index(drop=True)

    logger.info(
        f"Sampled {len(sampled)} queries across "
        f"{sampled['intent'].nunique()} intents"
    )
    return sampled


# ── main evaluation loop ──────────────────────────────────────────────────────

def run(
    config_path:    str  = CONFIG_PATH,
    n_samples:      int  = 50,
    use_bertscore:  bool = True,
):
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ── Step 1: Load components ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 1 – Loading RAG pipeline & evaluators")
    logger.info("=" * 60)

    pipeline = RAGPipeline(config_path, top_k=20)
    embedder = pipeline.searcher.embedder          # reuse — don't load twice

    retrieval_evaluator = RetrievalEvaluator(k_values=[1, 3, 5])
    relevance_evaluator = RelevanceEvaluator(
        embedder=embedder,
        use_bertscore=use_bertscore,
    )

    # ── Step 2: Load test set ─────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"STEP 2 – Loading test set ({n_samples} samples)")
    logger.info("=" * 60)

    test_df = load_test_set(config_path, n_samples)

    # ── Step 3: Evaluate each query ───────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 3 – Running RAG pipeline on test queries")
    logger.info("=" * 60)

    all_results = []
    for i, row in test_df.iterrows():
        query          = row["instruction"]
        ground_truth   = row["response"]
        ground_intent  = row["intent"]

        logger.info(f"[{i+1}/{len(test_df)}] intent={ground_intent} | {query[:60]}...")

        # Run full RAG: retrieve + generate
        result = pipeline.run(query)

        # Feed to retrieval evaluator
        retrieval_evaluator.add(
            ground_truth_intent = ground_intent,
            retrieved_intents   = result.top_intents,
            query               = query,
        )

        # Feed to relevance evaluator
        relevance_evaluator.add(
            generated  = result.generated_answer,
            reference  = ground_truth,
            query      = query,
        )

        # Store full result for the detailed report
        all_results.append({
            "query":               query,
            "ground_truth_intent": ground_intent,
            "ground_truth_response": ground_truth[:300],
            "retrieved_intents":   result.top_intents,
            "retrieved_chunks":    [
                {"chunk_id": c["chunk_id"], "score": c["score"], "intent": c["intent"]}
                for c in result.retrieved_chunks
            ],
            "generated_answer":    result.generated_answer,
        })

    # ── Step 4: Compute metrics ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 4 – Computing metrics")
    logger.info("=" * 60)

    retrieval_metrics = retrieval_evaluator.compute()
    relevance_metrics = relevance_evaluator.compute()

    # ── Step 5: Save reports ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("STEP 5 – Saving reports")
    logger.info("=" * 60)

    # Detailed per-query results
    results_path = REPORTS_DIR / f"eval_results_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(
            {
                "timestamp":         timestamp,
                "n_samples":         n_samples,
                "retrieval_metrics": {k: v for k, v in retrieval_metrics.items() if k != "per_query"},
                "relevance_metrics": {k: v for k, v in relevance_metrics.items() if k != "per_query"},
                "per_query_retrieval": retrieval_metrics.get("per_query", []),
                "per_query_relevance": relevance_metrics.get("per_query", []),
                "raw_results":       all_results,
            },
            f,
            indent=2,
        )
    logger.info(f"Detailed results → {results_path}")

    # Clean summary report
    report = {
        "timestamp":    timestamp,
        "config":       config_path,
        "n_samples":    n_samples,
        "model":        "mistral (Ollama)",
        "embedder":     "BAAI/bge-base-en-v1.5",
        "retrieval": {
            k: v for k, v in retrieval_metrics.items()
            if k not in ("per_query", "num_queries")
        },
        "relevance": {
            k: v for k, v in relevance_metrics.items()
            if k not in ("per_query", "num_samples")
        },
        "interpretation": _interpret(retrieval_metrics, relevance_metrics),
    }

    report_path = REPORTS_DIR / "eval_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Summary report   → {report_path}")

    # ── Final print ───────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION COMPLETE")
    logger.info("=" * 60)
    _print_report(report)

    return report


# ── helpers ───────────────────────────────────────────────────────────────────

def _interpret(retrieval: dict, relevance: dict) -> dict:
    """
    Add human-readable interpretation of each metric score.
    Thresholds are standard for RAG systems in production.
    """
    def grade(val, thresholds):
        # thresholds: [(value, label), ...] descending
        for threshold, label in thresholds:
            if val >= threshold:
                return label
        return "Poor"

    hit_rate_5 = retrieval.get("hit_rate@5", 0)
    mrr_5      = retrieval.get("mrr@5", 0)
    cosine     = relevance.get("mean_cosine_similarity", 0)

    return {
        "hit_rate@5": grade(hit_rate_5, [(0.85, "Excellent"), (0.70, "Good"), (0.50, "Fair")]),
        "mrr@5":      grade(mrr_5,      [(0.75, "Excellent"), (0.55, "Good"), (0.35, "Fair")]),
        "cosine_sim": grade(cosine,     [(0.80, "Excellent"), (0.65, "Good"), (0.50, "Fair")]),
    }


def _print_report(report: dict):
    r = report["retrieval"]
    v = report["relevance"]
    i = report["interpretation"]
    print(f"""
╔══════════════════════════════════════════════════════╗
║           RAG EVALUATION REPORT                      ║
╠══════════════════════════════════════════════════════╣
║  Samples : {report['n_samples']:<10}  Model: {report['model']:<18}║
╠══════════════════════════════════════════════════════╣
║  RETRIEVAL QUALITY                                   ║
║  Hit Rate @1 : {r.get('hit_rate@1', 'N/A'):<8}  MRR @1 : {r.get('mrr@1', 'N/A'):<10}     ║
║  Hit Rate @3 : {r.get('hit_rate@3', 'N/A'):<8}  MRR @3 : {r.get('mrr@3', 'N/A'):<10}     ║
║  Hit Rate @5 : {r.get('hit_rate@5', 'N/A'):<8}  MRR @5 : {r.get('mrr@5', 'N/A'):<10}     ║
║  Verdict     : {i.get('hit_rate@5',''):<20}                     ║
╠══════════════════════════════════════════════════════╣
║  RELEVANCE SCORING                                   ║
║  Cosine Sim  : {v.get('mean_cosine_similarity', 'N/A'):<8}  Verdict: {i.get('cosine_sim',''):<15}  ║
║  BERTScore F1: {v.get('mean_bertscore_f1', 'N/A'):<8}                              ║
╚══════════════════════════════════════════════════════╝
""")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the RAG pipeline")
    parser.add_argument("--config",        default=CONFIG_PATH)
    parser.add_argument("--samples",       type=int, default=50)
    parser.add_argument("--no-bertscore",  action="store_true")
    args = parser.parse_args()

    run(
        config_path   = args.config,
        n_samples     = args.samples,
        use_bertscore = not args.no_bertscore,
    )