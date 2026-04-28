#!/usr/bin/env python
"""
Automation script for RAG optimization and evaluation.

This script performs the complete optimization workflow:
1. Rebuild chunks with enriched metadata
2. Re-embed with BGE-M3
3. Run evaluation
4. Generate comparison report
"""

import json
import logging
import sys
import subprocess
from pathlib import Path
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent


def run_command(cmd, description):
    """Run a command and log progress."""
    logger.info(f"\n{'='*60}")
    logger.info(f"STEP: {description}")
    logger.info(f"{'='*60}")
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=False)
        if result.returncode != 0:
            logger.error(f"❌ Command failed with return code {result.returncode}")
            return False
        logger.info("✅ Command completed successfully")
        return True
    except Exception as e:
        logger.error(f"❌ Error executing command: {e}")
        return False


def main():
    logger.info("\n" + "="*60)
    logger.info("RAG SYSTEM OPTIMIZATION & EVALUATION")
    logger.info("="*60)
    logger.info("Optimizations to apply:")
    logger.info("  1. Rich Chunking (add category/intent metadata)")
    logger.info("  2. BGE-M3 Embedding (upgraded model, 1024-dim)")
    logger.info("  3. Few-Shot Prompt (improved generation)")
    logger.info("  4. Clean Context Formatting (reduce noise)")
    logger.info("\n")

    # Step 1: Rebuild chunks with new format
    if not run_command(
        [sys.executable, "src/preprocess_data.py"],
        "Rebuild chunks with enriched metadata"
    ):
        logger.error("Chunk rebuild failed. Aborting.")
        return False

    # Step 2: Re-embed corpus with BGE-M3
    if not run_command(
        [sys.executable, "src/vector_store/build_store.py"],
        "Re-embed corpus with BGE-M3 model"
    ):
        logger.error("Re-embedding failed. Aborting.")
        return False

    # Step 3: Run evaluation
    if not run_command(
        [sys.executable, "src/evaluation/run_evaluation.py"],
        "Run evaluation pipeline"
    ):
        logger.error("Evaluation failed. Aborting.")
        return False

    # Step 4: Generate comparison report
    logger.info(f"\n{'='*60}")
    logger.info("STEP: Generate comparison report")
    logger.info(f"{'='*60}")
    
    try:
        baseline_file = PROJECT_ROOT / "reports/eval_report.json"
        results_file = sorted(PROJECT_ROOT.glob("reports/eval_results_*.json"))[-1]
        output_file = PROJECT_ROOT / "reports/evaluation_after_optimization_v2.json"
        
        logger.info(f"Loading baseline: {baseline_file}")
        with open(baseline_file) as f:
            baseline = json.load(f)
        
        logger.info(f"Loading results: {results_file}")
        with open(results_file) as f:
            results = json.load(f)
        
        # Extract just the summary
        after_optimization = {
            "timestamp": results.get("timestamp", ""),
            "config": results.get("config", "config/config.yaml"),
            "n_samples": results.get("n_samples", 50),
            "model": results.get("model", "mistral (Ollama)"),
            "embedder": "BAAI/bge-m3",  # Updated embedder
            "retrieval": results.get("retrieval", {}),
            "relevance": results.get("relevance", {}),
            "interpretation": results.get("interpretation", {}),
        }
        
        # Calculate improvements
        improvement = {
            "retrieval": {},
            "relevance": {},
        }
        
        for key in baseline.get("retrieval", {}).keys():
            if key in after_optimization["retrieval"]:
                improvement["retrieval"][key] = round(
                    after_optimization["retrieval"][key] - baseline["retrieval"][key],
                    4
                )
        
        for key in baseline.get("relevance", {}).keys():
            if key in after_optimization["relevance"]:
                improvement["relevance"][key] = round(
                    after_optimization["relevance"][key] - baseline["relevance"][key],
                    4
                )
        
        # Create comparison report
        report = {
            "optimization_summary": "Applied optimizations: (1) Rich chunking with category/intent metadata, (2) BGE-M3 embedding (1024-dim, 8K context), (3) Few-shot generation prompt, (4) Clean context formatting with noise reduction",
            "baseline": baseline,
            "after_optimization": after_optimization,
            "improvement": improvement,
            "optimization_timestamp": datetime.now().isoformat(),
        }
        
        # Save report
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Comparison report saved: {output_file}")
        
        # Print summary
        logger.info("\n" + "="*60)
        logger.info("OPTIMIZATION RESULTS SUMMARY")
        logger.info("="*60)
        
        print(f"\n{'METRIC':<30} {'BASELINE':<12} {'AFTER OPT':<12} {'IMPROVEMENT':<12}")
        print("-" * 66)
        
        # Retrieval metrics
        for key in ["hit_rate@5", "mrr@5", "precision@5"]:
            if key in baseline.get("retrieval", {}) and key in after_optimization.get("retrieval", {}):
                baseline_val = baseline["retrieval"][key]
                after_val = after_optimization["retrieval"][key]
                imp_val = improvement["retrieval"].get(key, 0)
                print(f"{key:<30} {baseline_val:<12.4f} {after_val:<12.4f} {imp_val:+.4f}")
        
        # Relevance metrics
        print()
        for key in ["mean_cosine_similarity", "mean_bertscore_f1"]:
            if key in baseline.get("relevance", {}) and key in after_optimization.get("relevance", {}):
                baseline_val = baseline["relevance"][key]
                after_val = after_optimization["relevance"][key]
                imp_val = improvement["relevance"].get(key, 0)
                pct_imp = (imp_val / baseline_val * 100) if baseline_val != 0 else 0
                print(f"{key:<30} {baseline_val:<12.4f} {after_val:<12.4f} {imp_val:+.4f} ({pct_imp:+.1f}%)")
        
        logger.info("\n✅ OPTIMIZATION COMPLETE!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Error generating report: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
