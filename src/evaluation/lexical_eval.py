"""
lexical_eval.py
───────────────
BLEU + ROUGE evaluation for generated answers vs ground-truth responses.

Why this exists
───────────────
The original project spec (MS2, Task 2) explicitly asks for BLEU and ROUGE.
Our main relevance metrics (BGE cosine similarity, BERTScore F1) live in
`relevance_eval.py` — they are paraphrase-robust and were chosen as the
team's preferred semantic measures. This module covers the spec's literal
ask with the standard lexical-overlap metrics:

    BLEU       — n-gram precision (NLTK, with smoothing for short outputs)
    ROUGE-1    — unigram recall  (rouge-score)
    ROUGE-2    — bigram recall
    ROUGE-L    — longest-common-subsequence

It is intentionally standalone:
  - Does not import or modify any of the existing eval / RAG modules
  - Reads per-query (generated, reference) pairs from an existing
    `reports/eval_results_*.json` so it never needs Ollama / MongoDB /
    a re-run of the RAG pipeline.

CLI
───
    # Auto-pick the most recent eval_results_*.json
    python -m src.evaluation.lexical_eval

    # Or point at a specific file
    python -m src.evaluation.lexical_eval --input reports/eval_results_20260314_031907.json

Output
──────
    reports/lexical_eval_<timestamp>.json   — per-query + aggregate
    reports/lexical_eval_report.json        — clean summary
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import nltk
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer

logger = logging.getLogger("lexical_eval")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
    datefmt="%H:%M:%S",
)

REPORTS_DIR = Path("reports")


# ── helpers ───────────────────────────────────────────────────────────────────

def _ensure_nltk_punkt() -> None:
    """BLEU tokenisation uses NLTK's word_tokenize (Punkt)."""
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)
    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        try:
            nltk.download("punkt_tab", quiet=True)
        except Exception:
            pass  # older nltk versions don't have punkt_tab


def _tokenize(text: str) -> list[str]:
    """Whitespace + light NLTK tokenisation, lowercased."""
    if not text:
        return []
    try:
        return nltk.word_tokenize(text.lower())
    except Exception:
        return text.lower().split()


def _mean(xs: Iterable[float]) -> float:
    xs = list(xs)
    return float(sum(xs) / len(xs)) if xs else 0.0


# ── core evaluator ────────────────────────────────────────────────────────────

class LexicalEvaluator:
    """
    Compute BLEU + ROUGE-{1,2,L} between generated and reference texts.

    Usage
    -----
    evaluator = LexicalEvaluator()
    for gen, ref in zip(generated_texts, reference_texts):
        evaluator.add(generated=gen, reference=ref, query=q)
    metrics = evaluator.compute()
    """

    def __init__(self) -> None:
        _ensure_nltk_punkt()
        self._scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
        )
        self._smooth = SmoothingFunction().method1  # avoids 0 for short outputs
        self._records: list[dict] = []

    def add(self, generated: str, reference: str, query: str | None = None) -> None:
        self._records.append(
            {"query": query or "", "generated": generated or "", "reference": reference or ""}
        )

    def compute(self) -> dict:
        if not self._records:
            raise ValueError("No records added. Call .add() first.")

        per_query = []
        bleu_scores, r1_scores, r2_scores, rL_scores = [], [], [], []

        for rec in self._records:
            gen, ref = rec["generated"], rec["reference"]

            # BLEU (sentence-level, smoothed; references must be list-of-token-lists)
            gen_tokens = _tokenize(gen)
            ref_tokens = _tokenize(ref)
            if gen_tokens and ref_tokens:
                bleu = sentence_bleu(
                    [ref_tokens],
                    gen_tokens,
                    smoothing_function=self._smooth,
                )
            else:
                bleu = 0.0

            # ROUGE
            rouge = self._scorer.score(ref, gen)
            r1 = rouge["rouge1"].fmeasure
            r2 = rouge["rouge2"].fmeasure
            rL = rouge["rougeL"].fmeasure

            bleu_scores.append(bleu)
            r1_scores.append(r1)
            r2_scores.append(r2)
            rL_scores.append(rL)

            per_query.append(
                {
                    "query":     rec["query"][:80],
                    "generated": gen[:150],
                    "reference": ref[:150],
                    "bleu":      round(bleu, 4),
                    "rouge1_f1": round(r1,   4),
                    "rouge2_f1": round(r2,   4),
                    "rougeL_f1": round(rL,   4),
                }
            )

        metrics = {
            "num_samples":    len(self._records),
            "mean_bleu":      round(_mean(bleu_scores), 4),
            "mean_rouge1_f1": round(_mean(r1_scores),   4),
            "mean_rouge2_f1": round(_mean(r2_scores),   4),
            "mean_rougeL_f1": round(_mean(rL_scores),   4),
            "per_query":      per_query,
        }
        self._log_summary(metrics)
        return metrics

    def _log_summary(self, m: dict) -> None:
        logger.info("─── Lexical Evaluation Results ───")
        logger.info(f"  BLEU       : {m['mean_bleu']:.4f}")
        logger.info(f"  ROUGE-1 F1 : {m['mean_rouge1_f1']:.4f}")
        logger.info(f"  ROUGE-2 F1 : {m['mean_rouge2_f1']:.4f}")
        logger.info(f"  ROUGE-L F1 : {m['mean_rougeL_f1']:.4f}")


# ── runner: score an existing eval_results_*.json ─────────────────────────────

def _latest_eval_results() -> Path:
    """Find the most recent reports/eval_results_*.json."""
    candidates = sorted(REPORTS_DIR.glob("eval_results_*.json"))
    if not candidates:
        raise FileNotFoundError(
            "No reports/eval_results_*.json found. "
            "Run `python -m src.evaluation.run_evaluation` first."
        )
    return candidates[-1]


def run(input_path: Path | None = None) -> dict:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    src_path = Path(input_path) if input_path else _latest_eval_results()
    logger.info(f"Scoring lexical metrics from: {src_path}")

    with open(src_path) as f:
        eval_results = json.load(f)

    raw = eval_results.get("raw_results", [])
    if not raw:
        raise ValueError(
            f"'{src_path}' has no 'raw_results' — cannot extract gen/ref pairs."
        )

    evaluator = LexicalEvaluator()
    for r in raw:
        evaluator.add(
            generated=r.get("generated_answer", ""),
            reference=r.get("ground_truth_response", ""),
            query=r.get("query", ""),
        )

    metrics = evaluator.compute()

    # Detailed dump (per-query)
    detailed_path = REPORTS_DIR / f"lexical_eval_{timestamp}.json"
    with open(detailed_path, "w") as f:
        json.dump(
            {
                "timestamp":      timestamp,
                "source":         str(src_path),
                "n_samples":      metrics["num_samples"],
                "mean_bleu":      metrics["mean_bleu"],
                "mean_rouge1_f1": metrics["mean_rouge1_f1"],
                "mean_rouge2_f1": metrics["mean_rouge2_f1"],
                "mean_rougeL_f1": metrics["mean_rougeL_f1"],
                "per_query":      metrics["per_query"],
            },
            f,
            indent=2,
        )
    logger.info(f"Per-query results → {detailed_path}")

    # Clean summary
    report = {
        "timestamp":      timestamp,
        "source":         str(src_path),
        "n_samples":      metrics["num_samples"],
        "mean_bleu":      metrics["mean_bleu"],
        "mean_rouge1_f1": metrics["mean_rouge1_f1"],
        "mean_rouge2_f1": metrics["mean_rouge2_f1"],
        "mean_rougeL_f1": metrics["mean_rougeL_f1"],
        "notes": (
            "BLEU/ROUGE measure n-gram overlap. Customer-support answers are "
            "paraphrased, so these scores are typically modest even when "
            "answers are semantically correct. For paraphrase-robust scoring "
            "see relevance_eval.py (BGE cosine + BERTScore)."
        ),
    }
    summary_path = REPORTS_DIR / "lexical_eval_report.json"
    with open(summary_path, "w") as f:
        json.dump(report, f, indent=2)
    logger.info(f"Summary report    → {summary_path}")

    return report


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute BLEU + ROUGE from an existing eval_results JSON."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Path to a reports/eval_results_*.json file "
             "(default: most recent in reports/).",
    )
    args = parser.parse_args()
    run(input_path=Path(args.input) if args.input else None)
