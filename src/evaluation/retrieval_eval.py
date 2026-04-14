"""
retrieval_eval.py
─────────────────
Computes retrieval quality metrics given a list of queries, their
ground-truth intents, and the chunks retrieved by the vector store.

Metrics
───────

Hit Rate @ K
    "Was the correct intent present anywhere in the top-K results?"
    Binary per query (1 or 0), averaged over the test set.
    → Tells you: does your retriever find the right answer at all?

MRR — Mean Reciprocal Rank
    For each query, find the rank position of the first correct chunk,
    score it as 1/rank. Average over all queries.
    MRR @ 1 = 1.0 (perfect), found at rank 5 = 0.2
    → Tells you: how high up in the list is the correct answer?

Precision @ K
    Of the K chunks returned, what fraction have the correct intent?
    → Tells you: is the retriever noisy or focused?

Why intent-based ground truth?
    Your corpus_test.parquet has ground-truth intent labels per query.
    A retrieved chunk is "relevant" if its intent matches the query's
    ground-truth intent. This is a principled proxy for relevance
    without needing human annotation of every query-chunk pair.
"""

import logging
from collections import defaultdict
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RetrievalEvaluator:
    """
    Computes Hit Rate, MRR, and Precision@K for a batch of queries.

    Usage
    -----
    evaluator = RetrievalEvaluator(k_values=[1, 3, 5])

    # Feed results one by one
    for query_row, rag_result in zip(test_df, results):
        evaluator.add(
            ground_truth_intent = query_row["intent"],
            retrieved_intents   = rag_result.top_intents,
        )

    metrics = evaluator.compute()
    """

    def __init__(self, k_values: list[int] = [1, 3, 5]):
        self.k_values = k_values
        self._records: list[dict] = []   # one entry per evaluated query

    def add(
        self,
        ground_truth_intent: str,
        retrieved_intents: list[str],
        query: Optional[str] = None,
    ):
        """
        Record the result for one query.

        Parameters
        ----------
        ground_truth_intent : the correct intent label from corpus_test
        retrieved_intents   : list of intents from retrieved chunks,
                              ordered by descending similarity score
        query               : optional — stored for inspection in report
        """
        gt = ground_truth_intent.strip().lower()

        # Find the rank of the first correct hit (1-indexed, 0 = not found)
        first_hit_rank = 0
        for rank, intent in enumerate(retrieved_intents, start=1):
            if intent.strip().lower() == gt:
                first_hit_rank = rank
                break

        self._records.append({
            "query":              query or "",
            "ground_truth_intent": gt,
            "retrieved_intents":  retrieved_intents,
            "first_hit_rank":     first_hit_rank,
        })

    def compute(self) -> dict:
        """
        Compute all metrics over all recorded queries.

        Returns
        -------
        dict with keys:
            hit_rate@K, mrr@K, precision@K for each K in k_values,
            plus per_query_scores for inspection
        """
        if not self._records:
            raise ValueError("No records added yet. Call .add() first.")

        n = len(self._records)
        metrics: dict = {"num_queries": n}

        for k in self.k_values:
            hit_rates  = []
            precisions = []
            rr_scores  = []   # reciprocal rank

            for rec in self._records:
                gt    = rec["ground_truth_intent"]
                ranks = rec["retrieved_intents"][:k]

                # Hit Rate @ K — was ground truth intent anywhere in top-K?
                hit = int(any(r.strip().lower() == gt for r in ranks))
                hit_rates.append(hit)

                # Precision @ K — fraction of top-K that are correct
                correct_count = sum(
                    1 for r in ranks if r.strip().lower() == gt
                )
                precisions.append(correct_count / k)

                # MRR @ K — 1/rank of first correct hit (0 if not in top-K)
                fhr = rec["first_hit_rank"]
                rr  = (1.0 / fhr) if (0 < fhr <= k) else 0.0
                rr_scores.append(rr)

            metrics[f"hit_rate@{k}"]  = round(float(np.mean(hit_rates)),  4)
            metrics[f"mrr@{k}"]       = round(float(np.mean(rr_scores)),   4)
            metrics[f"precision@{k}"] = round(float(np.mean(precisions)),  4)

        # Per-query breakdown (useful for finding failure cases)
        metrics["per_query"] = [
            {
                "query":               r["query"][:80],
                "ground_truth_intent": r["ground_truth_intent"],
                "retrieved_intents":   r["retrieved_intents"],
                "first_hit_rank":      r["first_hit_rank"],
                "hit@5":               int(0 < r["first_hit_rank"] <= 5),
                "rr@5":                round(
                    1.0 / r["first_hit_rank"] if 0 < r["first_hit_rank"] <= 5 else 0.0, 4
                ),
            }
            for r in self._records
        ]

        self._log_summary(metrics)
        return metrics

    def _log_summary(self, metrics: dict):
        logger.info("─── Retrieval Evaluation Results ───")
        for k in self.k_values:
            logger.info(
                f"  K={k} │ Hit Rate: {metrics[f'hit_rate@{k}']:.4f} │ "
                f"MRR: {metrics[f'mrr@{k}']:.4f} │ "
                f"Precision: {metrics[f'precision@{k}']:.4f}"
            )

    def reset(self):
        self._records.clear()