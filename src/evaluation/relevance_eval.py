"""
relevance_eval.py
─────────────────
Measures how semantically similar the generated answer is to the
ground-truth response using two complementary metrics.

Metric 1 — BGE Cosine Similarity
    Uses the same BGE model already loaded for retrieval.
    Embeds both the generated answer and the ground truth response,
    then computes cosine similarity (0–1, higher = more similar).

    Why: Fast, no extra model download, consistent with how the
    retriever scores relevance — so scores are directly comparable.

Metric 2 — BERTScore (F1)
    Uses a pre-trained BERT model to compute token-level semantic
    overlap between generated and reference texts.
    Reports Precision, Recall, F1.

    Why: BERTScore is robust to paraphrasing — it rewards answers
    that mean the same thing even with different wording.
    This is important for customer support where agents rephrase
    standard responses in varied ways.

Why NOT BLEU/ROUGE here?
    BLEU and ROUGE require near-exact n-gram overlap. Customer support
    responses are naturally paraphrased, so a semantically correct
    answer scores poorly on BLEU. Semantic metrics are more meaningful.
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class RelevanceEvaluator:
    """
    Computes BGE cosine similarity and BERTScore between generated
    answers and ground truth responses.

    Usage
    -----
    evaluator = RelevanceEvaluator(embedder)   # pass the already-loaded Embedder

    for result, ground_truth_response in zip(rag_results, test_responses):
        evaluator.add(
            generated  = result.generated_answer,
            reference  = ground_truth_response,
            query      = result.query,
        )

    metrics = evaluator.compute()
    """

    def __init__(self, embedder, use_bertscore: bool = True):
        """
        Parameters
        ----------
        embedder      : loaded Embedder instance (reuses BGE — no extra model load)
        use_bertscore : set False to skip BERTScore (faster, no extra download)
        """
        self.embedder       = embedder
        self.use_bertscore  = use_bertscore
        self._records: list[dict] = []

        if use_bertscore:
            try:
                from bert_score import score as bert_score_fn
                self._bert_score_fn = bert_score_fn
                logger.info("BERTScore loaded.")
            except ImportError:
                logger.warning(
                    "bert_score not installed. Run: pip install bert-score\n"
                    "Falling back to cosine similarity only."
                )
                self.use_bertscore = False

    def add(
        self,
        generated:  str,
        reference:  str,
        query:      Optional[str] = None,
    ):
        """
        Record one generated/reference pair for later scoring.

        We store strings here and batch-embed in compute() for efficiency.
        """
        self._records.append({
            "query":     query or "",
            "generated": generated,
            "reference": reference,
        })

    def compute(self) -> dict:
        """
        Compute all relevance metrics over recorded pairs.

        Returns
        -------
        dict with:
            mean_cosine_similarity   : float
            mean_bertscore_f1        : float (if enabled)
            per_query                : list of per-query scores
        """
        if not self._records:
            raise ValueError("No records added. Call .add() first.")

        generated_texts = [r["generated"] for r in self._records]
        reference_texts = [r["reference"] for r in self._records]

        # ── BGE Cosine Similarity ─────────────────────────────────────────────
        logger.info("Computing BGE cosine similarities...")
        gen_embeddings = self.embedder.encode_passages(generated_texts)
        ref_embeddings = self.embedder.encode_passages(reference_texts)

        # Both are L2-normalised → dot product = cosine similarity
        cosine_scores = np.sum(gen_embeddings * ref_embeddings, axis=1)
        cosine_scores = np.clip(cosine_scores, 0.0, 1.0)   # safety clip

        # ── BERTScore ─────────────────────────────────────────────────────────
        bertscore_f1s = None
        if self.use_bertscore:
            logger.info("Computing BERTScore (this may take a moment)...")
            try:
                P, R, F1 = self._bert_score_fn(
                    generated_texts,
                    reference_texts,
                    lang="en",
                    rescale_with_baseline=True,
                    verbose=False,
                )
                bertscore_f1s = F1.numpy()
            except Exception as e:
                logger.warning(f"BERTScore failed: {e}. Skipping.")

        # ── Assemble results ──────────────────────────────────────────────────
        per_query = []
        for i, rec in enumerate(self._records):
            entry = {
                "query":            rec["query"][:80],
                "generated":        rec["generated"][:150],
                "reference":        rec["reference"][:150],
                "cosine_similarity": round(float(cosine_scores[i]), 4),
            }
            if bertscore_f1s is not None:
                entry["bertscore_f1"] = round(float(bertscore_f1s[i]), 4)
            per_query.append(entry)

        metrics = {
            "num_samples":            len(self._records),
            "mean_cosine_similarity": round(float(np.mean(cosine_scores)), 4),
            "std_cosine_similarity":  round(float(np.std(cosine_scores)),  4),
            "per_query":              per_query,
        }

        if bertscore_f1s is not None:
            metrics["mean_bertscore_f1"] = round(float(np.mean(bertscore_f1s)), 4)
            metrics["std_bertscore_f1"]  = round(float(np.std(bertscore_f1s)),  4)

        self._log_summary(metrics)
        return metrics

    def _log_summary(self, metrics: dict):
        logger.info("─── Relevance Evaluation Results ───")
        logger.info(
            f"  Cosine Similarity: {metrics['mean_cosine_similarity']:.4f} "
            f"± {metrics['std_cosine_similarity']:.4f}"
        )
        if "mean_bertscore_f1" in metrics:
            logger.info(
                f"  BERTScore F1:      {metrics['mean_bertscore_f1']:.4f} "
                f"± {metrics['std_bertscore_f1']:.4f}"
            )

    def reset(self):
        self._records.clear()