# name: evaluate_rag.py
# description: Evaluation and threshold calibration script for RAG retrieval quality,
#              measuring Recall@K, Threshold Accuracy, Grounding Compliance, and Wilson 95% CIs.

import argparse
import asyncio
import json
import logging
import math
import random
from pathlib import Path
from typing import Any

from app.core.database import AsyncSessionFactory
from app.factory.pipeline_factory import create_embedder, create_llm, create_vector_store
from app.models.conversation import ConversationMode
from app.pipelines.naive_rag.pipeline import NaiveRagPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger(__name__)

REJECTION_PATTERNS: tuple[str, ...] = (
    "couldn't find",
    "could not find",
    "không tìm thấy",
    "không có thông tin",
    "tài liệu không nhắc đến",
    "tài liệu không có",
    "không thể tìm thấy",
    "không được đề cập",
    "không chứa thông tin",
)


def wilson_score_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Calculate the Wilson score confidence interval for a binomial proportion.

    Parameters:
        successes: Number of positive / accurate outcomes.
        total: Total number of trials.
        confidence: Confidence level (default: 0.95, corresponding to z=1.96).

    Returns:
        tuple[float, float, float]: (p_hat, lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0, 0.0

    p_hat = successes / total
    z = 1.95996  # 95% confidence z-score

    denominator = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2.0 * total)) / denominator
    margin = (z / denominator) * math.sqrt((p_hat * (1.0 - p_hat) / total) + (z**2) / (4.0 * (total**2)))

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return p_hat, lower, upper


def stratified_split(
    data: list[dict[str, Any]], dev_ratio: float = 0.7, seed: int = 42
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Split dataset into dev and test splits while preserving the distribution of each 'type'.

    Parameters:
        data: List of golden set items.
        dev_ratio: Fraction allocated to dev split (default: 0.7).
        seed: Random seed for reproducible shuffling.

    Returns:
        tuple: (dev_items, test_items)
    """
    rng = random.Random(seed)
    by_type: dict[str, list[dict[str, Any]]] = {}
    for item in data:
        by_type.setdefault(item.get("type", "unknown"), []).append(item)

    dev_split: list[dict[str, Any]] = []
    test_split: list[dict[str, Any]] = []
    for itype, group in sorted(by_type.items()):
        shuffled = list(group)
        rng.shuffle(shuffled)
        k = int(round(len(shuffled) * dev_ratio))
        dev_split.extend(shuffled[:k])
        test_split.extend(shuffled[k:])
        logger.debug("Type %s: %d total -> %d dev, %d test", itype, len(shuffled), k, len(shuffled) - k)

    return dev_split, test_split


class RagEvaluator:
    """
    Evaluator for RAG retrieval accuracy and model grounding compliance.
    """

    def __init__(self, golden_set_path: str | Path) -> None:
        self.golden_set_path = Path(golden_set_path)
        self.data: list[dict[str, Any]] = []

    def load_data(self) -> None:
        """Load golden set records from JSON file."""
        if not self.golden_set_path.exists():
            raise FileNotFoundError(f"Golden set file not found: {self.golden_set_path}")
        with open(self.golden_set_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        logger.info("Loaded %d golden set items from %s", len(self.data), self.golden_set_path)

    async def evaluate_split(
        self,
        items: list[dict[str, Any]],
        threshold: float,
        top_k: int = 5,
        eval_grounding: bool = True,
        skip_llm: bool = False,
    ) -> dict[str, Any]:
        """
        Evaluate a dataset split for Recall@K, Threshold Accuracy, and Grounding Compliance.

        Parameters:
            items: Golden set items belonging to the target split.
            threshold: Cosine similarity cutoff threshold.
            top_k: Number of retrieved chunks to consider.
            eval_grounding: Whether to evaluate Grounding Compliance on near-negatives.
            skip_llm: If True, skips actual LLM generation (fast retrieval-only mode).

        Returns:
            Dict containing scores and Wilson confidence intervals.
        """
        async with AsyncSessionFactory() as session:
            vector_store = create_vector_store(session)
            embedder = create_embedder()

            positives = [it for it in items if it.get("type") in ("positive_direct", "positive_followup")]
            negatives_far = [it for it in items if it.get("type") == "negative_far"]
            negatives_near = [it for it in items if it.get("type") == "negative_near"]

            # 1. Recall@K on positives
            recall_hits = 0
            for item in positives:
                query = item["query"]
                user_id = item["user_id"]
                target_doc_id = item.get("target_doc_id")
                target_chunk_idx = item.get("target_chunk_index")
                doc_ids = [target_doc_id] if target_doc_id else None

                query_vec = await asyncio.to_thread(embedder.embed, query)
                chunks = await vector_store.similarity_search(
                    query_vector=query_vec,
                    user_id=user_id,
                    top_k=top_k,
                    threshold=threshold,
                    document_ids=doc_ids,
                )

                hit = any(
                    str(c.document_id) == str(target_doc_id)
                    and (target_chunk_idx is None or c.metadata.get("chunk_index") == target_chunk_idx)
                    for c in chunks
                )
                if hit:
                    recall_hits += 1

            p_recall, rec_low, rec_high = wilson_score_interval(recall_hits, len(positives))

            # 2. Threshold Accuracy on far out-of-domain negatives (chunks should be 0)
            thresh_hits = 0
            for item in negatives_far:
                query = item["query"]
                user_id = item["user_id"]
                query_vec = await asyncio.to_thread(embedder.embed, query)
                chunks = await vector_store.similarity_search(
                    query_vector=query_vec,
                    user_id=user_id,
                    top_k=top_k,
                    threshold=threshold,
                )
                if len(chunks) == 0:
                    thresh_hits += 1

            p_thresh, th_low, th_high = wilson_score_interval(thresh_hits, len(negatives_far))

            # 3. Grounding Compliance on near-negatives (negatives_near)
            # Section 14 target: >= 70% of responses contain "couldn't find" or equivalent rejection
            grounding_hits = 0
            if eval_grounding and negatives_near:
                pipeline = None
                if not skip_llm:
                    try:
                        llm = create_llm()
                        pipeline = NaiveRagPipeline(
                            llm=llm,
                            vector_store=vector_store,
                            embedder=embedder,
                            db=session,
                            top_k=top_k,
                            similarity_threshold=threshold,
                        )
                    except Exception as exc:
                        logger.warning("Could not initialize LLM for grounding eval: %s", exc)

                for item in negatives_near:
                    query = item["query"]
                    user_id = item["user_id"]
                    target_doc_id = item.get("target_doc_id")
                    doc_ids = [target_doc_id] if target_doc_id else None

                    query_vec = await asyncio.to_thread(embedder.embed, query)
                    chunks = await vector_store.similarity_search(
                        query_vector=query_vec,
                        user_id=user_id,
                        top_k=top_k,
                        threshold=threshold,
                        document_ids=doc_ids,
                    )

                    if len(chunks) == 0:
                        # 0 chunks vượt threshold -> Hard Fallback tự động trả về thông báo từ chối
                        grounding_hits += 1
                    elif pipeline is not None and not skip_llm:
                        try:
                            # Chunks vượt threshold: Chạy pipeline với prompt DOCUMENT_QA strict
                            messages, prep_chunks = await pipeline.prepare_context(
                                user_message=query,
                                mode=ConversationMode.DOCUMENT_QA,
                                user_id=user_id,
                                user_level="B1",
                                conversation_history=[],
                                document_ids=doc_ids,
                                db=session,
                            )
                            tokens: list[str] = []
                            async for tok, _ in pipeline.stream_from_messages(messages, prep_chunks):
                                tokens.append(tok)
                            resp_text = "".join(tokens).lower()
                            if any(pat in resp_text for pat in REJECTION_PATTERNS):
                                grounding_hits += 1
                        except Exception as exc:
                            logger.warning("Grounding LLM evaluation failed for item query=%r: %s", query, exc)
                    else:
                        # LLM skipped or unavailable: only zero-chunk hard fallback counted
                        pass

            p_ground, gr_low, gr_high = wilson_score_interval(grounding_hits, len(negatives_near))

            result = {
                "threshold": threshold,
                "top_k": top_k,
                "recall_at_k": {
                    "score": round(p_recall, 4),
                    "ci_95": [round(rec_low, 4), round(rec_high, 4)],
                    "hits": recall_hits,
                    "total": len(positives),
                },
                "threshold_accuracy": {
                    "score": round(p_thresh, 4),
                    "ci_95": [round(th_low, 4), round(th_high, 4)],
                    "hits": thresh_hits,
                    "total": len(negatives_far),
                },
            }

            if eval_grounding and negatives_near:
                result["grounding_compliance"] = {
                    "score": round(p_ground, 4),
                    "ci_95": [round(gr_low, 4), round(gr_high, 4)],
                    "hits": grounding_hits,
                    "total": len(negatives_near),
                }

            return result

    async def sweep_thresholds(
        self,
        dev_items: list[dict[str, Any]],
        start: float = 0.3,
        end: float = 0.7,
        step: float = 0.05,
    ) -> float:
        """
        Sweep threshold values on dev split to find optimal threshold.
        Goal: Maximize Recall@5 while keeping Threshold Accuracy >= 0.95.

        Returns:
            Optimal threshold float.
        """
        best_threshold = 0.5
        best_recall = -1.0
        current = start

        logger.info("=== Starting Threshold Sweep on Dev Split (%d items) ===", len(dev_items))

        while current <= end + 1e-6:
            # Sweep iterates purely over retrieval metrics (eval_grounding=False for speed)
            metrics = await self.evaluate_split(dev_items, threshold=round(current, 2), eval_grounding=False)
            rec = metrics["recall_at_k"]["score"]
            th_acc = metrics["threshold_accuracy"]["score"]

            logger.info(
                "Threshold %.2f -> Recall@5: %.2f%% (95%% CI: %s), Far-Negative Accuracy: %.2f%%",
                current,
                rec * 100,
                metrics["recall_at_k"]["ci_95"],
                th_acc * 100,
            )

            # Constraint: Threshold Accuracy >= 95%
            if th_acc >= 0.95 and rec > best_recall:
                best_recall = rec
                best_threshold = round(current, 2)

            current += step

        logger.info("Optimal threshold chosen from dev sweep: %.2f (Recall@5: %.2f%%)", best_threshold, best_recall * 100)
        return best_threshold


async def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG pipeline quality on Golden Set.")
    parser.add_argument("--golden-set", type=str, default="data/golden_set.json", help="Path to golden set JSON")
    parser.add_argument("--dev-split-ratio", type=float, default=0.7, help="Ratio for dev split (default: 0.7)")
    parser.add_argument("--sweep", action="store_true", help="Perform threshold sweep on dev split")
    parser.add_argument("--threshold", type=float, default=0.5, help="Fixed threshold to evaluate")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM execution during grounding compliance eval")
    parser.add_argument("--seed", type=int, default=42, help="Seed for stratified splitting")
    args = parser.parse_args()

    evaluator = RagEvaluator(args.golden_set)
    try:
        evaluator.load_data()
    except FileNotFoundError:
        logger.warning("Golden set file %s not found. Exiting eval.", args.golden_set)
        return

    # Stratified split: preserves distribution across all query types
    dev_items, test_items = stratified_split(evaluator.data, dev_ratio=args.dev_split_ratio, seed=args.seed)
    logger.info("Stratified split: %d dev items, %d test items (seed=%d)", len(dev_items), len(test_items), args.seed)

    if args.sweep:
        optimal_th = await evaluator.sweep_thresholds(dev_items)
        logger.info("=== Running Final Evaluation on Test Split (%d items) at threshold %.2f ===", len(test_items), optimal_th)
        test_results = await evaluator.evaluate_split(
            test_items, threshold=optimal_th, eval_grounding=True, skip_llm=args.skip_llm
        )
        print(json.dumps(test_results, indent=2))
    else:
        results = await evaluator.evaluate_split(
            test_items, threshold=args.threshold, eval_grounding=True, skip_llm=args.skip_llm
        )
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
