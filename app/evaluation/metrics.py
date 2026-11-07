"""
Offline Recommendation Quality & Ranking Evaluation Metrics for RankForge.

Computes:
- HitRate@K
- Recall@K
- Precision@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- MRR (Mean Reciprocal Rank)
- MAP@K (Mean Average Precision)
- Intra-List Diversity (ILD)
"""

import math
from typing import Dict, List, Sequence, Set
import numpy as np

from app.services.embedding_space import l2_normalize


def hit_rate_at_k(recommended: Sequence[str], ground_truth: Set[str], k: int) -> float:
    """Returns 1.0 if any ground-truth item appears in top-k recommendations, else 0.0."""
    top_k = recommended[:k]
    return 1.0 if any(item in ground_truth for item in top_k) else 0.0


def recall_at_k(recommended: Sequence[str], ground_truth: Set[str], k: int) -> float:
    """Computes Recall@K: fraction of relevant items retrieved in top-k."""
    if not ground_truth:
        return 0.0
    top_k = set(recommended[:k])
    hits = len(top_k.intersection(ground_truth))
    return hits / float(len(ground_truth))


def precision_at_k(recommended: Sequence[str], ground_truth: Set[str], k: int) -> float:
    """Computes Precision@K: fraction of top-k items that are relevant."""
    if k <= 0:
        return 0.0
    top_k = set(recommended[:k])
    hits = len(top_k.intersection(ground_truth))
    return hits / float(k)


def reciprocal_rank(recommended: Sequence[str], ground_truth: Set[str]) -> float:
    """Computes Reciprocal Rank (RR): reciprocal of the 1-based rank of the first hit."""
    for rank_idx, item in enumerate(recommended, start=1):
        if item in ground_truth:
            return 1.0 / float(rank_idx)
    return 0.0


def ndcg_at_k(recommended: Sequence[str], ground_truth: Set[str], k: int) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at K (NDCG@K) with binary relevance.
    """
    top_k = recommended[:k]
    dcg = 0.0
    for idx, item in enumerate(top_k):
        if item in ground_truth:
            # Discounted gain at rank idx + 1: 1.0 / log2(rank + 1)
            dcg += 1.0 / math.log2((idx + 1) + 1.0)

    # Ideal DCG (all hits placed at the highest available ranks)
    num_hits_possible = min(len(ground_truth), k)
    if num_hits_possible == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(rank + 1.0) for rank in range(1, num_hits_possible + 1))
    return dcg / idcg if idcg > 0 else 0.0


def intra_list_diversity(embeddings: np.ndarray) -> float:
    """
    Computes Intra-List Diversity (ILD) based on average pairwise cosine distance (1 - cosine).
    High ILD indicates diverse, non-redundant item selections.
    """
    k = len(embeddings)
    if k <= 1:
        return 0.0

    norm_embs = l2_normalize(embeddings)
    sim_matrix = np.dot(norm_embs, norm_embs.T)

    # Sum upper triangular elements (excluding diagonal)
    triu_indices = np.triu_indices(k, k=1)
    pairwise_distances = 1.0 - sim_matrix[triu_indices]

    return float(np.mean(pairwise_distances))


class RankingEvaluator:
    """Batch evaluation engine computing ranking metrics across multiple users/sessions."""

    @staticmethod
    def evaluate_batch(
        recommendations: Dict[str, List[str]],
        ground_truth: Dict[str, Set[str]],
        k_list: List[int] = [5, 10],
    ) -> Dict[str, float]:
        """
        Evaluates a set of user recommendations against ground truth holdout sets.
        
        Args:
            recommendations: Dict mapping user_id -> list of recommended item_ids.
            ground_truth: Dict mapping user_id -> set of true positive item_ids.
            k_list: List of K cutoffs to evaluate (e.g. [5, 10]).
            
        Returns:
            Dictionary of aggregate mean metrics: HitRate@K, Recall@K, Precision@K, NDCG@K, MRR.
        """
        metrics: Dict[str, List[float]] = {f"hit_rate@{k}": [] for k in k_list}
        metrics.update({f"recall@{k}": [] for k in k_list})
        metrics.update({f"precision@{k}": [] for k in k_list})
        metrics.update({f"ndcg@{k}": [] for k in k_list})
        metrics["mrr"] = []

        for user_id, rec_items in recommendations.items():
            true_items = ground_truth.get(user_id, set())
            if not true_items:
                continue

            metrics["mrr"].append(reciprocal_rank(rec_items, true_items))

            for k in k_list:
                metrics[f"hit_rate@{k}"].append(hit_rate_at_k(rec_items, true_items, k))
                metrics[f"recall@{k}"].append(recall_at_k(rec_items, true_items, k))
                metrics[f"precision@{k}"].append(precision_at_k(rec_items, true_items, k))
                metrics[f"ndcg@{k}"].append(ndcg_at_k(rec_items, true_items, k))

        # Aggregate averages
        summary: Dict[str, float] = {}
        for m_name, vals in metrics.items():
            summary[m_name] = round(float(np.mean(vals)) if vals else 0.0, 4)

        return summary
