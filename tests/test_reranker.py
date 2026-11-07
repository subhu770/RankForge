"""
Unit tests for Diversity Reranker (MMR) and Evaluation Metrics.
"""

import numpy as np
import pytest

from app.evaluation.metrics import (
    RankingEvaluator,
    hit_rate_at_k,
    intra_list_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.services.reranker import reranker


def test_mmr_diversity():
    """MMR must prioritize diversity when lambda < 1.0."""
    query_vec = np.array([1.0, 0.0], dtype=np.float32)
    candidate_ids = ["item_1", "item_2", "item_3"]
    # item_1 and item_2 are identical vectors
    candidate_embs = np.array([
        [1.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
    ], dtype=np.float32)
    candidate_scores = np.array([0.95, 0.94, 0.70], dtype=np.float32)

    # With high diversity emphasis (lambda=0.3), item_3 should be selected before item_2
    selected = reranker.mmr_rerank(
        query_vector=query_vec,
        candidate_ids=candidate_ids,
        candidate_embeddings=candidate_embs,
        candidate_scores=candidate_scores,
        top_k=2,
        diversity_lambda=0.3,
    )
    assert selected == [0, 2]


def test_ranking_evaluation_metrics():
    """Validates NDCG, Recall, Precision, MRR, and HitRate math."""
    recommended = ["it_1", "it_2", "it_3", "it_4", "it_5"]
    ground_truth = {"it_2", "it_5"}

    hr_5 = hit_rate_at_k(recommended, ground_truth, k=5)
    assert hr_5 == 1.0

    hr_1 = hit_rate_at_k(recommended, ground_truth, k=1)
    assert hr_1 == 0.0

    rec_5 = recall_at_k(recommended, ground_truth, k=5)
    assert rec_5 == 1.0

    prec_5 = precision_at_k(recommended, ground_truth, k=5)
    assert prec_5 == 2.0 / 5.0

    mrr = reciprocal_rank(recommended, ground_truth)
    assert mrr == 1.0 / 2.0  # it_2 is at rank 2

    ndcg_5 = ndcg_at_k(recommended, ground_truth, k=5)
    assert 0.0 < ndcg_5 <= 1.0


def test_intra_list_diversity():
    """ILD of identical items is 0; orthogonal items is > 0."""
    identical = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    assert pytest.approx(intra_list_diversity(identical), abs=1e-5) == 0.0

    orthogonal = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    assert pytest.approx(intra_list_diversity(orthogonal), abs=1e-5) == 1.0
