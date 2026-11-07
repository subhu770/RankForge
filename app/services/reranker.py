"""
Re-Ranking & Diversity Engine for RankForge.

Implements:
1. Maximal Marginal Relevance (MMR) for candidate diversification.
2. Multi-Objective Scoring combining vector similarity with business signals
   (ratings, popularity priors, and category fatigue penalties).
"""

from collections import Counter
import logging
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from app.services.embedding_space import l2_normalize

logger = logging.getLogger("rankforge.reranker")


class DiversityReranker:
    """
    Maximal Marginal Relevance (MMR) and Multi-Objective Re-Ranker.
    
    Prevents recommendation list redundancy (e.g., 5 identical keyboards)
    by balancing candidate relevance against inter-item similarity.
    """

    @staticmethod
    def mmr_rerank(
        query_vector: np.ndarray,
        candidate_ids: List[str],
        candidate_embeddings: np.ndarray,
        candidate_scores: np.ndarray,
        top_k: int = 10,
        diversity_lambda: float = 0.7,
    ) -> List[int]:
        """
        Executes Maximal Marginal Relevance (MMR) re-ranking.
        
        Args:
            query_vector: 64-dim L2-normalized user query embedding.
            candidate_ids: List of candidate string IDs.
            candidate_embeddings: 2D array of shape (M, 64) containing candidate item embeddings.
            candidate_scores: Array of shape (M,) containing user-item similarity scores.
            top_k: Number of final diverse candidates to select.
            diversity_lambda: Balance between relevance (1.0) and diversity (0.0).
            
        Returns:
            List of integer indices into the original candidate pool, in ranked order.
        """
        n_candidates = len(candidate_ids)
        if n_candidates == 0 or top_k <= 0:
            return []

        actual_k = min(top_k, n_candidates)
        if diversity_lambda >= 1.0 or n_candidates <= 1:
            # Pure relevance sorting
            return list(np.argsort(-candidate_scores)[:actual_k])

        # Pre-normalize embeddings
        item_vectors = l2_normalize(candidate_embeddings)

        # Precompute item-item cosine similarity matrix: (M, M)
        sim_matrix = np.dot(item_vectors, item_vectors.T)

        selected_indices: List[int] = []
        unselected_indices = set(range(n_candidates))

        # First item is always the highest relevance candidate
        best_initial = int(np.argmax(candidate_scores))
        selected_indices.append(best_initial)
        unselected_indices.remove(best_initial)

        # Greedily select remaining K - 1 items using MMR criterion
        for _ in range(1, actual_k):
            if not unselected_indices:
                break

            best_mmr_score = -float("inf")
            best_candidate_idx = -1

            for c_idx in unselected_indices:
                relevance = float(candidate_scores[c_idx])
                
                # Maximum similarity to any already-selected item
                max_sim_to_selected = float(np.max(sim_matrix[c_idx, selected_indices]))

                # MMR Score = λ * Rel - (1 - λ) * MaxSim
                mmr_score = diversity_lambda * relevance - (1.0 - diversity_lambda) * max_sim_to_selected

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate_idx = c_idx

            selected_indices.append(best_candidate_idx)
            unselected_indices.remove(best_candidate_idx)

        return selected_indices

    @staticmethod
    def multi_objective_rerank(
        candidates: List[Tuple[str, float, dict]],
        weight_relevance: float = 0.70,
        weight_rating: float = 0.15,
        weight_popularity: float = 0.15,
        category_fatigue_penalty: float = 0.05,
    ) -> List[Tuple[str, float, dict]]:
        """
        Applies multi-objective scoring combining vector similarity with
        rating signals, popularity priors, and category repetition fatigue.
        """
        if not candidates:
            return []

        category_counts: Counter = Counter()
        rescored: List[Tuple[str, float, dict]] = []

        for item_id, sim_score, meta in candidates:
            cat = meta.get("category", "General")
            rating_norm = float(np.clip(float(meta.get("rating", 4.0)) / 5.0, 0.0, 1.0))
            pop_norm = float(np.clip(float(meta.get("popularity_score", 0.5)), 0.0, 1.0))

            # Penalty scales with number of previous items from the same category
            fatigue = category_counts[cat] * category_fatigue_penalty

            composite_score = (
                weight_relevance * sim_score
                + weight_rating * rating_norm
                + weight_popularity * pop_norm
                - fatigue
            )

            category_counts[cat] += 1
            rescored.append((item_id, float(composite_score), meta))

        # Sort descending by composite score
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored


# Global reranker singleton
reranker = DiversityReranker()
