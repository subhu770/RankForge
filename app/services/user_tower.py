"""
User Tower (Query Encoder) for RankForge.

Projects user demographic context, explicit/implicit category affinities,
interaction history, and real-time query tokens into a normalized 64-dimensional embedding vector.
"""

import logging
from typing import List, Optional
import numpy as np

from app.core.config import settings
from app.models.user import UserProfile
from app.services.embedding_space import (
    SharedEmbeddingSpace,
    gelu,
    l2_normalize,
    layer_norm,
    shared_space,
)

logger = logging.getLogger("rankforge.user_tower")


class UserTower:
    """
    User Query Tower neural projection model.
    Transforms heterogeneous user profile and behavioral signals into the shared 64-dim latent space.
    """

    def __init__(
        self,
        shared_embedding_space: Optional[SharedEmbeddingSpace] = None,
        embedding_dim: int = 64,
        seed: int = 42,
    ):
        self.space = shared_embedding_space or shared_space
        self.embedding_dim = embedding_dim
        self.seed = seed

        # Input dimensionalities:
        # Category Affinities (32) + History (32) + Search Query (32) + Context (16) = 112
        self.cat_dim = self.space.category_dim
        self.text_dim = self.space.text_dim
        self.context_raw_dim = 8 + 8 + 2  # device (8) + region (8) + activity & count (2) = 18
        self.context_projected_dim = 16
        self.input_dim = self.cat_dim + self.cat_dim + self.text_dim + self.context_projected_dim
        self.hidden_dim = 128

        # Dual-encoder architecture: align base projection seeds with ItemTower
        # to ensure strong semantic synchronization in the shared embedding space
        rng = np.random.RandomState(seed + 10)

        # Context projection weights: 18 -> 16
        ctx_scale = np.sqrt(2.0 / (self.context_raw_dim + self.context_projected_dim))
        self.w_ctx = rng.randn(self.context_raw_dim, self.context_projected_dim).astype(np.float32) * ctx_scale
        self.b_ctx = np.zeros(self.context_projected_dim, dtype=np.float32)

        # MLP Layer 1: 112 -> 128
        w1_scale = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        self.w1 = rng.randn(self.input_dim, self.hidden_dim).astype(np.float32) * w1_scale
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)

        # MLP Layer 2 (Projection to 64-dim embedding space): 128 -> 64
        w2_scale = np.sqrt(2.0 / (self.hidden_dim + self.embedding_dim))
        self.w2 = rng.randn(self.hidden_dim, self.embedding_dim).astype(np.float32) * w2_scale
        self.b2 = np.zeros(self.embedding_dim, dtype=np.float32)

    def _extract_user_features(self, user: UserProfile) -> tuple[np.ndarray, np.ndarray]:
        """Extract both raw concatenated input and aligned shared residual representation."""
        # 1. Category affinity vector (32-dim)
        if user.category_affinities:
            cats = list(user.category_affinities.keys())
            weights = list(user.category_affinities.values())
            v_aff = self.space.pool_categories(cats, weights=weights)
        else:
            v_aff = np.zeros(self.cat_dim, dtype=np.float32)

        # 2. Historical interactions pooling (32-dim)
        v_hist = self.space.pool_categories(user.historical_interactions)

        # 3. Real-time search query pooling (32-dim text + multi-token category pooling)
        v_query = self.space.pool_text(user.search_query)
        query_tokens = self.space.tokenize(user.search_query)
        v_query_cat = self.space.pool_categories(query_tokens) if query_tokens else np.zeros(self.cat_dim, dtype=np.float32)

        # 4. Contextual and demographic signals (18-dim -> projected to 16-dim)
        device_key = (user.device or "mobile").lower()
        v_dev = self.space.device_table.get(device_key, self.space.device_table["mobile"])

        region_key = (user.region or "us").lower()
        v_reg = self.space.region_table.get(region_key, self.space.region_table["us"])

        act_level = float(np.clip(user.activity_level, 0.0, 1.0))
        hist_count = float(np.log1p(len(user.historical_interactions)) / 5.0)
        v_scalars = np.array([act_level, hist_count], dtype=np.float32)

        raw_ctx = np.concatenate([v_dev, v_reg, v_scalars], axis=0)
        v_ctx = np.dot(raw_ctx, self.w_ctx) + self.b_ctx

        # Concatenate into 112-dim input vector for MLP
        mlp_input = np.concatenate([v_aff, v_hist, v_query, v_ctx], axis=0)

        # Semantic aligned residual projection: affinities + history + search query directly into 64-dim space
        residual = (
            np.dot(v_aff, self.space.cat_to_latent)
            + 0.5 * np.dot(v_hist, self.space.cat_to_latent)
            + 0.8 * np.dot(v_query_cat, self.space.cat_to_latent)
            + 0.8 * np.dot(v_query, self.space.text_to_latent)
        )

        return mlp_input, residual

    def predict_one(self, user: UserProfile) -> np.ndarray:
        """
        Projects a single user profile into a 64-dimensional L2-normalized embedding.
        
        Returns:
            np.ndarray of shape (64,), with unit L2 norm.
        """
        x, residual = self._extract_user_features(user)

        # Dense Layer 1 with GELU & LayerNorm
        h1 = gelu(np.dot(x, self.w1) + self.b1)
        h1_norm = layer_norm(h1)

        # Dense Layer 2 (Projection)
        mlp_out = np.dot(h1_norm, self.w2) + self.b2

        # Combine aligned semantic residual projection with MLP non-linear features
        res_norm = l2_normalize(residual) if np.linalg.norm(residual) > 1e-6 else np.zeros_like(residual)
        mlp_norm = l2_normalize(mlp_out)
        out = res_norm + 0.2 * mlp_norm

        # Final L2 normalization
        return l2_normalize(out)

    def predict_batch(self, users: List[UserProfile]) -> np.ndarray:
        """
        Vectorized batch projection for multiple user profiles.
        
        Args:
            users: List of UserProfile objects.
            
        Returns:
            np.ndarray of shape (len(users), 64), where each row has unit L2 norm.
        """
        if not users:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        extracted = [self._extract_user_features(user) for user in users]
        feature_matrix = np.vstack([e[0] for e in extracted])
        residuals = np.vstack([e[1] for e in extracted])

        # Batch forward pass: (N, 112) -> (N, 128)
        h1 = gelu(np.dot(feature_matrix, self.w1) + self.b1)
        h1_norm = layer_norm(h1)

        # (N, 128) -> (N, 64)
        mlp_out = np.dot(h1_norm, self.w2) + self.b2

        res_norm = l2_normalize(residuals)
        mlp_norm = l2_normalize(mlp_out)
        out = res_norm + 0.2 * mlp_norm

        # Final batch L2 normalization
        return l2_normalize(out)


# Global default instance
user_tower = UserTower(
    shared_embedding_space=shared_space,
    embedding_dim=settings.EMBEDDING_DIM,
    seed=settings.RANDOM_SEED
)
