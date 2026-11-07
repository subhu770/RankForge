"""
Item Tower (Candidate Encoder) for RankForge.

Projects catalog item metadata, categorical tags, textual descriptions,
and numerical signals into a normalized 64-dimensional embedding vector.
"""

import logging
from typing import List, Optional
import numpy as np

from app.core.config import settings
from app.models.item import Item
from app.services.embedding_space import (
    SharedEmbeddingSpace,
    gelu,
    l2_normalize,
    layer_norm,
    shared_space,
)

logger = logging.getLogger("rankforge.item_tower")


class ItemTower:
    """
    Candidate Item Tower neural projection model.
    Transforms heterogeneous item metadata into a normalized latent vector.
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
        # Category (32) + Tags (32) + Text (32) + Numerical (16) = 112
        self.cat_dim = self.space.category_dim
        self.text_dim = self.space.text_dim
        self.num_raw_dim = 4
        self.num_projected_dim = 16
        self.input_dim = self.cat_dim + self.cat_dim + self.text_dim + self.num_projected_dim
        self.hidden_dim = 128

        rng = np.random.RandomState(seed + 10)

        # Numerical projection weights: 4 -> 16
        num_scale = np.sqrt(2.0 / (self.num_raw_dim + self.num_projected_dim))
        self.w_num = rng.randn(self.num_raw_dim, self.num_projected_dim).astype(np.float32) * num_scale
        self.b_num = np.zeros(self.num_projected_dim, dtype=np.float32)

        # MLP Layer 1: 112 -> 128
        w1_scale = np.sqrt(2.0 / (self.input_dim + self.hidden_dim))
        self.w1 = rng.randn(self.input_dim, self.hidden_dim).astype(np.float32) * w1_scale
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)

        # MLP Layer 2 (Projection to 64-dim embedding space): 128 -> 64
        w2_scale = np.sqrt(2.0 / (self.hidden_dim + self.embedding_dim))
        self.w2 = rng.randn(self.hidden_dim, self.embedding_dim).astype(np.float32) * w2_scale
        self.b2 = np.zeros(self.embedding_dim, dtype=np.float32)

    def _extract_item_features(self, item: Item) -> tuple[np.ndarray, np.ndarray]:
        """Extract both raw concatenated input and aligned shared residual representation."""
        # 1. Primary category embedding (32-dim)
        v_cat = self.space.get_category_vector(item.category)

        # 2. Pooled multi-hot tags embedding (32-dim)
        v_tags = self.space.pool_categories(item.tags)

        # 3. Pooled title + description text embedding (32-dim)
        full_text = f"{item.title} {item.description or ''}"
        v_text = self.space.pool_text(full_text)

        # 4. Numerical signals (4-dim -> projected to 16-dim)
        norm_price = float(np.log1p(max(item.price, 0.0)) / np.log1p(10000.0))
        pop = float(np.clip(item.popularity_score, 0.0, 1.0))
        rating = float(np.clip(item.rating / 5.0, 0.0, 1.0))
        in_stock = 1.0 if item.in_stock else 0.0

        raw_num = np.array([norm_price, pop, rating, in_stock], dtype=np.float32)
        v_num = np.dot(raw_num, self.w_num) + self.b_num

        # Concatenate into 112-dim input vector for MLP
        mlp_input = np.concatenate([v_cat, v_tags, v_text, v_num], axis=0)

        # Semantic aligned residual projection: category + tags + text directly into 64-dim space
        residual = (
            np.dot(v_cat, self.space.cat_to_latent)
            + 0.8 * np.dot(v_tags, self.space.cat_to_latent)
            + 0.8 * np.dot(v_text, self.space.text_to_latent)
        )

        return mlp_input, residual

    def predict_one(self, item: Item) -> np.ndarray:
        """
        Projects a single catalog item into a 64-dimensional L2-normalized embedding.
        
        Returns:
            np.ndarray of shape (64,), with unit L2 norm.
        """
        x, residual = self._extract_item_features(item)
        
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

    def predict_batch(self, items: List[Item]) -> np.ndarray:
        """
        Vectorized batch projection for multiple catalog items.
        
        Args:
            items: List of Item objects.
            
        Returns:
            np.ndarray of shape (len(items), 64), where each row has unit L2 norm.
        """
        if not items:
            return np.empty((0, self.embedding_dim), dtype=np.float32)

        extracted = [self._extract_item_features(item) for item in items]
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
item_tower = ItemTower(
    shared_embedding_space=shared_space,
    embedding_dim=settings.EMBEDDING_DIM,
    seed=settings.RANDOM_SEED
)
