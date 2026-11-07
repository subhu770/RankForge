"""
Shared embedding projections and neural layers for RankForge Two-Tower architecture.

Provides deterministic feature hashing, shared category/text embedding tables,
and vectorized MLP operations with GELU activations and Layer Normalization.
"""

import hashlib
from typing import List, Optional
import numpy as np

from app.core.config import settings


def deterministic_hash(token: str, num_buckets: int) -> int:
    """Computes a deterministic bucket index for any string token."""
    h = hashlib.md5(token.strip().lower().encode("utf-8")).digest()
    int_val = int.from_bytes(h[:8], byteorder="big", signed=False)
    return int_val % num_buckets


def gelu(x: np.ndarray) -> np.ndarray:
    """Gaussian Error Linear Unit (GELU) activation function."""
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * np.power(x, 3))))


def layer_norm(x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Vectorized Layer Normalization over the last dimension."""
    if x.ndim == 1:
        mean = np.mean(x)
        var = np.var(x)
        return (x - mean) / np.sqrt(var + eps)
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def l2_normalize(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalizes vector or matrix along the last axis."""
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        norm = float(np.linalg.norm(vectors))
        return vectors / max(norm, eps)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms = np.maximum(norms, eps)
    return vectors / norms


class SharedEmbeddingSpace:
    """
    Shared latent parameter space for Two-Tower recommendation.
    Synchronizes the semantic representations of categories, tags, and context tokens
    between the User Tower and Item Tower.
    """

    def __init__(
        self,
        embedding_dim: int = 64,
        category_dim: int = 32,
        text_dim: int = 32,
        num_category_buckets: int = 512,
        num_text_buckets: int = 2048,
        seed: int = 42,
    ):
        self.embedding_dim = embedding_dim
        self.category_dim = category_dim
        self.text_dim = text_dim
        self.num_category_buckets = num_category_buckets
        self.num_text_buckets = num_text_buckets
        self.seed = seed

        # Initialize deterministic pseudo-random weights
        rng = np.random.RandomState(seed)

        # Category and tag shared embedding table (unit-normalized)
        raw_cat = rng.randn(num_category_buckets, category_dim).astype(np.float32)
        self.category_table = l2_normalize(raw_cat)

        # Text token shared embedding table (unit-normalized)
        raw_text = rng.randn(num_text_buckets, text_dim).astype(np.float32)
        self.text_table = l2_normalize(raw_text)

        # Orthonormal projection matrices mapping 32-dim feature spaces into 64-dim latent space
        q_cat, _ = np.linalg.qr(rng.randn(embedding_dim, category_dim).astype(np.float32))
        self.cat_to_latent = q_cat.T.astype(np.float32)

        q_text, _ = np.linalg.qr(rng.randn(embedding_dim, text_dim).astype(np.float32))
        self.text_to_latent = q_text.T.astype(np.float32)

        # Categorical demographic lookup tables for user context
        self.device_table = {
            "mobile": l2_normalize(rng.randn(8).astype(np.float32)),
            "desktop": l2_normalize(rng.randn(8).astype(np.float32)),
            "tablet": l2_normalize(rng.randn(8).astype(np.float32)),
            "tv": l2_normalize(rng.randn(8).astype(np.float32)),
        }
        self.region_table = {
            "us": l2_normalize(rng.randn(8).astype(np.float32)),
            "eu": l2_normalize(rng.randn(8).astype(np.float32)),
            "apac": l2_normalize(rng.randn(8).astype(np.float32)),
            "latam": l2_normalize(rng.randn(8).astype(np.float32)),
            "global": l2_normalize(rng.randn(8).astype(np.float32)),
        }

    def tokenize(self, text: Optional[str]) -> List[str]:
        """Fast delimiter-based tokenization without heavy dependencies."""
        if not text or not text.strip():
            return []
        cleaned = text.replace(",", " ").replace("-", " ").replace("/", " ").replace(":", " ")
        return [t.strip().lower() for t in cleaned.split() if len(t.strip()) > 1]

    def get_category_vector(self, category: str) -> np.ndarray:
        """Lookup embedding for a single category string."""
        bucket = deterministic_hash(category, self.num_category_buckets)
        return self.category_table[bucket].copy()

    def pool_categories(self, categories: List[str], weights: Optional[List[float]] = None) -> np.ndarray:
        """Weighted pooling of category embeddings."""
        if not categories:
            return np.zeros(self.category_dim, dtype=np.float32)

        indices = [deterministic_hash(c, self.num_category_buckets) for c in categories]
        vectors = self.category_table[indices]

        if weights is not None and len(weights) == len(categories):
            w = np.array(weights, dtype=np.float32).reshape(-1, 1)
            total_w = float(np.sum(np.abs(w)))
            if total_w > 1e-6:
                return l2_normalize(np.sum(vectors * w, axis=0))

        return l2_normalize(np.mean(vectors, axis=0))

    def pool_text(self, text: Optional[str]) -> np.ndarray:
        """Tokenize and pool text tokens against text embedding table."""
        tokens = self.tokenize(text)
        if not tokens:
            return np.zeros(self.text_dim, dtype=np.float32)

        indices = [deterministic_hash(t, self.num_text_buckets) for t in tokens]
        vectors = self.text_table[indices]
        return l2_normalize(np.mean(vectors, axis=0))


# Global shared instance synchronized with project settings
shared_space = SharedEmbeddingSpace(
    embedding_dim=settings.EMBEDDING_DIM,
    seed=settings.RANDOM_SEED
)
