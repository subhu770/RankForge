"""
Unit tests for Dual-Encoder User Tower and Item Tower.
Verifies embedding dimensions, L2 normalization, mathematical properties, and semantic alignment.
"""

import numpy as np
import pytest

from app.models.item import Item
from app.models.user import UserProfile
from app.services.item_tower import item_tower
from app.services.user_tower import user_tower


class TestItemTower:
    """Tests for Candidate Item Tower projection."""

    def test_single_item_embedding_shape_and_norm(self, sample_items):
        """Item Tower must output exactly 64-dim vector with unit L2 norm."""
        item = sample_items[0]
        emb = item_tower.predict_one(item)

        assert isinstance(emb, np.ndarray)
        assert emb.shape == (64,)
        assert emb.dtype == np.float32

        norm = float(np.linalg.norm(emb))
        assert pytest.approx(norm, abs=1e-5) == 1.0

    def test_batch_item_embedding_shape_and_norms(self, sample_items):
        """Batch inference must output (N, 64) matrix with unit L2 norm on each row."""
        embs = item_tower.predict_batch(sample_items)

        assert isinstance(embs, np.ndarray)
        assert embs.shape == (len(sample_items), 64)

        norms = np.linalg.norm(embs, axis=-1)
        for norm in norms:
            assert pytest.approx(float(norm), abs=1e-5) == 1.0

    def test_empty_batch(self):
        """Item Tower should return empty array of shape (0, 64) for empty batch."""
        res = item_tower.predict_batch([])
        assert res.shape == (0, 64)

    def test_item_edge_cases(self):
        """Handles items with zero price, empty tags, and None description."""
        minimal_item = Item(
            item_id="min_01",
            title="Minimal Item",
            category="General",
            tags=[],
            price=0.0,
            popularity_score=0.0,
            rating=0.0,
            in_stock=False,
            description=None,
        )
        emb = item_tower.predict_one(minimal_item)
        assert emb.shape == (64,)
        assert pytest.approx(float(np.linalg.norm(emb)), abs=1e-5) == 1.0


class TestUserTower:
    """Tests for Query User Tower projection."""

    def test_single_user_embedding_shape_and_norm(self, sample_users):
        """User Tower must output exactly 64-dim vector with unit L2 norm."""
        user = sample_users[0]
        emb = user_tower.predict_one(user)

        assert isinstance(emb, np.ndarray)
        assert emb.shape == (64,)
        assert emb.dtype == np.float32

        norm = float(np.linalg.norm(emb))
        assert pytest.approx(norm, abs=1e-5) == 1.0

    def test_batch_user_embedding_shape_and_norms(self, sample_users):
        """Batch user inference must output (N, 64) matrix with unit L2 norm on each row."""
        embs = user_tower.predict_batch(sample_users)

        assert isinstance(embs, np.ndarray)
        assert embs.shape == (len(sample_users), 64)

        norms = np.linalg.norm(embs, axis=-1)
        for norm in norms:
            assert pytest.approx(float(norm), abs=1e-5) == 1.0

    def test_empty_batch(self):
        """User Tower should return empty array of shape (0, 64) for empty batch."""
        res = user_tower.predict_batch([])
        assert res.shape == (0, 64)

    def test_cold_start_user(self):
        """Handles user with no historical interactions, no affinities, and no search query."""
        cold_user = UserProfile(
            user_id="cold_usr",
            device=None,
            region=None,
            category_affinities={},
            historical_interactions=[],
            search_query=None,
            activity_level=0.0,
        )
        emb = user_tower.predict_one(cold_user)
        assert emb.shape == (64,)
        assert pytest.approx(float(np.linalg.norm(emb)), abs=1e-5) == 1.0


class TestDualEncoderSemanticAlignment:
    """Tests semantic alignment and cosine similarity mathematical correctness."""

    def test_category_affinity_semantic_alignment(self, sample_items):
        """
        User with strong affinity for Gaming should produce a higher cosine
        similarity with Gaming items than with Kitchen or Book items.
        """
        gamer = UserProfile(
            user_id="u_gamer",
            category_affinities={"Gaming": 1.0},
            historical_interactions=[],
        )
        u_emb = user_tower.predict_one(gamer)

        gaming_item = next(it for it in sample_items if it.category == "Gaming")
        kitchen_item = next(it for it in sample_items if it.category == "Home & Kitchen")
        book_item = next(it for it in sample_items if it.category == "Books & Literature")

        v_gaming = item_tower.predict_one(gaming_item)
        v_kitchen = item_tower.predict_one(kitchen_item)
        v_book = item_tower.predict_one(book_item)

        score_gaming = float(np.dot(u_emb, v_gaming))
        score_kitchen = float(np.dot(u_emb, v_kitchen))
        score_book = float(np.dot(u_emb, v_book))

        # Gaming score must be significantly higher than cross-category scores
        assert score_gaming > score_kitchen
        assert score_gaming > score_book
        assert score_gaming > 0.5

    def test_search_query_semantic_alignment(self, sample_items):
        """
        User with real-time search query 'espresso coffee machine'
        should score highest on the espresso machine candidate.
        """
        coffee_searcher = UserProfile(
            user_id="u_search",
            category_affinities={},
            search_query="espresso coffee machine steam",
        )
        u_emb = user_tower.predict_one(coffee_searcher)

        espresso_item = next(it for it in sample_items if "Espresso" in it.title)
        keyboard_item = next(it for it in sample_items if "Keyboard" in it.title)

        v_espresso = item_tower.predict_one(espresso_item)
        v_keyboard = item_tower.predict_one(keyboard_item)

        score_espresso = float(np.dot(u_emb, v_espresso))
        score_keyboard = float(np.dot(u_emb, v_keyboard))

        assert score_espresso > score_keyboard

    def test_cosine_similarity_bounds(self, sample_users, sample_items):
        """All dot products of unit vectors must lie strictly in [-1.0, 1.0]."""
        u_embs = user_tower.predict_batch(sample_users)
        i_embs = item_tower.predict_batch(sample_items)

        # Matrix multiplication computes all pairwise cosine similarities
        sim_matrix = np.dot(u_embs, i_embs.T)
        assert sim_matrix.shape == (len(sample_users), len(sample_items))

        assert np.all(sim_matrix >= -1.0 - 1e-6)
        assert np.all(sim_matrix <= 1.0 + 1e-6)

