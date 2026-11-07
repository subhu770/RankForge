"""
Pytest configuration and test fixtures for RankForge.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.models.item import Item
from app.models.user import UserProfile
from app.services.indexer import VectorIndexer, default_indexer
from app.services.item_tower import item_tower
from app.services.user_tower import user_tower


@pytest.fixture(scope="session")
def client() -> TestClient:
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def clean_indexer() -> VectorIndexer:
    """Provides a dedicated, fresh in-memory VectorIndexer for testing."""
    return VectorIndexer(dim=64, backend_type="cosine_simd", max_capacity=10_000)


@pytest.fixture
def sample_items() -> list[Item]:
    """Sample items across categories."""
    return [
        Item(
            item_id="gam_01",
            title="Pro Wireless Gaming Mouse RGB",
            category="Gaming",
            tags=["gaming", "wireless", "rgb"],
            price=79.99,
            popularity_score=0.9,
            rating=4.8,
            in_stock=True,
            description="Ultra-fast esports wireless mouse",
        ),
        Item(
            item_id="gam_02",
            title="Mechanical Gaming Keyboard TKL",
            category="Gaming",
            tags=["gaming", "keyboard", "rgb"],
            price=129.99,
            popularity_score=0.85,
            rating=4.6,
            in_stock=True,
            description="Tactile mechanical switches",
        ),
        Item(
            item_id="kit_01",
            title="Espresso Coffee Machine with Steam Wand",
            category="Home & Kitchen",
            tags=["coffee", "espresso", "kitchen"],
            price=299.99,
            popularity_score=0.75,
            rating=4.5,
            in_stock=True,
            description="Italian pump 15-bar espresso maker",
        ),
        Item(
            item_id="kit_02",
            title="Ceramic Non-Stick Frying Pan 10-inch",
            category="Home & Kitchen",
            tags=["cooking", "cookware", "kitchen"],
            price=39.99,
            popularity_score=0.6,
            rating=4.2,
            in_stock=False,  # Out of stock
            description="Toxin-free nonstick skillet",
        ),
        Item(
            item_id="boo_01",
            title="Designing Data-Intensive Applications",
            category="Books & Literature",
            tags=["books", "engineering", "database"],
            price=45.00,
            popularity_score=0.95,
            rating=4.9,
            in_stock=True,
            description="The definitive guide to distributed data systems",
        ),
    ]


@pytest.fixture
def sample_users() -> list[UserProfile]:
    """Sample user profiles with distinct affinities and contexts."""
    return [
        UserProfile(
            user_id="usr_gamer",
            device="desktop",
            region="US",
            category_affinities={"Gaming": 1.0},
            search_query="wireless optical mouse",
            historical_interactions=["gam_01"],
            activity_level=0.9,
        ),
        UserProfile(
            user_id="usr_barista",
            device="mobile",
            region="EU",
            category_affinities={"Home & Kitchen": 1.0},
            search_query="espresso coffee machine",
            historical_interactions=[],
            activity_level=0.5,
        ),
        UserProfile(
            user_id="usr_empty",
            device="mobile",
            region="US",
            category_affinities={},
            search_query=None,
            historical_interactions=[],
            activity_level=0.1,
        ),
    ]
