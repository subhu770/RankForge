"""
Integration tests for RankForge REST API endpoints.
Verifies /health, /metrics, /api/v1/items, /api/v1/recommend, /api/v1/index/rebuild, and error handling.
"""

import pytest
from fastapi.testclient import TestClient


class TestRankForgeAPI:
    """REST API endpoint tests using FastAPI TestClient."""

    def test_health_check(self, client: TestClient):
        """GET /health must return status 200 with service metadata."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "RankForge"
        assert "backend" in data
        assert "total_items_indexed" in data

    def test_root_endpoint(self, client: TestClient):
        """GET / must return API info and documentation routes."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "RankForge"
        assert data["docs_url"] == "/docs"

    def test_metrics_endpoint(self, client: TestClient):
        """GET /metrics must return telemetry metrics dictionary."""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "uptime_seconds" in data
        assert "recommendation_latency_ms" in data
        assert "retrieval_latency_ms" in data

    def test_batch_ingest_items(self, client: TestClient, sample_items):
        """POST /api/v1/items must ingest items and update index."""
        payload = {"items": [item.model_dump() for item in sample_items]}
        response = client.post("/api/v1/items", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["inserted_count"] == len(sample_items)
        assert data["total_indexed"] >= len(sample_items)
        assert data["ingestion_latency_ms"] > 0

    def test_get_item_metadata(self, client: TestClient):
        """GET /api/v1/items/{item_id} returns cached metadata for indexed items."""
        # Item gam_01 was ingested in previous test
        response = client.get("/api/v1/items/gam_01")
        assert response.status_code == 200
        data = response.json()
        assert data["item_id"] == "gam_01"
        assert data["indexed"] is True
        assert data["metadata"]["category"] == "Gaming"

    def test_get_item_not_found(self, client: TestClient):
        """GET /api/v1/items/non_existent returns 404 Not Found."""
        response = client.get("/api/v1/items/non_existent_item_id_999")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_recommend_candidates(self, client: TestClient):
        """POST /api/v1/recommend must return ranked candidates with similarity scores and latency breakdown."""
        req_payload = {
            "user_profile": {
                "user_id": "api_test_user",
                "device": "desktop",
                "region": "US",
                "age_bucket": "25-34",
                "historical_interactions": [],
                "category_affinities": {"Gaming": 1.0},
                "search_query": "gaming mouse wireless",
                "activity_level": 0.8,
            },
            "top_k": 3,
            "in_stock_only": True,
            "min_rating": 4.0,
            "exclude_interacted": True,
        }

        response = client.post("/api/v1/recommend", json=req_payload)
        assert response.status_code == 200
        data = response.json()

        assert data["user_id"] == "api_test_user"
        assert data["top_k"] <= 3
        assert len(data["candidates"]) > 0

        # Verify candidate structure
        for cand in data["candidates"]:
            assert "rank" in cand
            assert "item_id" in cand
            assert "similarity_score" in cand
            assert cand["rating"] >= 4.0

        # Verify rank ordering
        ranks = [c["rank"] for c in data["candidates"]]
        assert ranks == list(range(1, len(ranks) + 1))

        # Verify latency breakdown
        latency = data["latency_breakdown"]
        assert "user_tower_latency_ms" in latency
        assert "retrieval_latency_ms" in latency
        assert "total_latency_ms" in latency
        assert latency["total_latency_ms"] >= latency["retrieval_latency_ms"]

    def test_index_rebuild(self, client: TestClient):
        """POST /api/v1/index/rebuild rebuilds index and returns status 200."""
        response = client.post("/api/v1/index/rebuild")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "Index successfully rebuilt"
        assert data["items_indexed"] >= 1
        assert "rebuild_latency_ms" in data

    def test_ingest_empty_batch_validation_error(self, client: TestClient):
        """Ingesting empty items batch returns 422 or 400 validation error."""
        response = client.post("/api/v1/items", json={"items": []})
        assert response.status_code in [400, 422]

    def test_recommend_invalid_top_k_validation_error(self, client: TestClient):
        """Top_k <= 0 or > 100 triggers 422 validation error."""
        req_payload = {
            "user_profile": {"user_id": "test_err"},
            "top_k": 0,  # Invalid: ge=1
        }
        response = client.post("/api/v1/recommend", json=req_payload)
        assert response.status_code == 422

