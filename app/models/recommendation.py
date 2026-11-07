"""
Recommendation request and candidate response models for RankForge.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from app.models.user import UserProfile


class RecommendationRequest(BaseModel):
    """
    Candidate retrieval request specifying user profile and retrieval constraints.
    """
    user_profile: UserProfile = Field(..., description="User profile and real-time context")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of top candidates to retrieve")
    category_filter: Optional[str] = Field(
        default=None,
        description="Optional filter to restrict retrieval to a single primary category"
    )
    min_rating: Optional[float] = Field(
        default=0.0,
        ge=0.0,
        le=5.0,
        description="Minimum review rating threshold"
    )
    in_stock_only: bool = Field(
        default=True,
        description="If True, filters out out-of-stock items"
    )
    exclude_interacted: bool = Field(
        default=True,
        description="If True, filters out items already present in user historical_interactions"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_profile": {
                    "user_id": "usr_9912",
                    "device": "desktop",
                    "region": "EU",
                    "age_bucket": "25-34",
                    "historical_interactions": ["itm_gaming_010"],
                    "category_affinities": {
                        "Gaming": 0.95,
                        "Electronics": 0.70
                    },
                    "search_query": "ergonomic mechanical keyboard",
                    "activity_level": 0.85
                },
                "top_k": 5,
                "category_filter": None,
                "min_rating": 4.0,
                "in_stock_only": True,
                "exclude_interacted": True
            }
        }
    }


class CandidateItem(BaseModel):
    """Retrieved candidate item with cosine similarity score and catalog attributes."""
    rank: int = Field(..., description="1-based retrieval ranking position")
    item_id: str = Field(..., description="Unique item identifier")
    title: str = Field(..., description="Item title")
    category: str = Field(..., description="Item category")
    tags: List[str] = Field(default_factory=list, description="Descriptive tags")
    price: float = Field(..., description="Item price")
    rating: float = Field(..., description="Item rating")
    popularity_score: float = Field(..., description="Popularity score")
    similarity_score: float = Field(
        ...,
        description="Inner product / Cosine similarity between user and item embeddings (-1.0 to 1.0)"
    )


class LatencyBreakdown(BaseModel):
    """Sub-millisecond inference and retrieval latency breakdown."""
    user_tower_latency_ms: float = Field(..., description="User tower embedding inference latency")
    retrieval_latency_ms: float = Field(..., description="In-memory vector retrieval latency")
    filtering_latency_ms: float = Field(..., description="Metadata filtering and rank formatting latency")
    total_latency_ms: float = Field(..., description="End-to-end recommendation request latency")


class RecommendationResponse(BaseModel):
    """Response containing ranked candidates and performance telemetry."""
    user_id: str = Field(..., description="User ID for whom recommendations were generated")
    top_k: int = Field(..., description="Number of items returned")
    total_candidates_searched: int = Field(..., description="Total index candidate space searched")
    candidates: List[CandidateItem] = Field(default_factory=list, description="Ranked list of candidates")
    latency_breakdown: LatencyBreakdown = Field(..., description="Execution latency breakdown in ms")


class IndexRebuildResponse(BaseModel):
    """Status response for vector index rebuild operation."""
    status: str = Field(..., description="Status message")
    items_indexed: int = Field(..., description="Total items in rebuilt index")
    rebuild_latency_ms: float = Field(..., description="Time taken to rebuild index in ms")
    backend: str = Field(..., description="Active vector index backend (hnsw or cosine_simd)")

