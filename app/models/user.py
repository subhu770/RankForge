"""
User profile and context Pydantic models for RankForge.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class UserProfile(BaseModel):
    """
    User context and interaction state consumed by UserTower
    to project into the shared 64-dimensional embedding space.
    """
    user_id: str = Field(..., description="Unique user or session identifier")
    device: Optional[str] = Field(default="mobile", description="Device category: mobile, desktop, tablet, etc.")
    region: Optional[str] = Field(default="US", description="Geographic region code")
    age_bucket: Optional[str] = Field(default="25-34", description="Age bracket: 18-24, 25-34, 35-50, 50+")
    historical_interactions: List[str] = Field(
        default_factory=list,
        description="List of past item IDs or interacted entity tokens"
    )
    category_affinities: Dict[str, float] = Field(
        default_factory=dict,
        description="Affinity weights by category, where values typically range from 0.0 to 1.0"
    )
    search_query: Optional[str] = Field(
        default=None,
        description="Optional real-time search context or query terms"
    )
    activity_level: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="User activity level index between 0.0 (infrequent) and 1.0 (power user)"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "user_id": "usr_7721",
                "device": "mobile",
                "region": "US",
                "age_bucket": "25-34",
                "historical_interactions": ["itm_electronics_001", "itm_gaming_042"],
                "category_affinities": {
                    "Electronics": 0.85,
                    "Gaming": 0.90,
                    "Books": 0.15
                },
                "search_query": "wireless noise cancelling headset",
                "activity_level": 0.8
            }
        }
    }

