"""
Item metadata and ingestion Pydantic models for RankForge.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class Item(BaseModel):
    """
    Catalog item schema containing semantic metadata, categorical attributes,
    and numerical signals used by ItemTower.
    """
    item_id: str = Field(..., description="Unique item SKU or identifier")
    title: str = Field(..., description="Item display title")
    category: str = Field(..., description="Primary catalog category")
    tags: List[str] = Field(default_factory=list, description="Subcategory tags or descriptive tokens")
    price: float = Field(default=0.0, ge=0.0, description="Item price in USD")
    popularity_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Normalized item popularity/velocity score (0.0 to 1.0)"
    )
    rating: float = Field(
        default=4.0,
        ge=0.0,
        le=5.0,
        description="Average customer review rating"
    )
    in_stock: bool = Field(default=True, description="Inventory availability flag")
    description: Optional[str] = Field(default=None, description="Detailed product description")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary auxiliary attributes")

    model_config = {
        "json_schema_extra": {
            "example": {
                "item_id": "itm_elec_482",
                "title": "Quantum ANC Wireless Noise-Cancelling Headphones",
                "category": "Electronics",
                "tags": ["audio", "wireless", "bluetooth", "noise-canceling"],
                "price": 199.99,
                "popularity_score": 0.92,
                "rating": 4.7,
                "in_stock": True,
                "description": "High-fidelity over-ear headphones with 40-hour battery life.",
                "metadata": {"brand": "AcousticLab", "year": 2025}
            }
        }
    }


class ItemBatchIngestRequest(BaseModel):
    """Batch ingestion request containing multiple catalog items."""
    items: List[Item] = Field(..., min_length=1, description="List of items to embed and index")


class ItemBatchIngestResponse(BaseModel):
    """Batch ingestion response with summary metrics."""
    inserted_count: int = Field(..., description="Count of successfully ingested items")
    total_indexed: int = Field(..., description="Total items currently indexed in memory")
    ingestion_latency_ms: float = Field(..., description="Total batch ingestion time in milliseconds")

