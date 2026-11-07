"""
Health check endpoints for RankForge.
"""

from typing import Any, Dict
from fastapi import APIRouter
from app.core.config import settings
from app.services.indexer import default_indexer

router = APIRouter(tags=["Health"])


@router.get("/health", summary="Service health and index status")
def get_health() -> Dict[str, Any]:
    """
    Returns operational health status, index backend, item count, and memory footprint.
    """
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "backend": default_indexer.backend_name,
        "dim": default_indexer.dim,
        "total_items_indexed": default_indexer.total_items,
        "estimated_index_memory_bytes": default_indexer.memory_bytes,
    }

