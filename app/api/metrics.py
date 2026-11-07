"""
Performance telemetry and operational metrics endpoints for RankForge.
"""

from typing import Any, Dict
from fastapi import APIRouter
from app.core.telemetry import telemetry
from app.services.indexer import default_indexer

router = APIRouter(tags=["Metrics"])


@router.get("/metrics", summary="Performance telemetry and latency percentiles")
def get_metrics() -> Dict[str, Any]:
    """
    Returns query throughput counters, index capacity stats, and
    p50/p95/p99 latency metrics in milliseconds.
    """
    summary = telemetry.get_summary()
    summary["index_status"] = {
        "backend": default_indexer.backend_name,
        "total_items": default_indexer.total_items,
        "memory_bytes": default_indexer.memory_bytes,
    }
    return summary

