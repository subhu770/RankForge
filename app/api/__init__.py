"""
API routers packaging for RankForge.
"""

from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.api.routes import router as core_router

__all__ = ["health_router", "metrics_router", "core_router"]

