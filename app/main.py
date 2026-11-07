"""
Main FastAPI Application Entrypoint for RankForge.
"""

from contextlib import asynccontextmanager
import time
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.api.routes import router as core_router
from app.core.config import settings
from app.core.logging import logger
from app.services.indexer import default_indexer


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager handling startup and shutdown procedures."""
    logger.info(
        f"Starting {settings.PROJECT_NAME} v{settings.VERSION} "
        f"[Backend: {default_indexer.backend_name}, Dim: {default_indexer.dim}]"
    )
    yield
    logger.info(f"Shutting down {settings.PROJECT_NAME}. Cleaning up resources.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description=settings.PROJECT_DESCRIPTION,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Adds X-Process-Time-Ms header to all outgoing HTTP responses."""
    start_time = time.perf_counter()
    response = await call_next(request)
    process_time = (time.perf_counter() - start_time) * 1000.0
    response.headers["X-Process-Time-Ms"] = f"{process_time:.3f}"
    return response


# Include Routers
app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(core_router)


@app.get("/", summary="Root metadata and welcome endpoint", tags=["Root"])
def root_endpoint():
    """Returns general service information and available endpoint documentation."""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "description": settings.PROJECT_DESCRIPTION,
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "health_check": "/health",
        "metrics_url": "/metrics",
        "vector_backend": default_indexer.backend_name,
        "total_items_indexed": default_indexer.total_items,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        log_level=settings.LOG_LEVEL.lower(),
        reload=False,
    )

