"""
Configuration settings for RankForge using Pydantic Settings.
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # Project metadata
    PROJECT_NAME: str = "RankForge"
    PROJECT_DESCRIPTION: str = (
        "Production-grade candidate retrieval and ranking engine implementing "
        "a Dual-Encoder (Two-Tower) architecture with in-memory ANN vector indexing."
    )
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"

    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    LOG_LEVEL: str = "INFO"

    # Vector Architecture settings
    EMBEDDING_DIM: int = 64
    INDEX_BACKEND: Literal["auto", "hnsw", "cosine_simd"] = "auto"
    MAX_INDEX_CAPACITY: int = 100_000
    
    # HNSW ANN Hyperparameters (used when backend is hnsw)
    HNSW_M: int = 16
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_SEARCH: int = 50

    # Retrieval Defaults
    DEFAULT_TOP_K: int = 10
    MAX_TOP_K: int = 100

    # Determinism
    RANDOM_SEED: int = 42


settings = Settings()

