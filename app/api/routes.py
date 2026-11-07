"""
Candidate Retrieval & Item Ingestion API Routes for RankForge.
"""

import logging
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, status

from app.core.telemetry import telemetry
from app.models.item import (
    Item,
    ItemBatchIngestRequest,
    ItemBatchIngestResponse,
)
from app.models.recommendation import (
    CandidateItem,
    IndexRebuildResponse,
    LatencyBreakdown,
    RecommendationRequest,
    RecommendationResponse,
)
from app.services.indexer import default_indexer
from app.services.item_tower import item_tower
from app.services.user_tower import user_tower

logger = logging.getLogger("rankforge.routes")

router = APIRouter(prefix="/api/v1", tags=["Candidate Retrieval & Ranking"])


@router.post(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Retrieve ranked candidate items for a user profile",
    status_code=status.HTTP_200_OK,
)
def recommend_candidates(request: RecommendationRequest) -> RecommendationResponse:
    """
    Candidate retrieval endpoint:
    1. Ingests user profile, real-time query, and interaction context.
    2. Runs UserTower inference to project user state into a 64-dim normalized embedding.
    3. Retrieves top-K candidates from the in-memory ANN vector index using normalized dot-product scoring.
    4. Applies real-time filtering (category, min_rating, stock status, interaction deduplication).
    5. Returns ranked candidates with similarity scores and detailed sub-millisecond latency breakdown.
    """
    total_start = time.perf_counter()

    # Step 1: User Tower Inference
    user_start = time.perf_counter()
    user_vector = user_tower.predict_one(request.user_profile)
    user_tower_latency_ms = (time.perf_counter() - user_start) * 1000.0

    # Step 2: Build Real-time Metadata Filter Closure
    excluded_ids = set(request.user_profile.historical_interactions) if request.exclude_interacted else set()
    category_filter = request.category_filter.strip().lower() if request.category_filter else None

    def candidate_filter(meta: dict) -> bool:
        item_id = meta.get("item_id", "")
        if item_id in excluded_ids:
            return False

        if request.in_stock_only and not meta.get("in_stock", True):
            return False

        if request.min_rating is not None and request.min_rating > 0:
            if float(meta.get("rating", 0.0)) < request.min_rating:
                return False

        if category_filter:
            item_cat = str(meta.get("category", "")).strip().lower()
            if item_cat != category_filter:
                return False

        return True

    # Step 3: Fast In-Memory ANN Vector Retrieval
    retrieval_start = time.perf_counter()
    raw_results = default_indexer.query(
        vector=user_vector,
        top_k=request.top_k,
        filter_fn=candidate_filter,
    )
    retrieval_latency_ms = (time.perf_counter() - retrieval_start) * 1000.0

    # Step 4: Format and Rank Candidates
    filtering_start = time.perf_counter()
    candidates: List[CandidateItem] = []
    for rank_idx, (item_id, similarity_score, meta) in enumerate(raw_results, start=1):
        candidates.append(
            CandidateItem(
                rank=rank_idx,
                item_id=item_id,
                title=meta.get("title", f"Item {item_id}"),
                category=meta.get("category", "General"),
                tags=meta.get("tags", []),
                price=float(meta.get("price", 0.0)),
                rating=float(meta.get("rating", 0.0)),
                popularity_score=float(meta.get("popularity_score", 0.0)),
                similarity_score=round(float(similarity_score), 5),
            )
        )
    filtering_latency_ms = (time.perf_counter() - filtering_start) * 1000.0
    total_latency_ms = (time.perf_counter() - total_start) * 1000.0

    # Step 5: Record Telemetry
    telemetry.record_recommendation(
        total_latency_ms=total_latency_ms,
        retrieval_latency_ms=retrieval_latency_ms,
        user_tower_latency_ms=user_tower_latency_ms,
    )

    logger.debug(
        f"Recommendation query for user='{request.user_profile.user_id}' "
        f"retrieved {len(candidates)} items in {total_latency_ms:.3f} ms "
        f"(UserTower: {user_tower_latency_ms:.3f} ms, Retrieval: {retrieval_latency_ms:.3f} ms)"
    )

    return RecommendationResponse(
        user_id=request.user_profile.user_id,
        top_k=len(candidates),
        total_candidates_searched=default_indexer.total_items,
        candidates=candidates,
        latency_breakdown=LatencyBreakdown(
            user_tower_latency_ms=round(user_tower_latency_ms, 3),
            retrieval_latency_ms=round(retrieval_latency_ms, 3),
            filtering_latency_ms=round(filtering_latency_ms, 3),
            total_latency_ms=round(total_latency_ms, 3),
        ),
    )


@router.post(
    "/items",
    response_model=ItemBatchIngestResponse,
    summary="Batch ingest and embed catalog items into the vector index",
    status_code=status.HTTP_201_CREATED,
)
def batch_ingest_items(request: ItemBatchIngestRequest) -> ItemBatchIngestResponse:
    """
    Batch item ingestion endpoint:
    1. Validates incoming catalog items.
    2. Runs ItemTower batch inference to compute 64-dim normalized item embeddings.
    3. Adds embeddings and metadata into the live in-memory vector index.
    4. Updates telemetry metrics.
    """
    if not request.items:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Item batch cannot be empty."
        )

    start_time = time.perf_counter()
    items = request.items

    # 1. Batch inference through ItemTower
    embeddings = item_tower.predict_batch(items)

    # 2. Extract IDs and serializable metadata
    item_ids = [item.item_id for item in items]
    metadatas = [item.model_dump() for item in items]

    # 3. Add to live index
    inserted_count = default_indexer.add_items(
        embeddings=embeddings,
        ids=item_ids,
        metadatas=metadatas,
    )

    ingestion_latency_ms = (time.perf_counter() - start_time) * 1000.0
    telemetry.record_ingestion(inserted_count)

    logger.info(
        f"Batch ingested {inserted_count} items into vector index "
        f"in {ingestion_latency_ms:.2f} ms"
    )

    return ItemBatchIngestResponse(
        inserted_count=inserted_count,
        total_indexed=default_indexer.total_items,
        ingestion_latency_ms=round(ingestion_latency_ms, 3),
    )


@router.get(
    "/items/{item_id}",
    summary="Retrieve catalog metadata for an indexed item",
    status_code=status.HTTP_200_OK,
)
def get_item(item_id: str) -> Dict[str, Any]:
    """Retrieve indexed item metadata by its unique ID."""
    meta = default_indexer.get_item_metadata(item_id)
    if not meta:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Item with ID '{item_id}' not found in vector index."
        )
    return {
        "item_id": item_id,
        "indexed": True,
        "metadata": meta,
    }


@router.post(
    "/index/rebuild",
    response_model=IndexRebuildResponse,
    summary="Rebuild and optimize the in-memory vector index",
    status_code=status.HTTP_200_OK,
)
def rebuild_index() -> IndexRebuildResponse:
    """
    Rebuilds the active vector index from cached normalized embeddings,
    optimizing memory allocation and graph structure.
    """
    rebuild_latency_ms = default_indexer.rebuild()
    telemetry.record_rebuild()

    return IndexRebuildResponse(
        status="Index successfully rebuilt",
        items_indexed=default_indexer.total_items,
        rebuild_latency_ms=round(rebuild_latency_ms, 3),
        backend=default_indexer.backend_name,
    )

