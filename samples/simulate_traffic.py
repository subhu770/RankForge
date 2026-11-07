"""
Traffic Simulation Script for RankForge:
Sends live HTTP requests against the running FastAPI REST API to test
item ingestion, candidate recommendations, and health endpoints over HTTP.
"""

import os
import sys
import time
import httpx

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from samples.bootstrap_catalog import CATEGORIES_DATA, generate_synthetic_catalog

API_BASE_URL = os.getenv("RANKFORGE_URL", "http://127.0.0.1:8000")


def run_traffic_simulation():
    """Executes end-to-end traffic simulation against the RankForge REST API."""
    print(f"[*] Targeting RankForge API at: {API_BASE_URL}")

    with httpx.Client(base_url=API_BASE_URL, timeout=10.0) as client:
        # 1. Health Check
        print("[1] Querying GET /health...")
        health_resp = client.get("/health")
        if health_resp.status_code != 200:
            print(f"[!] Health check failed: {health_resp.status_code} {health_resp.text}")
            return
        print(f"    -> Status: {health_resp.json()}\n")

        # 2. Ingest 200 items in a batch
        print("[2] Batch ingesting 200 items via POST /api/v1/items...")
        items = generate_synthetic_catalog(200)
        payload = {"items": [item.model_dump() for item in items]}

        t0 = time.perf_counter()
        ingest_resp = client.post("/api/v1/items", json=payload)
        ingest_ms = (time.perf_counter() - t0) * 1000.0

        if ingest_resp.status_code == 201:
            print(f"    -> Success: {ingest_resp.json()}")
            print(f"    -> Client-side turnaround time: {ingest_ms:.2f} ms\n")
        else:
            print(f"[!] Ingestion failed: {ingest_resp.status_code} {ingest_resp.text}")
            return

        # 3. Request recommendations
        print("[3] Requesting recommendations via POST /api/v1/recommend...")
        rec_payload = {
            "user_profile": {
                "user_id": "sim_user_99",
                "device": "desktop",
                "region": "US",
                "age_bucket": "25-34",
                "historical_interactions": [],
                "category_affinities": {"Gaming": 0.95},
                "search_query": "ergonomic mechanical gaming keyboard",
                "activity_level": 0.85,
            },
            "top_k": 5,
            "in_stock_only": True,
            "min_rating": 3.5,
        }

        t1 = time.perf_counter()
        rec_resp = client.post("/api/v1/recommend", json=rec_payload)
        rec_ms = (time.perf_counter() - t1) * 1000.0

        if rec_resp.status_code == 200:
            data = rec_resp.json()
            print(f"    -> Retrieved {len(data['candidates'])} candidates in {rec_ms:.2f} ms client-time")
            print(f"    -> Server Latency Breakdown: {data['latency_breakdown']}")
            for cand in data["candidates"]:
                print(f"       #{cand['rank']} [Score: {cand['similarity_score']:+.4f}] [{cand['category']}] {cand['title']}")
        else:
            print(f"[!] Recommend request failed: {rec_resp.status_code} {rec_resp.text}")

        # 4. Check Metrics
        print("\n[4] Querying GET /metrics...")
        metrics_resp = client.get("/metrics")
        if metrics_resp.status_code == 200:
            print(f"    -> Telemetry: {metrics_resp.json()}")


if __name__ == "__main__":
    run_traffic_simulation()

