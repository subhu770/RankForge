"""
Telemetry and performance metrics tracking for RankForge.
"""

from collections import deque
import threading
import time
from typing import Dict, List, Optional
import numpy as np


class MetricsTracker:
    """Thread-safe telemetry tracker recording query latencies and operational counters."""

    def __init__(self, max_history: int = 5000):
        self._lock = threading.Lock()
        self.start_time = time.time()
        self.total_recommend_requests = 0
        self.total_items_ingested = 0
        self.total_index_rebuilds = 0
        
        # Ring buffer of recent latencies in milliseconds
        self._recommend_latencies: deque = deque(maxlen=max_history)
        self._retrieval_latencies: deque = deque(maxlen=max_history)
        self._user_tower_latencies: deque = deque(maxlen=max_history)

    def record_recommendation(
        self,
        total_latency_ms: float,
        retrieval_latency_ms: float,
        user_tower_latency_ms: float,
    ) -> None:
        with self._lock:
            self.total_recommend_requests += 1
            self._recommend_latencies.append(total_latency_ms)
            self._retrieval_latencies.append(retrieval_latency_ms)
            self._user_tower_latencies.append(user_tower_latency_ms)

    def record_ingestion(self, count: int) -> None:
        with self._lock:
            self.total_items_ingested += count

    def record_rebuild(self) -> None:
        with self._lock:
            self.total_index_rebuilds += 1

    def get_summary(self) -> Dict[str, object]:
        with self._lock:
            uptime_seconds = time.time() - self.start_time
            latencies = list(self._recommend_latencies)
            retrievals = list(self._retrieval_latencies)

            def calc_percentiles(arr: List[float]) -> Dict[str, float]:
                if not arr:
                    return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}
                np_arr = np.array(arr)
                return {
                    "p50": round(float(np.percentile(np_arr, 50)), 3),
                    "p95": round(float(np.percentile(np_arr, 95)), 3),
                    "p99": round(float(np.percentile(np_arr, 99)), 3),
                    "mean": round(float(np.mean(np_arr)), 3),
                }

            return {
                "uptime_seconds": round(uptime_seconds, 2),
                "total_recommend_requests": self.total_recommend_requests,
                "total_items_ingested": self.total_items_ingested,
                "total_index_rebuilds": self.total_index_rebuilds,
                "recommendation_latency_ms": calc_percentiles(latencies),
                "retrieval_latency_ms": calc_percentiles(retrievals),
            }


# Global metrics tracker instance
telemetry = MetricsTracker()

