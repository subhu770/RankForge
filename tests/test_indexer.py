"""
Unit tests for Vector Indexer and Approximate / Exact Nearest Neighbor retrieval.
Verifies Top-K rank ordering, cosine similarity math, recall@K, filtering, and latency.
"""

import time
import numpy as np
import pytest

from app.services.indexer import CosineMatrixIndex, VectorIndexer, l2_normalize


class TestVectorIndexer:
    """Tests for VectorIndexer managing candidate retrieval."""

    def test_add_and_query_ordering(self, clean_indexer):
        """Top-K candidates must be returned strictly in descending order of similarity."""
        np.random.seed(42)
        n_items = 50
        dim = 64
        embeddings = np.random.randn(n_items, dim).astype(np.float32)
        ids = [f"item_{i}" for i in range(n_items)]
        metadatas = [{"item_id": ids[i], "index": i} for i in range(n_items)]

        clean_indexer.add_items(embeddings, ids, metadatas=metadatas)
        assert clean_indexer.total_items == n_items

        query_vec = np.random.randn(dim).astype(np.float32)
        results = clean_indexer.query(query_vec, top_k=10)

        assert len(results) == 10
        scores = [score for _, score, _ in results]

        # Verify monotonic non-increasing ordering: score[i] >= score[i+1]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1]

    def test_exact_cosine_similarity_math(self, clean_indexer):
        """
        Verify that VectorIndexer scores match manual brute-force
        inner products of L2-normalized vectors.
        """
        np.random.seed(123)
        n = 20
        dim = 64
        raw_embs = np.random.randn(n, dim).astype(np.float32)
        ids = [f"doc_{i}" for i in range(n)]

        clean_indexer.add_items(raw_embs, ids)

        query_raw = np.random.randn(dim).astype(np.float32)
        q_norm = l2_normalize(query_raw)
        db_norm = l2_normalize(raw_embs)

        # Brute force manual dot products
        expected_scores = np.dot(db_norm, q_norm)
        sorted_indices = np.argsort(-expected_scores)

        results = clean_indexer.query(query_raw, top_k=5)

        for rank, (retrieved_id, retrieved_score, _) in enumerate(results):
            expected_idx = sorted_indices[rank]
            expected_id = ids[expected_idx]
            expected_score = float(expected_scores[expected_idx])

            assert retrieved_id == expected_id
            assert pytest.approx(retrieved_score, abs=1e-5) == expected_score

    def test_recall_at_k(self, clean_indexer):
        """
        Recall@10 against brute force must be 1.0 (100% precision)
        for the SIMD Cosine Matrix index.
        """
        np.random.seed(999)
        n_items = 200
        dim = 64
        embs = np.random.randn(n_items, dim).astype(np.float32)
        ids = [f"item_{i:03d}" for i in range(n_items)]
        clean_indexer.add_items(embs, ids)

        db_norm = l2_normalize(embs)
        k = 10

        recalls = []
        for _ in range(15):
            q = np.random.randn(dim).astype(np.float32)
            q_norm = l2_normalize(q)

            # Ground truth top-K
            ground_truth_indices = np.argsort(-np.dot(db_norm, q_norm))[:k]
            ground_truth_ids = set(ids[idx] for idx in ground_truth_indices)

            # Indexer top-K
            results = clean_indexer.query(q, top_k=k)
            retrieved_ids = set(item_id for item_id, _, _ in results)

            # Recall@K = |Retrieved ∩ GroundTruth| / K
            recall = len(retrieved_ids.intersection(ground_truth_ids)) / k
            recalls.append(recall)

        mean_recall = float(np.mean(recalls))
        assert mean_recall == 1.0

    def test_sub_5ms_retrieval_latency(self, clean_indexer):
        """
        Retrieval latency over 1,000+ items must be well under the 5ms requirement.
        """
        np.random.seed(77)
        n_items = 1000
        dim = 64
        embs = np.random.randn(n_items, dim).astype(np.float32)
        ids = [f"it_{i}" for i in range(n_items)]
        clean_indexer.add_items(embs, ids)

        query_vec = np.random.randn(dim).astype(np.float32)

        # Warmup
        _ = clean_indexer.query(query_vec, top_k=10)

        # Timed queries
        latencies = []
        for _ in range(100):
            t0 = time.perf_counter()
            _ = clean_indexer.query(query_vec, top_k=10)
            latencies.append((time.perf_counter() - t0) * 1000.0)

        p95_latency = float(np.percentile(latencies, 95))
        assert p95_latency < 5.0, f"Expected p95 latency < 5.0ms, got {p95_latency:.3f}ms"

    def test_metadata_filtering(self, clean_indexer):
        """Metadata filter must eliminate non-matching candidates."""
        np.random.seed(42)
        embs = np.random.randn(10, 64).astype(np.float32)
        ids = [f"id_{i}" for i in range(10)]
        metas = [
            {"item_id": f"id_{i}", "category": "Gaming" if i % 2 == 0 else "Kitchen", "in_stock": i > 3}
            for i in range(10)
        ]
        clean_indexer.add_items(embs, ids, metadatas=metas)

        q = np.random.randn(64).astype(np.float32)

        # Filter: only Gaming items that are in stock
        results = clean_indexer.query(
            q,
            top_k=10,
            filter_fn=lambda m: m["category"] == "Gaming" and m["in_stock"],
        )

        for _, _, meta in results:
            assert meta["category"] == "Gaming"
            assert meta["in_stock"] is True

    def test_item_removal_and_rebuild(self, clean_indexer):
        """Item removal and index rebuild correctly updates total items and queries."""
        embs = np.random.randn(5, 64).astype(np.float32)
        ids = ["a", "b", "c", "d", "e"]
        clean_indexer.add_items(embs, ids)
        assert clean_indexer.total_items == 5

        removed = clean_indexer.remove_items(["a", "c"])
        assert removed == 2
        assert clean_indexer.total_items == 3
        assert not clean_indexer.contains("a")
        assert not clean_indexer.contains("c")
        assert clean_indexer.contains("b")

        results = clean_indexer.query(np.random.randn(64), top_k=10)
        retrieved_ids = [item_id for item_id, _, _ in results]
        assert "a" not in retrieved_ids
        assert "c" not in retrieved_ids

    def test_empty_index_queries(self, clean_indexer):
        """Querying an empty index returns empty candidate list without error."""
        q = np.random.randn(64).astype(np.float32)
        res = clean_indexer.query(q, top_k=10)
        assert res == []

    def test_dimension_mismatch_error(self, clean_indexer):
        """Adding vectors with mismatched dimensionality raises ValueError."""
        invalid_embs = np.random.randn(5, 32).astype(np.float32)
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            clean_indexer.add_items(invalid_embs, ["1", "2", "3", "4", "5"])

