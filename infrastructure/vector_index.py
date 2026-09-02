"""In-memory vector index with numpy cosine similarity and metadata filtering."""

from __future__ import annotations

from typing import Any
import numpy as np

from domain.ports import VectorIndexPort


class InMemoryVectorIndex(VectorIndexPort):
    """Vector accelerator using numpy cosine similarity with metadata filtering."""

    def __init__(self) -> None:
        self.vectors: dict[str, np.ndarray] = {}
        self.metadata_store: dict[str, dict[str, Any]] = {}

    def upsert(self, item_id: str, vector: list[float], metadata: dict[str, Any]) -> None:
        """Index a vector representation alongside metadata."""
        arr = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        self.vectors[item_id] = arr
        self.metadata_store[item_id] = metadata

    def search(
        self,
        query_vector: list[float],
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Find nearest vectors satisfying metadata constraints."""
        if not self.vectors:
            return []

        q_arr = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q_arr)
        if q_norm > 0:
            q_arr = q_arr / q_norm

        results: list[dict[str, Any]] = []

        for item_id, v_arr in self.vectors.items():
            meta = self.metadata_store.get(item_id, {})

            # Check metadata filters if specified
            if filter_metadata:
                match = True
                for k, v in filter_metadata.items():
                    if meta.get(k) != v:
                        match = False
                        break
                if not match:
                    continue

            # Cosine similarity on normalized vectors is dot product
            score = float(np.dot(q_arr, v_arr))
            results.append({
                "item_id": item_id,
                "score": score,
                "metadata": meta,
            })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def remove(self, item_id: str) -> None:
        """Remove a vector from the index."""
        self.vectors.pop(item_id, None)
        self.metadata_store.pop(item_id, None)
