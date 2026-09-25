"""Dense embedding adapters for semantic similarity and vector indexing."""

from __future__ import annotations

import hashlib
import math
from typing import Any
import httpx
import numpy as np

from domain.ports import EmbeddingProviderPort


class MockEmbeddingAdapter(EmbeddingProviderPort):
    """Deterministic mock embedding provider producing normalized vectors."""

    def __init__(self, dimensions: int = 1536) -> None:
        self.dimensions = dimensions

    def _generate_vector(self, text: str) -> list[float]:
        """Produce a deterministic normalized vector from text tokens and hash."""
        clean_text = text.strip().lower()
        arr = np.zeros(self.dimensions, dtype=np.float32)

        # Seed based on full text hash
        full_hash = int(hashlib.sha256(clean_text.encode("utf-8")).hexdigest()[:8], 16)
        np.random.seed(full_hash)
        base_noise = np.random.normal(0, 0.05, self.dimensions).astype(np.float32)
        arr += base_noise

        # Add distinct signal for individual words
        words = clean_text.split()
        for idx, word in enumerate(words):
            word_hash = int(hashlib.md5(word.encode("utf-8")).hexdigest()[:8], 16)
            pos = word_hash % self.dimensions
            weight = 1.0 / (idx + 1.0)
            arr[pos] += weight

        norm = float(np.linalg.norm(arr))
        if norm > 0:
            arr = arr / norm
        return [float(x) for x in arr]

    def embed_text(self, text: str) -> list[float]:
        """Compute an embedding vector synchronously."""
        return self._generate_vector(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Compute embedding vectors for a batch of strings synchronously."""
        return [self._generate_vector(t) for t in texts]

    async def embed_text_async(self, text: str) -> list[float]:
        """Compute an embedding vector asynchronously."""
        return self._generate_vector(text)

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Compute embedding vectors for a batch of strings asynchronously."""
        return [self._generate_vector(t) for t in texts]


class OpenAICompatibleEmbeddingAdapter(EmbeddingProviderPort):
    """Production HTTP embedding adapter supporting OpenAI, OpenRouter, and local vLLM."""

    def __init__(
        self,
        api_key: str = "mock-key",
        base_url: str = "https://api.openai.com/v1",
        model: str = "text-embedding-3-small",
        timeout_s: float = 15.0,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout_s = timeout_s

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def embed_text(self, text: str) -> list[float]:
        """Compute an embedding vector synchronously via HTTP."""
        res = self.embed_batch([text])
        if not res:
            raise RuntimeError("Embedding API returned empty vector array.")
        return res[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Compute embedding vectors for a batch of strings synchronously via HTTP."""
        url = f"{self.base_url}/embeddings"
        payload = {"input": texts, "model": self.model}

        with httpx.Client(timeout=self.timeout_s) as client:
            resp = client.post(url, headers=self._headers(), json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Embedding API error ({resp.status_code}): {resp.text}")
            data = resp.json()

        sorted_data = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in sorted_data]

    async def embed_text_async(self, text: str) -> list[float]:
        """Compute an embedding vector asynchronously via HTTP."""
        res = await self.embed_batch_async([text])
        if not res:
            raise RuntimeError("Embedding API returned empty vector array.")
        return res[0]

    async def embed_batch_async(self, texts: list[str]) -> list[list[float]]:
        """Compute embedding vectors for a batch of strings asynchronously via HTTP."""
        url = f"{self.base_url}/embeddings"
        payload = {"input": texts, "model": self.model}

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.post(url, headers=self._headers(), json=payload)
            if resp.status_code != 200:
                raise RuntimeError(f"Embedding API error ({resp.status_code}): {resp.text}")
            data = resp.json()

        sorted_data = sorted(data.get("data", []), key=lambda x: x.get("index", 0))
        return [item["embedding"] for item in sorted_data]
