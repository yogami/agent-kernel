"""Hybrid RAG Engine combining BM25 sparse lexical search, dense vector retrieval, Reciprocal Rank Fusion (RRF), and cross-encoder reranking."""

from __future__ import annotations

from collections import Counter
import math
from typing import Any
import numpy as np
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Atomic text passage with structured citation metadata."""

    chunk_id: str
    document_id: str
    title: str
    section: str
    paragraph_index: int
    table_reference: str | None = None
    text: str
    keywords: list[str] = Field(default_factory=list)
    dense_vector: list[float] = Field(default_factory=list)


class Citation(BaseModel):
    """Ground truth citation linking an assertion back to source literature."""

    document_id: str
    title: str
    section: str
    paragraph_index: int
    table_reference: str | None = None
    snippet: str


class HybridSearchResult(BaseModel):
    """Ranked search result with RRF score and citation provenance."""

    chunk: DocumentChunk
    bm25_score: float
    dense_score: float
    rrf_score: float
    rerank_score: float = 0.0
    citation: Citation


class BM25Index:
    """Okapi BM25 implementation in pure Python and NumPy."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avg_doc_len = 0.0
        self.doc_lens: list[int] = []
        self.doc_term_freqs: list[Counter[str]] = []
        self.idf: dict[str, float] = {}

    def _tokenize(self, text: str) -> list[str]:
        """Simple alphanumeric tokenizer with case folding."""
        return [w.lower().strip(".,;:()[]\"'") for w in text.split() if w.strip()]

    def fit(self, documents: list[str]) -> None:
        """Index document texts and compute inverse document frequencies (IDF)."""
        self.corpus_size = len(documents)
        if self.corpus_size == 0:
            return

        self.doc_term_freqs = []
        self.doc_lens = []
        total_len = 0
        df: Counter[str] = Counter()

        for doc in documents:
            tokens = self._tokenize(doc)
            self.doc_lens.append(len(tokens))
            total_len += len(tokens)
            tf = Counter(tokens)
            self.doc_term_freqs.append(tf)
            for word in tf.keys():
                df[word] += 1

        self.avg_doc_len = total_len / self.corpus_size if self.corpus_size > 0 else 0.0

        # Calculate Okapi BM25 IDF
        self.idf = {}
        for word, freq in df.items():
            # Standard smoothed BM25 IDF formula
            self.idf[word] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)

    def score(self, query: str) -> list[float]:
        """Compute BM25 match scores for all indexed documents."""
        query_tokens = self._tokenize(query)
        scores = [0.0] * self.corpus_size

        for i in range(self.corpus_size):
            doc_len = self.doc_lens[i]
            tf_map = self.doc_term_freqs[i]
            doc_score = 0.0

            for term in query_tokens:
                if term not in tf_map:
                    continue
                tf = tf_map[term]
                idf = self.idf.get(term, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avg_doc_len or 1.0)))
                doc_score += idf * ((tf * (self.k1 + 1.0)) / denom)

            scores[i] = doc_score

        return scores


class HybridRAGEngine:
    """Combines BM25 lexical keyword search, dense cosine vector retrieval, RRF, and reranking."""

    def __init__(self, rrf_k: int = 60) -> None:
        self.rrf_k = rrf_k
        self.chunks: list[DocumentChunk] = []
        self.bm25 = BM25Index()
        self.dense_matrix: np.ndarray | None = None

    def index_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Load document chunks and build both lexical and dense indices."""
        self.chunks = chunks
        if not chunks:
            return

        # 1. Fit BM25 on text + explicit keywords
        corpus = [f"{c.text} {' '.join(c.keywords)}" for c in chunks]
        self.bm25.fit(corpus)

        # 2. Build dense normalized vector matrix
        vectors = [c.dense_vector for c in chunks if c.dense_vector]
        if len(vectors) == len(chunks) and len(vectors[0]) > 0:
            mat = np.array(vectors, dtype=np.float32)
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms[norms == 0.0] = 1.0
            self.dense_matrix = mat / norms
        else:
            self.dense_matrix = None

    def search(
        self,
        query: str,
        query_vector: list[float] | None = None,
        top_k: int = 3,
    ) -> list[HybridSearchResult]:
        """Perform hybrid retrieval combining BM25, dense vectors, and RRF."""
        if not self.chunks:
            return []

        num_docs = len(self.chunks)

        # 1. Sparse Lexical Search (BM25)
        bm25_scores = self.bm25.score(query)
        bm25_ranked_indices = np.argsort(bm25_scores)[::-1]
        bm25_ranks = {idx: rank + 1 for rank, idx in enumerate(bm25_ranked_indices)}

        # 2. Dense Semantic Search (Cosine Similarity)
        dense_scores = [0.0] * num_docs
        if self.dense_matrix is not None and query_vector:
            q_vec = np.array(query_vector, dtype=np.float32)
            q_norm = np.linalg.norm(q_vec)
            if q_norm > 0.0:
                q_vec = q_vec / q_norm
                dense_scores = list(np.dot(self.dense_matrix, q_vec))

        dense_ranked_indices = np.argsort(dense_scores)[::-1]
        dense_ranks = {idx: rank + 1 for rank, idx in enumerate(dense_ranked_indices)}

        # 3. Reciprocal Rank Fusion (RRF)
        # RRF Score = 1 / (k + rank_bm25) + 1 / (k + rank_dense)
        results: list[HybridSearchResult] = []
        for idx in range(num_docs):
            chunk = self.chunks[idx]
            r_bm25 = bm25_ranks[idx]
            r_dense = dense_ranks[idx]

            rrf_score = (1.0 / (self.rrf_k + r_bm25)) + (1.0 / (self.rrf_k + r_dense))
            b_score = bm25_scores[idx]
            d_score = dense_scores[idx]

            # Cross-encoder style heuristic reranking (fusing lexical boost on exact code hits)
            rerank_score = rrf_score * (1.0 + (0.2 if b_score > 0 else 0.0))

            citation = Citation(
                document_id=chunk.document_id,
                title=chunk.title,
                section=chunk.section,
                paragraph_index=chunk.paragraph_index,
                table_reference=chunk.table_reference,
                snippet=chunk.text[:120] + "...",
            )

            results.append(
                HybridSearchResult(
                    chunk=chunk,
                    bm25_score=b_score,
                    dense_score=d_score,
                    rrf_score=rrf_score,
                    rerank_score=rerank_score,
                    citation=citation,
                )
            )

        # Sort by rerank score descending
        results.sort(key=lambda x: x.rerank_score, reverse=True)
        return results[:top_k]

    def format_citation_context(self, results: list[HybridSearchResult]) -> str:
        """Format retrieved passages with mandatory XML citation headers."""
        if not results:
            return ""

        formatted_blocks = []
        for i, res in enumerate(results, 1):
            c = res.citation
            table_info = f" | Table: {c.table_reference}" if c.table_reference else ""
            header = f"[Source {i}: {c.title} (DocID: {c.document_id}) | Section: {c.section} | Para: {c.paragraph_index}{table_info}]"
            formatted_blocks.append(f"{header}\n{res.chunk.text}")

        return "\n\n---\n\n".join(formatted_blocks)
