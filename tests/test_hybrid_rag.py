"""Tests for Hybrid RAG Engine (BM25 + Dense + RRF + Citations)."""

import pytest
from infrastructure.hybrid_rag import DocumentChunk, HybridRAGEngine


@pytest.fixture
def sample_guideline_corpus() -> list[DocumentChunk]:
    """Sample clinical treatment guideline corpus."""
    return [
        DocumentChunk(
            chunk_id="guideline_htn_1",
            document_id="ESC_ESH_2024_HTN",
            title="ESC/ESH Guidelines for Arterial Hypertension",
            section="Section 4.2 Pharmacotherapy",
            paragraph_index=3,
            table_reference="Table 8",
            text="Initial monotherapy for Grade 1 hypertension: ACE inhibitors (e.g. ramipril 5mg daily) or ARBs are recommended for primary organ protection. ICD-10 code I10.",
            keywords=["I10", "hypertension", "ramipril", "ACE", "ESC"],
            dense_vector=[0.9, 0.1, 0.2, 0.0],
        ),
        DocumentChunk(
            chunk_id="guideline_t2d_1",
            document_id="ADA_EASD_2024_DM",
            title="ADA Standards of Medical Care in Diabetes",
            section="Section 9 Pharmacologic Approaches",
            paragraph_index=12,
            table_reference="Table 9.1",
            text="First-line therapy for type 2 diabetes (E11.9): Metformin 500mg titration combined with comprehensive lifestyle modification.",
            keywords=["E11.9", "diabetes", "metformin", "T2D"],
            dense_vector=[0.1, 0.9, 0.0, 0.2],
        ),
        DocumentChunk(
            chunk_id="guideline_asthma_1",
            document_id="GINA_2024_ASTHMA",
            title="Global Initiative for Asthma Guidelines",
            section="Section 3 Stepwise Management",
            paragraph_index=5,
            table_reference=None,
            text="Track 1: Low-dose inhaled corticosteroid (ICS)-formoterol as the preferred reliever for acute bronchospasm.",
            keywords=["J45.9", "asthma", "formoterol", "bronchospasm"],
            dense_vector=[0.1, 0.1, 0.8, 0.1],
        ),
    ]


def test_bm25_exact_code_matching(sample_guideline_corpus):
    """Verify BM25 strongly ranks exact alphanumeric ICD-10 codes."""
    engine = HybridRAGEngine(rrf_k=60)
    engine.index_chunks(sample_guideline_corpus)

    # Search exact alphanumeric code "I10"
    results = engine.search(query="I10 ramipril", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.document_id == "ESC_ESH_2024_HTN"
    assert results[0].bm25_score > 0.0
    assert results[0].citation.table_reference == "Table 8"


def test_dense_semantic_matching(sample_guideline_corpus):
    """Verify dense vector matches semantic meaning when keywords differ."""
    engine = HybridRAGEngine(rrf_k=60)
    engine.index_chunks(sample_guideline_corpus)

    # Search with a dense query vector close to diabetes [0.1, 0.9, 0.0, 0.2]
    query_vector = [0.12, 0.88, 0.05, 0.18]
    results = engine.search(query="blood sugar elevation pills", query_vector=query_vector, top_k=1)

    assert len(results) == 1
    assert results[0].chunk.document_id == "ADA_EASD_2024_DM"
    assert results[0].dense_score > 0.95


def test_rrf_and_citation_formatting(sample_guideline_corpus):
    """Verify Reciprocal Rank Fusion computes valid combined scores and formats citations."""
    engine = HybridRAGEngine(rrf_k=60)
    engine.index_chunks(sample_guideline_corpus)

    results = engine.search(query="ramipril for blood pressure", query_vector=[0.85, 0.15, 0.2, 0.0], top_k=2)

    assert len(results) == 2
    # Verify RRF score is within theoretical bounds: 2 * (1 / (60 + 1)) = ~0.0327
    top_hit = results[0]
    assert 0.0 < top_hit.rrf_score <= (2.0 / 61.0)

    # Verify citation formatting
    context_str = engine.format_citation_context(results)
    assert "[Source 1: ESC/ESH Guidelines for Arterial Hypertension (DocID: ESC_ESH_2024_HTN)" in context_str
    assert "Section: Section 4.2 Pharmacotherapy" in context_str
    assert "Table: Table 8" in context_str
