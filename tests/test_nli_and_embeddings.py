"""Unit and integration tests for embeddings, NLI contradiction gating, and quarantine cleanup."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest
from fastapi.testclient import TestClient
import numpy as np

from api.app import app
from core.memory_promoter import MemoryPromoter
from core.quarantine_cleaner import QuarantineCleaner
from domain.models import (
    AdmissionStatus,
    CandidateFact,
    NLILabel,
    NLIResult,
    QuarantineRetentionPolicy,
    RejectionReason,
    SemanticFact,
)
from domain.ports import EmbeddingProviderPort, NLIProviderPort
from infrastructure.embedding_adapter import (
    MockEmbeddingAdapter,
    OpenAICompatibleEmbeddingAdapter,
)
from infrastructure.llm_adapter import MockLLMAdapter
from infrastructure.nli_adapter import LLMBasedNLIAdapter, MockNLIAdapter
from infrastructure.sqlite_fact_store import SQLiteFactStore
from infrastructure.vector_index import InMemoryVectorIndex


@pytest.fixture
def memory_fact_store() -> SQLiteFactStore:
    return SQLiteFactStore(":memory:")


# ---------------------------------------------------------------------------
# 1. Embedding Adapter Tests
# ---------------------------------------------------------------------------

def test_mock_embedding_adapter_dimensions_and_normalization():
    """Mock embedding adapter must produce unit-normalized vectors with exact dimensions."""
    adapter = MockEmbeddingAdapter(dimensions=512)
    vec = adapter.embed_text("hypertension treatment protocol")

    assert len(vec) == 512
    norm = float(np.linalg.norm(np.array(vec)))
    assert pytest.approx(norm, rel=1e-4) == 1.0


def test_mock_embedding_adapter_determinism():
    """Identical input strings must yield identical vector outputs."""
    adapter = MockEmbeddingAdapter(dimensions=256)
    vec1 = adapter.embed_text("penicillin allergy")
    vec2 = adapter.embed_text("penicillin allergy")
    assert vec1 == vec2

    vec3 = adapter.embed_text("unrelated text")
    assert vec1 != vec3


@pytest.mark.asyncio
async def test_mock_embedding_adapter_batch_async():
    """Batch asynchronous embedding generation should match synchronous results."""
    adapter = MockEmbeddingAdapter(dimensions=128)
    texts = ["heart failure", "diabetes mellitus", "chronic kidney disease"]

    sync_results = adapter.embed_batch(texts)
    async_results = await adapter.embed_batch_async(texts)

    assert len(async_results) == 3
    for s_vec, a_vec in zip(sync_results, async_results):
        assert s_vec == a_vec


def test_openai_compatible_embedding_adapter_sync(monkeypatch):
    """Test HTTP embedding adapter parses response correctly."""
    adapter = OpenAICompatibleEmbeddingAdapter(api_key="test-key", base_url="https://mock.api/v1")

    class MockResponse:
        status_code = 200

        def json(self):
            return {
                "data": [
                    {"index": 0, "embedding": [0.1, 0.2, 0.3]},
                    {"index": 1, "embedding": [0.4, 0.5, 0.6]},
                ]
            }

    class MockClient:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def post(self, url, headers, json):
            return MockResponse()

    monkeypatch.setattr("httpx.Client", MockClient)

    results = adapter.embed_batch(["text1", "text2"])
    assert len(results) == 2
    assert results[0] == [0.1, 0.2, 0.3]
    assert results[1] == [0.4, 0.5, 0.6]


# ---------------------------------------------------------------------------
# 2. NLI Adapter Tests
# ---------------------------------------------------------------------------

def test_mock_nli_adapter_clinical_contradiction():
    """NLI adapter must flag clinical semantic opposites as contradictions."""
    nli = MockNLIAdapter()

    # Hypertensive vs normotensive
    res = nli.classify_pair(
        premise="Patient is hypertensive with elevated systolic pressure.",
        hypothesis="Patient is strictly normotensive with stable pressure.",
    )
    assert res.label == NLILabel.CONTRADICTION
    assert res.contradiction_score > 0.90

    # Malignant vs benign
    res2 = nli.classify_pair(
        premise="Biopsy revealed malignant carcinoma.",
        hypothesis="Histology confirms benign tissue growth.",
    )
    assert res2.label == NLILabel.CONTRADICTION

    # Tobacco use contradiction
    res3 = nli.classify_pair(
        premise="Patient smokes 1 pack per day.",
        hypothesis="Patient denies tobacco use.",
    )
    assert res3.label == NLILabel.CONTRADICTION


def test_mock_nli_adapter_entailment_and_neutral():
    """Identical statements yield entailment; unrelated statements yield neutral."""
    nli = MockNLIAdapter()

    entail = nli.classify_pair(
        premise="Patient has penicillin allergy.",
        hypothesis="Patient has penicillin allergy.",
    )
    assert entail.label == NLILabel.ENTAILMENT

    neutral = nli.classify_pair(
        premise="Patient has penicillin allergy.",
        hypothesis="Patient enjoys running marathons.",
    )
    assert neutral.label == NLILabel.NEUTRAL


def test_llm_based_nli_adapter():
    """LLM cross-encoder adapter should construct prompt and parse JSON classification."""
    llm = MockLLMAdapter()
    llm.set_scripted_response(
        "premise",
        {
            "content": '{"label": "contradiction", "contradiction_score": 0.94, "entailment_score": 0.01, "neutral_score": 0.05, "detail": "Hypertension conflicts with normotension."}',
        },
    )

    adapter = LLMBasedNLIAdapter(llm)
    res = adapter.classify_pair(
        premise="Patient has stage 2 hypertension.",
        hypothesis="Patient is normotensive.",
    )
    assert res.label == NLILabel.CONTRADICTION
    assert res.contradiction_score == 0.94


# ---------------------------------------------------------------------------
# 3. Memory Promoter with NLI Contradiction Gating
# ---------------------------------------------------------------------------

def test_memory_promoter_nli_contradiction_rejection(memory_fact_store: SQLiteFactStore):
    """MemoryPromoter with NLI provider must catch semantic contradictions that pass keyword rules."""
    nli = MockNLIAdapter()
    promoter = MemoryPromoter(memory_fact_store, nli_provider=nli, min_confidence=0.80)

    # 1. Promote initial baseline fact: patient has high blood pressure
    fact_initial = SemanticFact(
        fact_id="fact_initial_bp",
        tenant_id="clinic_a",
        candidate_id="cand_init",
        source_episode_id="ep_init",
        session_id="sess_init",
        subject="cardiovascular_status",
        predicate="condition",
        object="hypertensive crisis",
        confidence=0.95,
        valid_from=datetime.now(timezone.utc),
        is_active=True,
        promoted_at=datetime.now(timezone.utc),
        provenance={"source": "physician_note"},
    )
    memory_fact_store.insert_semantic_fact(fact_initial)

    # 2. Candidate asserts normotensive state without negation keywords ("no", "none")
    cand_normotensive = CandidateFact(
        candidate_id="cand_normotensive",
        tenant_id="clinic_a",
        source_episode_id="ep_new",
        session_id="sess_new",
        subject="cardiovascular_status",
        predicate="condition",
        object="normotensive resting state",
        confidence=0.88,
    )
    promoter.submit_candidate(cand_normotensive)

    promoted, detail = promoter.evaluate_and_promote(cand_normotensive.candidate_id)
    assert promoted is False
    assert "NLI Contradiction" in detail

    # Verify status in quarantine
    stored_cand = memory_fact_store.get_candidate(cand_normotensive.candidate_id)
    assert stored_cand.status == AdmissionStatus.REJECTED
    assert stored_cand.rejection_reason == RejectionReason.CONTRADICTION_DETECTED


# ---------------------------------------------------------------------------
# 4. Memory Promoter Vector Index Synchronization on Admission
# ---------------------------------------------------------------------------

def test_memory_promoter_vector_index_sync(memory_fact_store: SQLiteFactStore):
    """Admitted facts must be converted to embeddings and upserted to vector index."""
    vector_index = InMemoryVectorIndex()
    embedding_adapter = MockEmbeddingAdapter(dimensions=64)
    promoter = MemoryPromoter(
        memory_fact_store,
        vector_index=vector_index,
        embedding_provider=embedding_adapter,
        min_confidence=0.80,
    )

    cand = CandidateFact(
        candidate_id="cand_asthma",
        tenant_id="hospital_berlin",
        source_episode_id="ep_asthma",
        session_id="sess_asthma",
        subject="respiratory_system",
        predicate="chronic_diagnosis",
        object="bronchial asthma",
        confidence=0.92,
    )
    promoter.submit_candidate(cand)
    promoted, _ = promoter.evaluate_and_promote(cand.candidate_id)
    assert promoted is True

    # Search in vector index using a related query
    query_vec = embedding_adapter.embed_text("respiratory system bronchial asthma")
    matches = vector_index.search(query_vec, top_k=5, tenant_id="hospital_berlin")

    assert len(matches) == 1
    assert matches[0]["metadata"]["subject"] == "respiratory_system"
    assert matches[0]["metadata"]["object"] == "bronchial asthma"
    assert matches[0]["tenant_id"] == "hospital_berlin"

    # Must be isolated from other tenants
    empty_matches = vector_index.search(query_vec, top_k=5, tenant_id="hospital_munich")
    assert len(empty_matches) == 0


# ---------------------------------------------------------------------------
# 5. Quarantine Retention and Garbage Collection
# ---------------------------------------------------------------------------

def test_quarantine_garbage_collection(memory_fact_store: SQLiteFactStore):
    """Quarantine cleaner must purge expired rejected records while retaining recent or pending ones."""
    cleaner = QuarantineCleaner(memory_fact_store)

    now = datetime.now(timezone.utc)
    old_time = now - timedelta(days=45)
    recent_time = now - timedelta(days=5)

    # 1. Old rejected candidate (45 days old) -> should be purged
    c_old_rejected = CandidateFact(
        candidate_id="c_old_rejected",
        tenant_id="tenant_gc",
        source_episode_id="ep_1",
        session_id="s_1",
        subject="lab",
        predicate="glucose",
        object="erroneous",
        confidence=0.1,
        status=AdmissionStatus.REJECTED,
        created_at=old_time,
    )
    memory_fact_store.insert_candidate(c_old_rejected)

    # 2. Recent rejected candidate (5 days old) -> should be kept
    c_recent_rejected = CandidateFact(
        candidate_id="c_recent_rejected",
        tenant_id="tenant_gc",
        source_episode_id="ep_2",
        session_id="s_2",
        subject="lab",
        predicate="glucose",
        object="erroneous",
        confidence=0.2,
        status=AdmissionStatus.REJECTED,
        created_at=recent_time,
    )
    memory_fact_store.insert_candidate(c_recent_rejected)

    # 3. Old pending candidate (45 days old) -> must NEVER be deleted automatically
    c_old_pending = CandidateFact(
        candidate_id="c_old_pending",
        tenant_id="tenant_gc",
        source_episode_id="ep_3",
        session_id="s_3",
        subject="lab",
        predicate="sodium",
        object="140",
        confidence=0.85,
        status=AdmissionStatus.PENDING,
        created_at=old_time,
    )
    memory_fact_store.insert_candidate(c_old_pending)

    # Run purge cycle with 30-day retention
    policy = QuarantineRetentionPolicy(max_age_days=30, statuses_to_purge=[AdmissionStatus.REJECTED])
    report = cleaner.run_purge_cycle(policy=policy, tenant_id="tenant_gc")

    assert report["purged_count"] == 1
    assert report["tenant_id"] == "tenant_gc"

    # Verify state in database
    assert memory_fact_store.get_candidate("c_old_rejected") is None
    assert memory_fact_store.get_candidate("c_recent_rejected") is not None
    assert memory_fact_store.get_candidate("c_old_pending") is not None


# ---------------------------------------------------------------------------
# 6. FastAPI Purge Endpoint
# ---------------------------------------------------------------------------

def test_fastapi_quarantine_purge_endpoint():
    """Test /api/quarantine/purge REST endpoint."""
    client = TestClient(app, headers={"X-API-Key": "dev-secret-key"})

    resp = client.post(
        "/api/quarantine/purge",
        json={
            "max_age_days": 14,
            "tenant_id": "api_test_tenant",
            "statuses": ["REJECTED"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "purged_count" in data
    assert data["tenant_id"] == "api_test_tenant"
    assert data["max_age_days"] == 14
