"""Comprehensive tests for multi-tenant isolation, persistence, and vector indexing."""

from __future__ import annotations

from datetime import datetime, timezone
import uuid
import pytest

from core.context_ram import ContextRAM
from core.execution_loop import ExecutionEngine
from core.memory_promoter import MemoryPromoter
from domain.models import (
    AdmissionStatus,
    CandidateFact,
    ConfigPack,
    SemanticFact,
    Turn,
)
from domain.ports import EpisodeStorePort, FactStorePort
from infrastructure.llm_adapter import MockLLMAdapter
from infrastructure.postgres_store import PostgresEpisodeStore, PostgresFactStore
from infrastructure.sqlite_episode_store import SQLiteEpisodeStore
from infrastructure.sqlite_fact_store import SQLiteFactStore
from infrastructure.vector_index import InMemoryVectorIndex
from tools.registry import ToolRegistry


@pytest.fixture
def sqlite_fact_store() -> SQLiteFactStore:
    return SQLiteFactStore(":memory:")


@pytest.fixture
def sqlite_episode_store() -> SQLiteEpisodeStore:
    return SQLiteEpisodeStore(":memory:")


def test_cross_tenant_fact_isolation(sqlite_fact_store: SQLiteFactStore):
    """Ensure semantic facts are strictly isolated between tenants."""
    fact_a = SemanticFact(
        fact_id="fact_a_1",
        tenant_id="tenant_alpha",
        candidate_id="cand_a_1",
        source_episode_id="ep_a",
        session_id="sess_a",
        subject="patient_allergy",
        predicate="has_allergy",
        object="penicillin",
        confidence=0.98,
        valid_from=datetime.now(timezone.utc),
        is_active=True,
        promoted_at=datetime.now(timezone.utc),
        provenance={"source": "test"},
    )
    fact_b = SemanticFact(
        fact_id="fact_b_1",
        tenant_id="tenant_beta",
        candidate_id="cand_b_1",
        source_episode_id="ep_b",
        session_id="sess_b",
        subject="patient_allergy",
        predicate="has_allergy",
        object="peanuts",
        confidence=0.95,
        valid_from=datetime.now(timezone.utc),
        is_active=True,
        promoted_at=datetime.now(timezone.utc),
        provenance={"source": "test"},
    )

    sqlite_fact_store.insert_semantic_fact(fact_a)
    sqlite_fact_store.insert_semantic_fact(fact_b)

    # Query tenant_alpha
    facts_alpha = sqlite_fact_store.get_active_facts_for_subject("patient_allergy", tenant_id="tenant_alpha")
    assert len(facts_alpha) == 1
    assert facts_alpha[0].object == "penicillin"
    assert facts_alpha[0].tenant_id == "tenant_alpha"

    # Query tenant_beta
    facts_beta = sqlite_fact_store.get_active_facts_for_subject("patient_allergy", tenant_id="tenant_beta")
    assert len(facts_beta) == 1
    assert facts_beta[0].object == "peanuts"
    assert facts_beta[0].tenant_id == "tenant_beta"

    # Query unknown tenant
    facts_gamma = sqlite_fact_store.get_active_facts_for_subject("patient_allergy", tenant_id="tenant_gamma")
    assert len(facts_gamma) == 0


def test_cross_tenant_contradiction_isolation(sqlite_fact_store: SQLiteFactStore):
    """Verify contradictory facts can coexist peacefully across separate tenants."""
    promoter = MemoryPromoter(sqlite_fact_store, min_confidence=0.80)

    # Tenant Alpha asserts allergy
    cand_a = CandidateFact(
        candidate_id="cand_alpha_allergy",
        tenant_id="tenant_alpha",
        source_episode_id="ep_1",
        session_id="sess_1",
        subject="patient_allergy",
        predicate="has_allergy",
        object="penicillin",
        confidence=0.95,
    )
    promoter.submit_candidate(cand_a)
    promoted_a, _ = promoter.evaluate_and_promote(cand_a.candidate_id)
    assert promoted_a is True

    # Tenant Beta asserts NO allergies for the same subject key
    cand_b = CandidateFact(
        candidate_id="cand_beta_no_allergy",
        tenant_id="tenant_beta",
        source_episode_id="ep_2",
        session_id="sess_2",
        subject="patient_allergy",
        predicate="has_allergy",
        object="none",
        confidence=0.95,
    )
    promoter.submit_candidate(cand_b)
    promoted_b, _ = promoter.evaluate_and_promote(cand_b.candidate_id)
    # Must succeed because Beta is isolated from Alpha's allergy record
    assert promoted_b is True

    # However, within Tenant Alpha, another candidate asserting "none" will be rejected
    cand_a_contradict = CandidateFact(
        candidate_id="cand_alpha_contradict",
        tenant_id="tenant_alpha",
        source_episode_id="ep_3",
        session_id="sess_3",
        subject="patient_allergy",
        predicate="has_allergy",
        object="none",
        confidence=0.85,
    )
    promoter.submit_candidate(cand_a_contradict)
    promoted_c, detail_c = promoter.evaluate_and_promote(cand_a_contradict.candidate_id)
    assert promoted_c is False
    assert "Contradiction detected" in detail_c


def test_cross_tenant_episodic_isolation(sqlite_episode_store: SQLiteEpisodeStore):
    """Turns from the same session ID in different tenants must never collide."""
    turn_alpha = Turn(
        turn_id="turn_alpha_1",
        tenant_id="tenant_alpha",
        session_id="shared_session_key",
        turn_index=0,
        user_input="Alpha user message",
        model_output="Alpha assistant response",
        created_at=datetime.now(timezone.utc),
    )
    turn_beta = Turn(
        turn_id="turn_beta_1",
        tenant_id="tenant_beta",
        session_id="shared_session_key",
        turn_index=0,
        user_input="Beta user message",
        model_output="Beta assistant response",
        created_at=datetime.now(timezone.utc),
    )

    sqlite_episode_store.append_turn(turn_alpha)
    sqlite_episode_store.append_turn(turn_beta)

    turns_alpha = sqlite_episode_store.get_recent_turns("shared_session_key", tenant_id="tenant_alpha")
    assert len(turns_alpha) == 1
    assert turns_alpha[0].user_input == "Alpha user message"

    turns_beta = sqlite_episode_store.get_recent_turns("shared_session_key", tenant_id="tenant_beta")
    assert len(turns_beta) == 1
    assert turns_beta[0].user_input == "Beta user message"


def test_context_ram_tenant_scoping(sqlite_fact_store: SQLiteFactStore):
    """ContextRAM must assemble facts exclusively belonging to the requesting tenant."""
    context_ram = ContextRAM(sqlite_fact_store)

    fact_alpha = SemanticFact(
        fact_id="f_alpha",
        tenant_id="clinic_north",
        candidate_id="c_alpha",
        source_episode_id="ep_1",
        session_id="s_1",
        subject="dosage",
        predicate="daily_limit",
        object="50mg",
        confidence=0.99,
        valid_from=datetime.now(timezone.utc),
        is_active=True,
        promoted_at=datetime.now(timezone.utc),
        provenance={},
    )
    fact_beta = SemanticFact(
        fact_id="f_beta",
        tenant_id="clinic_south",
        candidate_id="c_beta",
        source_episode_id="ep_2",
        session_id="s_2",
        subject="dosage",
        predicate="daily_limit",
        object="100mg",
        confidence=0.99,
        valid_from=datetime.now(timezone.utc),
        is_active=True,
        promoted_at=datetime.now(timezone.utc),
        provenance={},
    )
    sqlite_fact_store.insert_semantic_fact(fact_alpha)
    sqlite_fact_store.insert_semantic_fact(fact_beta)

    config = ConfigPack(version="1.0.0", system_prompt="System Prompt")

    # Compose context for clinic_north
    messages_north = context_ram.compose_context(
        user_input="Check dosage limit",
        recent_turns=[],
        active_config=config,
        session_id="s_1",
        tenant_id="clinic_north",
    )
    sys_content_north = messages_north[0]["content"]
    assert "50mg" in sys_content_north
    assert "100mg" not in sys_content_north

    # Compose context for clinic_south
    messages_south = context_ram.compose_context(
        user_input="Check dosage limit",
        recent_turns=[],
        active_config=config,
        session_id="s_2",
        tenant_id="clinic_south",
    )
    sys_content_south = messages_south[0]["content"]
    assert "100mg" in sys_content_south
    assert "50mg" not in sys_content_south


def test_vector_index_tenant_isolation():
    """InMemoryVectorIndex must filter candidates strictly by tenant."""
    index = InMemoryVectorIndex()

    # Alpha item: close to [1.0, 0.0]
    index.upsert(
        item_id="item_alpha",
        vector=[1.0, 0.1],
        metadata={"title": "Alpha Protocol"},
        tenant_id="tenant_alpha",
    )
    # Beta item: even closer to [1.0, 0.0]
    index.upsert(
        item_id="item_beta",
        vector=[1.0, 0.01],
        metadata={"title": "Beta Protocol"},
        tenant_id="tenant_beta",
    )

    query_vec = [1.0, 0.0]

    # Search in tenant_alpha: item_beta must NOT appear even though it has higher cosine similarity
    res_alpha = index.search(query_vec, top_k=5, tenant_id="tenant_alpha")
    assert len(res_alpha) == 1
    assert res_alpha[0]["item_id"] == "item_alpha"
    assert res_alpha[0]["tenant_id"] == "tenant_alpha"

    # Search in tenant_beta
    res_beta = index.search(query_vec, top_k=5, tenant_id="tenant_beta")
    assert len(res_beta) == 1
    assert res_beta[0]["item_id"] == "item_beta"
    assert res_beta[0]["tenant_id"] == "tenant_beta"


def test_execution_engine_tenant_stamping(sqlite_episode_store: SQLiteEpisodeStore, sqlite_fact_store: SQLiteFactStore):
    """ExecutionEngine must propagate tenant_id down to Turn records in synchronous and async runs."""
    llm = MockLLMAdapter()
    llm.set_scripted_response("hello", {"content": "Hello from agent", "tokens_in": 10, "tokens_out": 10, "cost_usd": 0.001})

    context_ram = ContextRAM(sqlite_fact_store)
    engine = ExecutionEngine(
        llm=llm,
        episode_store=sqlite_episode_store,
        tool_registry=ToolRegistry(),
        context_ram=context_ram,
    )

    config = ConfigPack(version="1.0.0", system_prompt="Test")

    # Run turn with explicit tenant
    turn = engine.run_turn(
        user_input="hello",
        session_id="sess_engine_tenant",
        active_config=config,
        tenant_id="hospital_charite",
    )
    assert turn.tenant_id == "hospital_charite"

    # Verify persistence in episode store
    persisted_turns = sqlite_episode_store.get_recent_turns("sess_engine_tenant", tenant_id="hospital_charite")
    assert len(persisted_turns) == 1
    assert persisted_turns[0].tenant_id == "hospital_charite"

    # Should not be accessible from default_tenant
    empty_turns = sqlite_episode_store.get_recent_turns("sess_engine_tenant", tenant_id="default_tenant")
    assert len(empty_turns) == 0


@pytest.mark.asyncio
async def test_execution_engine_async_stream_tenant_stamping(sqlite_episode_store: SQLiteEpisodeStore, sqlite_fact_store: SQLiteFactStore):
    """Async streaming turn must also record tenant_id properly."""
    llm = MockLLMAdapter()
    llm.set_scripted_response("streaming query", {"content": "Async response", "tokens_in": 12, "tokens_out": 8, "cost_usd": 0.0005})

    context_ram = ContextRAM(sqlite_fact_store)
    engine = ExecutionEngine(
        llm=llm,
        episode_store=sqlite_episode_store,
        tool_registry=ToolRegistry(),
        context_ram=context_ram,
    )
    config = ConfigPack(version="1.0.0", system_prompt="Test")

    turn = await engine.run_turn_async(
        user_input="streaming query",
        session_id="sess_stream_tenant",
        active_config=config,
        tenant_id="tenant_telehealth",
    )
    assert turn.tenant_id == "tenant_telehealth"

    stored = sqlite_episode_store.get_recent_turns("sess_stream_tenant", tenant_id="tenant_telehealth")
    assert len(stored) == 1
    assert stored[0].tenant_id == "tenant_telehealth"


def test_postgres_store_protocol_compliance():
    """Verify PostgreSQL store classes satisfy domain interface protocols."""
    pg_ep = PostgresEpisodeStore(connection_url="postgresql://user:pass@localhost/db")
    pg_fact = PostgresFactStore(connection_url="postgresql://user:pass@localhost/db")

    assert isinstance(pg_ep, EpisodeStorePort)
    assert isinstance(pg_fact, FactStorePort)


def test_fastapi_tenant_endpoints():
    """Verify REST endpoints correctly forward and isolate by tenant_id."""
    from fastapi.testclient import TestClient
    from api.app import app

    client = TestClient(app, headers={"X-API-Key": "dev-secret-key"})

    # 1. Chat turn with custom tenant
    resp = client.post(
        "/api/chat",
        json={
            "user_input": "hello",
            "session_id": "api_test_session",
            "tenant_id": "tenant_api_1",
            "max_steps": 2,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["turn"]["tenant_id"] == "tenant_api_1"

    # 2. Submit quarantine fact with custom tenant
    unique_subj = f"api_subject_{uuid.uuid4().hex[:6]}"
    fact_resp = client.post(
        "/api/quarantine/submit",
        json={
            "session_id": "api_test_session",
            "source_episode_id": "ep_test",
            "subject": unique_subj,
            "predicate": "api_pred",
            "object": "api_obj",
            "confidence": 0.95,
            "tenant_id": "tenant_api_1",
        },
    )
    assert fact_resp.status_code == 200
    assert fact_resp.json()["promoted"] is True

    # 3. Facts list for tenant_api_1 should include it
    facts_resp1 = client.get("/api/facts?tenant_id=tenant_api_1")
    assert facts_resp1.status_code == 200
    facts1 = facts_resp1.json()
    assert any(f["subject"] == unique_subj and f["tenant_id"] == "tenant_api_1" for f in facts1)

    # 4. Facts list for tenant_api_2 must NOT include it
    facts_resp2 = client.get("/api/facts?tenant_id=tenant_api_2")
    assert facts_resp2.status_code == 200
    facts2 = facts_resp2.json()
    assert not any(f["subject"] == unique_subj for f in facts2)

