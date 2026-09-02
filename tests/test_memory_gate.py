"""Tests for Tri-State Memory Gate: Quarantine, Validation, and Semantic Promotion."""

import pytest
from core.memory_promoter import MemoryPromoter
from domain.models import AdmissionStatus, CandidateFact, RejectionReason
from infrastructure.sqlite_fact_store import SQLiteFactStore


def test_quarantine_isolation():
    """Verify candidate facts in quarantine are invisible to semantic retrieval."""
    fact_store = SQLiteFactStore(":memory:")
    promoter = MemoryPromoter(fact_store)

    candidate = CandidateFact(
        source_episode_id="ep_001",
        session_id="patient_100",
        subject="patient_100_allergies",
        predicate="status",
        object="no known allergies",
        confidence=0.95,
    )

    # Submit to quarantine
    cand_id = promoter.submit_candidate(candidate)

    # Verify quarantine has it with PENDING status
    quarantine_list = fact_store.get_quarantine_records()
    assert len(quarantine_list) == 1
    assert quarantine_list[0].status == AdmissionStatus.PENDING

    # Verify active semantic retrieval returns NOTHING
    active_facts = fact_store.get_active_facts_for_subject("patient_100_allergies")
    assert len(active_facts) == 0


def test_promotion_valid_fact():
    """Verify high-confidence valid candidate promotes to semantic facts with provenance."""
    fact_store = SQLiteFactStore(":memory:")
    promoter = MemoryPromoter(fact_store)

    candidate = CandidateFact(
        source_episode_id="ep_002",
        session_id="patient_200",
        subject="patient_200_diagnosis",
        predicate="condition",
        object="essential hypertension I10",
        confidence=0.92,
    )

    cand_id = promoter.submit_candidate(candidate)
    promoted, msg = promoter.evaluate_and_promote(cand_id)

    assert promoted is True
    assert "Promoted to Fact ID:" in msg

    # Verify semantic storage contains promoted fact
    active_facts = fact_store.get_active_facts_for_subject("patient_200_diagnosis")
    assert len(active_facts) == 1
    fact = active_facts[0]
    assert fact.object == "essential hypertension I10"
    assert fact.is_active is True
    assert fact.provenance["quarantine_created_at"] != ""


def test_rejection_low_confidence():
    """Verify candidate below minimum confidence is rejected with LOW_CONFIDENCE code."""
    fact_store = SQLiteFactStore(":memory:")
    promoter = MemoryPromoter(fact_store, min_confidence=0.85)

    candidate = CandidateFact(
        source_episode_id="ep_003",
        session_id="patient_300",
        subject="patient_300_symptom",
        predicate="observed",
        object="possible mild headache",
        confidence=0.60,
    )

    cand_id = promoter.submit_candidate(candidate)
    promoted, msg = promoter.evaluate_and_promote(cand_id)

    assert promoted is False
    assert "Low confidence" in msg

    # Verify status in quarantine
    records = fact_store.get_quarantine_records()
    assert records[0].status == AdmissionStatus.REJECTED
    assert records[0].rejection_reason == RejectionReason.LOW_CONFIDENCE


def test_contradiction_detection():
    """Verify contradictory assertion against active fact is caught and blocked."""
    fact_store = SQLiteFactStore(":memory:")
    promoter = MemoryPromoter(fact_store)

    # Session 1: Patient records smoker status as 'denies smoking'
    c1 = CandidateFact(
        source_episode_id="ep_sess_1",
        session_id="patient_400",
        subject="patient_400_smoking",
        predicate="smoker",
        object="denies smoking (no)",
        confidence=0.95,
    )
    promoter.submit_candidate(c1)
    promoter.evaluate_and_promote(c1.candidate_id)

    # Session 2: Patient asserts active smoking (confidence is 0.88, above 0.85 threshold, but polarity conflicts)
    c2 = CandidateFact(
        source_episode_id="ep_sess_2",
        session_id="patient_400",
        subject="patient_400_smoking",
        predicate="smoker",
        object="active smoker 1 pack daily (yes)",
        confidence=0.88,
    )
    promoter.submit_candidate(c2)
    promoted, msg = promoter.evaluate_and_promote(c2.candidate_id)

    assert promoted is False
    assert "Contradiction detected" in msg

    records = fact_store.get_quarantine_records()
    rejected_cand = next(r for r in records if r.candidate_id == c2.candidate_id)
    assert rejected_cand.status == AdmissionStatus.REJECTED
    assert rejected_cand.rejection_reason == RejectionReason.CONTRADICTION_DETECTED
