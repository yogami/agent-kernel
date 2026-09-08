"""Tri-State Memory Admission Gate: Quarantine, Validation, and Semantic Promotion."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
import uuid

from domain.models import (
    AdmissionStatus,
    CandidateFact,
    RejectionReason,
    SemanticFact,
)
from domain.ports import FactStorePort, VectorIndexPort


class MemoryPromoter:
    """Admission controller gating facts between quarantine and durable semantic storage."""

    def __init__(
        self,
        fact_store: FactStorePort,
        vector_index: VectorIndexPort | None = None,
        min_confidence: float = 0.85,
        causal_reasoner: Any | None = None,
    ) -> None:
        self.fact_store = fact_store
        self.vector_index = vector_index
        self.min_confidence = min_confidence
        self.causal_reasoner = causal_reasoner


    def submit_candidate(self, candidate: CandidateFact) -> str:
        """Step 1 & 2: Write candidate fact strictly to quarantine table."""
        self.fact_store.insert_candidate(candidate)
        return candidate.candidate_id

    def evaluate_and_promote(self, candidate_id: str) -> tuple[bool, str]:
        """Step 3: Evaluate admission criteria for a quarantined candidate."""
        # Find candidate in quarantine records
        quarantine_records = self.fact_store.get_quarantine_records(limit=200)
        candidate = next((c for c in quarantine_records if c.candidate_id == candidate_id), None)

        if not candidate:
            return False, f"Candidate ID '{candidate_id}' not found in quarantine."

        # Criteria A: Schema validation
        if not candidate.subject or not candidate.predicate or not candidate.object:
            self.fact_store.update_candidate_status(
                candidate_id,
                status=AdmissionStatus.REJECTED,
                reason=RejectionReason.SCHEMA_INVALID,
                detail="Empty subject, predicate, or object field in fact extraction.",
            )
            return False, "Rejected: Schema invalid."

        # Criteria B: Confidence threshold
        if candidate.confidence < self.min_confidence:
            self.fact_store.update_candidate_status(
                candidate_id,
                status=AdmissionStatus.REJECTED,
                reason=RejectionReason.LOW_CONFIDENCE,
                detail=f"Confidence {candidate.confidence:.2f} below required threshold {self.min_confidence:.2f}.",
            )
            return False, "Rejected: Low confidence score."

        # Criteria C: Existing active facts check for near-duplicate or direct contradiction
        existing_facts = self.fact_store.get_active_facts_for_subject(candidate.subject)

        for ef in existing_facts:
            # Check near duplicate
            if (
                ef.predicate.lower() == candidate.predicate.lower()
                and ef.object.lower() == candidate.object.lower()
            ):
                self.fact_store.update_candidate_status(
                    candidate_id,
                    status=AdmissionStatus.REJECTED,
                    reason=RejectionReason.NEAR_DUPLICATE,
                    detail=f"Fact already exists in active semantic storage (Fact ID: {ef.fact_id}).",
                )
                return False, "Rejected: Duplicate fact."

            # Check direct semantic contradiction
            is_conflict, conflict_detail = self._check_contradiction(candidate, ef)
            if is_conflict:
                # If candidate is a high-confidence direct temporal update (e.g. status change from 'none' to specific allergen)
                if self._is_legitimate_temporal_supersede(candidate, ef):
                    # Retire the previous fact cleanly
                    self.fact_store.retire_fact(ef.fact_id, reason=f"Superseded by Candidate {candidate.candidate_id}")
                else:
                    self.fact_store.update_candidate_status(
                        candidate_id,
                        status=AdmissionStatus.REJECTED,
                        reason=RejectionReason.CONTRADICTION_DETECTED,
                        detail=conflict_detail,
                    )
                    return False, f"Rejected: Contradiction detected. {conflict_detail}"

        # Criteria D: Causal Consistency Check (if causal reasoner is configured)
        if self.causal_reasoner is not None:
            is_causally_valid, causal_err = self._check_causal_consistency(candidate)
            if not is_causally_valid:
                self.fact_store.update_candidate_status(
                    candidate_id,
                    status=AdmissionStatus.REJECTED,
                    reason=RejectionReason.CAUSAL_INCONSISTENCY,
                    detail=causal_err,
                )
                return False, f"Rejected: Causal inconsistency detected. {causal_err}"

        # Admission granted: promote to semantic_facts
        promoted_fact = SemanticFact(

            fact_id=str(uuid.uuid4()),
            candidate_id=candidate.candidate_id,
            source_episode_id=candidate.source_episode_id,
            session_id=candidate.session_id,
            subject=candidate.subject,
            predicate=candidate.predicate,
            object=candidate.object,
            confidence=candidate.confidence,
            valid_from=datetime.now(timezone.utc),
            valid_until=None,
            is_active=True,
            promoted_at=datetime.now(timezone.utc),
            provenance={
                "extractor_model": candidate.extractor_model,
                "quarantine_created_at": candidate.created_at.isoformat(),
                "promoted_by": "MemoryPromoter_v1",
            },
        )

        self.fact_store.insert_semantic_fact(promoted_fact)
        self.fact_store.update_candidate_status(
            candidate_id,
            status=AdmissionStatus.PROMOTED,
            reason=None,
            detail=f"Promoted to semantic fact {promoted_fact.fact_id}.",
        )

        return True, f"Promoted to Fact ID: {promoted_fact.fact_id}"

    def _check_contradiction(self, candidate: CandidateFact, existing: SemanticFact) -> tuple[bool, str]:
        """Detect logical or semantic contradictions between new candidate and existing fact."""
        c_pred = candidate.predicate.lower()
        e_pred = existing.predicate.lower()
        c_obj = candidate.object.lower()
        e_obj = existing.object.lower()

        # Negation / Polarity conflict
        negation_terms = {"no", "none", "denies", "negative", "false", "absent", "zero"}
        affirmative_terms = {"yes", "positive", "true", "present", "active", "has"}

        c_is_neg = any(term in c_obj for term in negation_terms)
        e_is_neg = any(term in e_obj for term in negation_terms)

        if c_pred == e_pred and c_is_neg != e_is_neg:
            return True, f"Polarity conflict on predicate '{c_pred}': '{c_obj}' contradicts existing '{e_obj}'."

        # Medical allergy specific conflict: "no known allergies" vs specific allergen
        if "allergy" in candidate.subject or "allerg" in c_pred:
            if "none" in e_obj or "no known" in e_obj or "denies" in e_obj:
                if not c_is_neg:
                    return True, f"Allergy assertion '{c_obj}' contradicts previous record of no allergies '{e_obj}'."

        # Direct mutually exclusive values for matching predicate
        if c_pred == e_pred and c_obj != e_obj:
            mutually_exclusive_predicates = {"biological_sex", "blood_type", "rh_factor", "date_of_birth"}
            if c_pred in mutually_exclusive_predicates:
                return True, f"Mutually exclusive value for '{c_pred}': '{c_obj}' vs '{e_obj}'."

        return False, ""

    def _is_legitimate_temporal_supersede(self, candidate: CandidateFact, existing: SemanticFact) -> bool:
        """Determine if new candidate is an intentional chronological update rather than a hallucination."""
        # High confidence update where user explicitly updated an evolving state
        if candidate.confidence >= 0.90 and "allergy" in candidate.subject:
            return True
        if candidate.confidence >= 0.90 and candidate.predicate.lower() in {"status", "current_medication", "treatment_plan"}:
            return True
        return False

    def _check_causal_consistency(self, candidate: CandidateFact) -> tuple[bool, str]:
        """Verify that asserted candidate fact is causally plausible under the SCM."""
        if not self.causal_reasoner or not hasattr(self.causal_reasoner, "scm"):
            return True, ""

        subj = candidate.subject.lower()
        pred = candidate.predicate.lower()
        obj = str(candidate.object).lower()
        scm = self.causal_reasoner.scm

        # Causal Rule 1: Anaphylaxis requires prior sensitized allergy or beta-lactam exposure
        if "anaphylaxis" in subj or "allergic" in pred or "anaphylaxis" in obj:
            if "penicillin_allergy" in scm.nodes:
                allergy_sensitized = scm.nodes["penicillin_allergy"].observed_value
                if allergy_sensitized is False and "penicillin" in obj:
                    return False, "Asserts beta-lactam anaphylaxis but patient has no sensitized beta-lactam allergy in SCM."

        # Causal Rule 2: Blood pressure normalization requires treatment or lifestyle intervention
        if ("hypertension" in subj or "blood_pressure" in subj) and ("cured" in pred or "normalized" in pred):
            if "without medication" in obj or "spontaneous" in obj:
                return False, "Asserts spontaneous normalization of essential hypertension without pharmacological intervention or lifestyle mechanism."

        return True, ""

