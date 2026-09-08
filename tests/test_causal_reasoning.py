"""Comprehensive unit tests for Causal Reasoning, Pearl's do-calculus, and Counterfactuals."""

import pytest
from core.causal_graph import CausalNodeType, StructuralCausalModel
from core.causal_reasoner import CausalReasoner
from core.memory_promoter import MemoryPromoter
from domain.models import AdmissionStatus, CandidateFact, RejectionReason
from infrastructure.sqlite_fact_store import SQLiteFactStore
from tools.causal_tools import (
    CausalGraphQueryTool,
    ExplainCounterfactualAttributionTool,
    SimulateCausalInterventionTool,
)


def test_scm_dag_creation_and_cycle_detection():
    """Verify that SCM enforces acyclicity and computes topological ordering."""
    scm = StructuralCausalModel(name="test_dag")

    scm.add_node("A", node_type=CausalNodeType.EXOGENOUS)
    scm.add_node("B", node_type=CausalNodeType.ENDOGENOUS)
    scm.add_node("C", node_type=CausalNodeType.OUTCOME)

    scm.add_edge("A", "B", "A causes B")
    scm.add_edge("B", "C", "B causes C")

    order = scm.topological_sort()
    assert order == ["A", "B", "C"]

    # Introducing a cycle C -> A must be rejected
    with pytest.raises(ValueError) as exc_info:
        scm.add_edge("C", "A", "Cycle trigger")
    assert "creates a cycle" in str(exc_info.value)


def test_scm_backdoor_path_detection():
    """Verify detection of confounding backdoor paths between treatment and outcome."""
    scm = StructuralCausalModel(name="confounded_trial")

    # Confounder Z (e.g. disease severity) affects both Treatment X and Outcome Y
    scm.add_node("Z", "Confounder", CausalNodeType.EXOGENOUS)
    scm.add_node("X", "Treatment", CausalNodeType.INTERVENTION)
    scm.add_node("Y", "Outcome", CausalNodeType.OUTCOME)

    scm.add_edge("Z", "X", "Severity affects treatment choice")
    scm.add_edge("Z", "Y", "Severity affects recovery outcome")
    scm.add_edge("X", "Y", "Treatment effect on outcome")

    backdoor_paths = scm.find_backdoor_paths("X", "Y")
    assert len(backdoor_paths) == 1
    # Path: X <- Z -> Y
    assert backdoor_paths[0] == ["X", "Z", "Y"]


def test_pearl_do_intervention_mutilated_graph():
    """Pearl's Level 2: Verify do(X=x) severs incoming arrows into X."""
    reasoner = CausalReasoner()
    scm = reasoner.scm

    # In baseline SCM: drug_prescription has incoming or internal mechanisms
    # Intervene: do(drug_prescription = 'amoxicillin', dosage_mg = 500)
    patient_context = {"penicillin_allergy": True, "baseline_gfr": 75.0}

    sim_res = reasoner.simulate_intervention_safety(
        proposed_action={"drug_prescription": "amoxicillin", "dosage_mg": 500.0},
        patient_evidence=patient_context,
    )

    # Anaphylaxis is triggered because amoxicillin (beta-lactam) + allergy = True
    assert sim_res["is_safe"] is False
    assert sim_res["simulated_outcomes"]["anaphylaxis_reaction"] is True
    assert any("ANAPHYLAXIS" in w for w in sim_res["warnings"])


def test_counterfactual_attribution_level_3():
    """Pearl's Level 3: 3-step counterfactual attribution (Abduction -> Action -> Prediction)."""
    reasoner = CausalReasoner()

    # Factual Evidence: Patient with penicillin allergy was given amoxicillin, resulting in anaphylaxis
    factual_world = {
        "penicillin_allergy": True,
        "drug_prescription": "amoxicillin",
        "dosage_mg": 500.0,
    }

    # Counterfactual Query: What if the agent had prescribed 'ramipril' instead?
    counterfactual_query = {"drug_prescription": "ramipril", "dosage_mg": 5.0}

    attribution = reasoner.explain_counterfactual_attribution(
        actual_evidence=factual_world,
        hypothetical_alternative=counterfactual_query,
        observed_bad_outcome="anaphylaxis_reaction",
    )

    assert attribution["factual_outcome"] is True
    assert attribution["counterfactual_outcome"] is False
    assert attribution["outcome_prevented"] is True
    assert "necessary cause" in attribution["explanation"]


def test_causal_tools_execution():
    """Verify causal tools execute cleanly inside the ToolRegistry."""
    sim_tool = SimulateCausalInterventionTool()
    cf_tool = ExplainCounterfactualAttributionTool()
    query_tool = CausalGraphQueryTool()

    # 1. Simulate safe intervention: ramipril 5mg for hypertension
    sim_res = sim_tool.execute({
        "intervention": {"drug_prescription": "ramipril", "dosage_mg": 5.0},
        "patient_context": {"penicillin_allergy": False, "baseline_gfr": 80.0},
    })
    assert sim_res.is_error is False
    assert sim_res.output["is_safe"] is True
    assert sim_res.output["simulated_outcomes"]["blood_pressure_reduction"] == 12.5

    # 2. Counterfactual explanation
    cf_res = cf_tool.execute({
        "factual_evidence": {"penicillin_allergy": True, "drug_prescription": "amoxicillin"},
        "hypothetical_action": {"drug_prescription": "ramipril"},
        "adverse_outcome_target": "anaphylaxis_reaction",
    })
    assert cf_res.is_error is False
    assert cf_res.output["outcome_prevented_by_alternative"] is True

    # 3. Query causal graph
    q_res = query_tool.execute({
        "treatment_node": "drug_prescription",
        "outcome_node": "anaphylaxis_reaction",
    })
    assert q_res.is_error is False
    assert len(q_res.output["direct_mechanisms"]) > 0


def test_memory_gate_causal_consistency_rejection():
    """Verify that Tri-State Memory Gate rejects causally impossible candidate facts."""
    fact_store = SQLiteFactStore(":memory:")
    reasoner = CausalReasoner()
    promoter = MemoryPromoter(fact_store=fact_store, causal_reasoner=reasoner)

    # Candidate claims spontaneous cure of essential hypertension without medication
    bad_cand = CandidateFact(
        source_episode_id="ep_causal_test",
        session_id="patient_causal_1",
        subject="patient_essential_hypertension",
        predicate="cured",
        object="spontaneous normalization without medication or intervention",
        confidence=0.95,
    )

    promoter.submit_candidate(bad_cand)
    promoted, reason = promoter.evaluate_and_promote(bad_cand.candidate_id)

    assert promoted is False
    assert "Causal inconsistency detected" in reason

    records = fact_store.get_quarantine_records()
    assert records[0].status == AdmissionStatus.REJECTED
    assert records[0].rejection_reason == RejectionReason.CAUSAL_INCONSISTENCY
