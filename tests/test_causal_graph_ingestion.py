"""Unit tests for Phase 7: Causal Graph Ingestion and Observational Verification.

Covers:
- JSON and GraphML round-trip serialization and deserialization
- Exact d-separation on colliders, chains, forks, and clinical graphs
- Implied conditional independence discovery
- Observational verification using partial correlation and conditional G-tests
- Violation detection when observational data contradicts DAG assumptions
- Integration with CausalReasoner and typed causal tools
"""

from __future__ import annotations

import json
import numpy as np
import pytest
import xml.etree.ElementTree as ET

from core.causal_graph import CausalNodeType, StructuralCausalModel
from core.causal_reasoner import CausalReasoner
from core.causal_verifier import ObservationalCausalVerifier
from tools.causal_tools import LoadCausalGraphTool, VerifyCausalAssumptionsTool


def test_scm_json_serialization_roundtrip(tmp_path):
    """Verify complete JSON export and import fidelity including formula mechanisms."""
    scm = StructuralCausalModel(name="pediatric_asthma")
    scm.add_node("pollen_count", "Environmental allergen level", CausalNodeType.EXOGENOUS, baseline_value=50.0)
    scm.add_node("inhaler_use", "Albuterol administration", CausalNodeType.INTERVENTION, baseline_value=0)
    scm.add_node("airway_inflammation", "Inflammation score", CausalNodeType.ENDOGENOUS, baseline_value=10.0)
    scm.add_node("peak_flow", "Peak expiratory flow rate", CausalNodeType.OUTCOME, baseline_value=350.0)

    scm.add_edge("pollen_count", "airway_inflammation", "Pollen triggers inflammatory response", weight=1.2)
    scm.add_edge("inhaler_use", "peak_flow", "Bronchodilator expands airways", weight=2.5)
    scm.add_edge("airway_inflammation", "peak_flow", "Inflammation restricts airflow", weight=-1.8)

    scm.register_formula_mechanism("airway_inflammation", "min(100.0, pollen_count * 0.8)")
    scm.register_formula_mechanism("peak_flow", "max(50.0, 400.0 - airway_inflammation * 2.0 + inhaler_use * 50.0)")

    # Serialize to JSON file
    file_path = tmp_path / "asthma_scm.json"
    json_str = scm.to_json(filepath=str(file_path))

    # Deserialize from string and from file
    loaded_from_str = StructuralCausalModel.from_json(json_str)
    loaded_from_file = StructuralCausalModel.from_json(str(file_path))

    for target in [loaded_from_str, loaded_from_file]:
        assert target.name == "pediatric_asthma"
        assert len(target.nodes) == 4
        assert len(target.all_edges) == 3
        assert target.nodes["pollen_count"].node_type == CausalNodeType.EXOGENOUS
        assert target.nodes["inhaler_use"].node_type == CausalNodeType.INTERVENTION
        assert "airway_inflammation" in target.formula_mechanisms

        # Verify simulation equivalence
        target.set_evidence({"pollen_count": 60.0, "inhaler_use": 1})
        state = target.forward_simulate()
        assert state["airway_inflammation"] == 48.0
        assert state["peak_flow"] == (400.0 - 48.0 * 2.0 + 50.0)


def test_scm_graphml_serialization_roundtrip(tmp_path):
    """Verify standard GraphML XML export and import structure."""
    scm = StructuralCausalModel(name="cardiology_triage")
    scm.add_node("troponin", "Cardiac marker", CausalNodeType.EXOGENOUS, baseline_value="0.02")
    scm.add_node("st_elevation", "ECG finding", CausalNodeType.EXOGENOUS, baseline_value="true")
    scm.add_node("cath_lab_activation", "Emergency angiogram decision", CausalNodeType.INTERVENTION, baseline_value="false")

    scm.add_edge("troponin", "cath_lab_activation", "Elevated enzymes trigger activation", weight=2.0)
    scm.add_edge("st_elevation", "cath_lab_activation", "STEMI pattern triggers immediate cath lab", weight=3.5)

    graphml_file = tmp_path / "cardio.graphml"
    xml_str = scm.to_graphml(filepath=str(graphml_file))

    # Verify standard XML syntax
    root = ET.fromstring(xml_str)
    assert root.tag.split("}")[-1] == "graphml"

    # Reload model from file
    loaded = StructuralCausalModel.from_graphml(str(graphml_file))
    assert loaded.name == "cardiology_triage"
    assert len(loaded.nodes) == 3
    assert len(loaded.all_edges) == 2
    assert "troponin" in loaded.nodes
    assert "cath_lab_activation" in loaded.nodes


def test_pearl_d_separation_canonical_structures():
    """Verify d-separation across colliders, chains, and forks."""
    # 1. Collider: A -> C <- B
    # A and B are marginally independent, but become dependent when conditioning on C or C's child D
    collider_scm = StructuralCausalModel("collider")
    collider_scm.add_node("A")
    collider_scm.add_node("B")
    collider_scm.add_node("C")
    collider_scm.add_node("D")
    collider_scm.add_edge("A", "C")
    collider_scm.add_edge("B", "C")
    collider_scm.add_edge("C", "D")

    assert collider_scm.is_d_separated("A", "B", set()) is True
    assert collider_scm.is_d_separated("A", "B", {"C"}) is False
    assert collider_scm.is_d_separated("A", "B", {"D"}) is False

    # 2. Chain: A -> B -> C
    # A and C are marginally dependent, but independent when conditioning on B
    chain_scm = StructuralCausalModel("chain")
    chain_scm.add_node("A")
    chain_scm.add_node("B")
    chain_scm.add_node("C")
    chain_scm.add_edge("A", "B")
    chain_scm.add_edge("B", "C")

    assert chain_scm.is_d_separated("A", "C", set()) is False
    assert chain_scm.is_d_separated("A", "C", {"B"}) is True

    # 3. Fork: A <- B -> C
    # A and C share common cause B; independent when conditioning on B
    fork_scm = StructuralCausalModel("fork")
    fork_scm.add_node("A")
    fork_scm.add_node("B")
    fork_scm.add_node("C")
    fork_scm.add_edge("B", "A")
    fork_scm.add_edge("B", "C")

    assert fork_scm.is_d_separated("A", "C", set()) is False
    assert fork_scm.is_d_separated("A", "C", {"B"}) is True


def test_d_separation_clinical_pharmacotherapy():
    """Verify d-separation implications on the clinical pharmacotherapy model."""
    reasoner = CausalReasoner()
    scm = reasoner.scm

    # Baseline patient variables are exogenous root nodes
    assert scm.is_d_separated("patient_age", "drug_prescription", set()) is True
    assert scm.is_d_separated("penicillin_allergy", "dosage_mg", set()) is True

    # Penicillin allergy directly influences anaphylaxis reaction via beta-lactam class
    assert scm.is_d_separated("penicillin_allergy", "anaphylaxis_reaction", set()) is False

    # Baseline GFR influences AKI exclusively through renal clearance
    assert scm.is_d_separated("baseline_gfr", "acute_kidney_injury", set()) is False
    assert scm.is_d_separated("baseline_gfr", "acute_kidney_injury", {"renal_clearance"}) is True


def test_find_implied_independencies():
    """Verify automated extraction of testable conditional independencies from DAG topology."""
    scm = StructuralCausalModel("test_discovery")
    scm.add_node("X")
    scm.add_node("M")
    scm.add_node("Y")
    scm.add_edge("X", "M")
    scm.add_edge("M", "Y")

    indeps = scm.find_implied_independencies(max_conditioning_size=1)
    assert len(indeps) == 1
    assert indeps[0]["var_x"] == "X"
    assert indeps[0]["var_y"] == "Y"
    assert indeps[0]["conditioning_set"] == ["M"]


def test_observational_verification_linear_continuous_data():
    """Verify observational verification engine with synthetic continuous data."""
    # SCM: X -> M -> Y (Chain where X _||_ Y | M)
    scm = StructuralCausalModel("linear_chain")
    scm.add_node("X")
    scm.add_node("M")
    scm.add_node("Y")
    scm.add_edge("X", "M")
    scm.add_edge("M", "Y")

    # Generate 400 data points strictly adhering to the DAG
    np.random.seed(42)
    n = 400
    x = np.random.normal(0, 1, n)
    m = 0.8 * x + np.random.normal(0, 0.4, n)
    y = 0.7 * m + np.random.normal(0, 0.4, n)

    data = [{"X": float(x[i]), "M": float(m[i]), "Y": float(y[i])} for i in range(n)]

    verifier = ObservationalCausalVerifier(alpha=0.05)
    report = verifier.verify_dataset(scm, data)

    # All DAG assumptions should pass cleanly
    assert report.is_valid is True
    assert report.failed_tests == 0
    assert report.passed_tests > 0

    # Locate the implied independence test (X _||_ Y | M)
    indep_test = next(t for t in report.all_results if t.conditioning_set == ["M"])
    assert indep_test.predicted_independent is True
    assert indep_test.empirically_independent is True
    assert indep_test.p_value >= 0.05


def test_observational_verification_detects_dag_violations():
    """Verify that verifier flags violations when data contradicts the assumed DAG."""
    # We claim DAG is X -> M and M -> Y with no direct X -> Y edge (predicting X _||_ Y | M)
    scm = StructuralCausalModel("incorrect_chain")
    scm.add_node("X")
    scm.add_node("M")
    scm.add_node("Y")
    scm.add_edge("X", "M")
    scm.add_edge("M", "Y")

    # Ground truth data has a direct causal link or unmodeled confounder between X and Y
    np.random.seed(42)
    n = 400
    x = np.random.normal(0, 1, n)
    m = 0.8 * x + np.random.normal(0, 0.4, n)
    # Direct strong effect of X on Y bypassing M
    y = 0.7 * m + 0.9 * x + np.random.normal(0, 0.4, n)

    data = [{"X": float(x[i]), "M": float(m[i]), "Y": float(y[i])} for i in range(n)]

    verifier = ObservationalCausalVerifier(alpha=0.05)
    report = verifier.verify_dataset(scm, data)

    # Must detect the broken conditional independence
    assert report.is_valid is False
    assert report.failed_tests >= 1
    assert len(report.violations) >= 1

    violation = report.violations[0]
    assert violation.var_x == "X"
    assert violation.var_y == "Y"
    assert violation.conditioning_set == ["M"]
    assert violation.predicted_independent is True
    assert violation.empirically_independent is False
    assert violation.p_value < 0.001
    assert "DAG predicts independence" in violation.violation_detail


def test_observational_verification_discrete_categorical_data():
    """Verify conditional G-test on categorical clinical features."""
    scm = StructuralCausalModel("discrete_clinical")
    scm.add_node("allergy_status")
    scm.add_node("drug_administered")
    scm.add_node("rash")
    scm.add_edge("allergy_status", "rash")
    scm.add_edge("drug_administered", "rash")

    # Generate categorical records where allergy_status and drug_administered are randomized independently
    np.random.seed(123)
    n = 300
    data = []
    drugs = ["amoxicillin", "ramipril", "paracetamol"]
    allergies = ["penicillin_allergic", "no_known_allergy"]

    for _ in range(n):
        dr = str(np.random.choice(drugs))
        al = str(np.random.choice(allergies))
        # Rash occurs if drug is amoxicillin and patient is allergic
        has_rash = "yes" if (dr == "amoxicillin" and al == "penicillin_allergic") else "no"
        data.append({
            "allergy_status": al,
            "drug_administered": dr,
            "rash": has_rash,
        })

    reasoner = CausalReasoner(scm=scm)
    report = reasoner.verify_observational_data(
        data=data,
        custom_tests=[
            {"x": "allergy_status", "y": "drug_administered", "z": []},
        ],
    )

    assert report.is_valid is True
    res = report.all_results[0]
    assert res.predicted_independent is True
    assert res.empirically_independent is True
    assert res.p_value >= 0.05


def test_causal_tools_load_and_verify():
    """Verify LoadCausalGraphTool and VerifyCausalAssumptionsTool executions."""
    reasoner = CausalReasoner()
    load_tool = LoadCausalGraphTool(reasoner=reasoner)
    verify_tool = VerifyCausalAssumptionsTool(reasoner=reasoner)

    # 1. Load model via JSON string
    dag_json = json.dumps({
        "name": "oncology_trial",
        "nodes": [
            {"name": "biomarker_positive", "node_type": "exogenous"},
            {"name": "targeted_therapy", "node_type": "intervention"},
            {"name": "progression_free_survival_months", "node_type": "outcome"},
        ],
        "edges": [
            {"source": "biomarker_positive", "target": "progression_free_survival_months"},
            {"source": "targeted_therapy", "target": "progression_free_survival_months"},
        ],
    })

    load_res = load_tool.execute({"format": "json", "content": dag_json})
    assert load_res.is_error is False
    assert load_res.output["scm_name"] == "oncology_trial"
    assert load_res.output["total_nodes"] == 3
    assert load_res.output["total_edges"] == 2
    assert reasoner.scm.name == "oncology_trial"

    # 2. Verify observational data with loaded tool
    np.random.seed(99)
    n = 250
    b = np.random.choice([0, 1], n)
    t = np.random.choice([0, 1], n)
    pfs = 6.0 + 4.0 * b + 8.0 * t + np.random.normal(0, 1.5, n)

    obs_records = [
        {
            "biomarker_positive": int(b[i]),
            "targeted_therapy": int(t[i]),
            "progression_free_survival_months": float(pfs[i]),
        }
        for i in range(n)
    ]

    verify_res = verify_tool.execute({
        "data": obs_records,
        "custom_tests": [
            {"x": "biomarker_positive", "y": "targeted_therapy", "z": []},
        ],
    })

    assert verify_res.is_error is False
    assert verify_res.output["is_valid"] is True
    assert verify_res.output["failed_tests"] == 0


def test_safe_ast_formula_blocks_code_execution():
    """Verify that formula evaluation rejects attribute traversal and code execution attempts."""
    scm = StructuralCausalModel("safety_test")
    scm.add_node("x")
    scm.add_node("y")
    scm.add_edge("x", "y")

    # Safe formula works as expected
    scm.register_formula_mechanism("y", "min(50.0, x * 2.0 + 5.0)")
    val = scm.mechanisms["y"]({"x": 10.0}, None)
    assert val == 25.0

    # Malicious attribute traversal is rejected by AST visitor
    scm.register_formula_mechanism("y", "().__class__.__bases__[0]")
    with pytest.raises(ValueError, match="Disallowed expression node in causal formula"):
        scm.mechanisms["y"]({"x": 10.0}, None)
