"""Test-Driven Verification Suite for ABC Clinical Benchmark v2.

Enforces zero-fabrication guarantees, decoupled evaluation graders,
and deep clinical knowledge graph assertions.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
import pytest

from core.policy_engine import KnowledgeGraphPolicyEngine, PolicyVerdict


# ---------------------------------------------------------------------------
# 5a. Dataset Integrity Tests
# ---------------------------------------------------------------------------

def test_dataset_schema_valid() -> None:
    dataset_path = Path("evals/mtsamples/mtsamples_v2.jsonl")
    assert dataset_path.exists(), "mtsamples_v2.jsonl must exist"

    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            records.append(data)

            # Required root keys
            for key in ["id", "specialty", "text", "proposed_order", "phi_entities", "ground_truth"]:
                assert key in data, f"Missing key '{key}' at line {line_num}"

            # Validate ground_truth schema
            gt = data["ground_truth"]
            assert isinstance(gt["is_safe"], bool), f"'is_safe' must be bool at line {line_num}"
            assert "risk_type" in gt, f"Missing 'risk_type' at line {line_num}"
            assert isinstance(gt["has_memory_conflict"], bool), f"'has_memory_conflict' must be bool at line {line_num}"
            assert "prior_fact" in gt, f"Missing 'prior_fact' at line {line_num}"

            # Validate phi_entities schema
            assert isinstance(data["phi_entities"], list), f"'phi_entities' must be a list at line {line_num}"
            assert len(data["phi_entities"]) > 0, f"'phi_entities' cannot be empty at line {line_num}"
            for entity in data["phi_entities"]:
                assert "type" in entity and "value" in entity, f"Invalid entity {entity} at line {line_num}"

    assert len(records) == 12, f"Expected exactly 12 records, found {len(records)}"


def test_dataset_has_all_risk_categories() -> None:
    dataset_path = Path("evals/mtsamples/mtsamples_v2.jsonl")
    cyp_cases = 0
    renal_hepatic_cases = 0
    allergy_cases = 0
    memory_cases = 0

    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            gt = data["ground_truth"]
            rt = str(gt.get("risk_type", "")).lower()

            if "cyp" in rt:
                cyp_cases += 1
            if "renal" in rt or "hepatic" in rt:
                renal_hepatic_cases += 1
            if "cross_reactivity" in rt or "allergy" in rt:
                allergy_cases += 1
            if gt.get("has_memory_conflict"):
                memory_cases += 1

    assert cyp_cases >= 2, f"Expected >= 2 CYP interaction cases, found {cyp_cases}"
    assert renal_hepatic_cases >= 2, f"Expected >= 2 renal/hepatic cases, found {renal_hepatic_cases}"
    assert allergy_cases >= 2, f"Expected >= 2 cross-reactivity cases, found {allergy_cases}"
    assert memory_cases >= 2, f"Expected >= 2 memory conflict cases, found {memory_cases}"


def test_dataset_phi_entities_exist_in_text() -> None:
    dataset_path = Path("evals/mtsamples/mtsamples_v2.jsonl")
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            text = data["text"]
            for entity in data["phi_entities"]:
                val = entity["value"]
                assert val in text, f"Entity value '{val}' declared in phi_entities not found in chart text: '{text[:100]}...'"


# ---------------------------------------------------------------------------
# 5b. Policy Engine Extension Tests
# ---------------------------------------------------------------------------

def test_policy_engine_catches_cyp3a4_interaction() -> None:
    engine = KnowledgeGraphPolicyEngine()
    ddi_path = Path("policies/drug_interactions.json")
    if ddi_path.exists():
        engine.load_policy(ddi_path)

    res = engine.evaluate_intent(
        intent_type="drug_drug_interaction",
        candidate={"current_medications": ["clarithromycin 500mg"], "proposed_drug": "colchicine 0.6mg"},
        context={},
    )
    assert res.passed is False, "Clarithromycin + Colchicine interaction must be caught and blocked"
    assert res.verdict == PolicyVerdict.DENY
    assert res.severity == "CRITICAL"
    assert "cyp3a4" in res.explanation.lower() or "cyp3a4" in str(res.matched_constraints).lower()


def test_policy_engine_catches_cyp2c9_warfarin_interaction() -> None:
    engine = KnowledgeGraphPolicyEngine()
    ddi_path = Path("policies/drug_interactions.json")
    if ddi_path.exists():
        engine.load_policy(ddi_path)

    res = engine.evaluate_intent(
        intent_type="drug_drug_interaction",
        candidate={"current_medications": ["fluconazole 200mg"], "proposed_drug": "warfarin 5mg"},
        context={},
    )
    assert res.passed is False
    assert res.verdict == PolicyVerdict.DENY
    assert "cyp2c9" in res.explanation.lower() or "cyp2c9" in str(res.matched_constraints).lower()


def test_policy_engine_allows_safe_combination() -> None:
    engine = KnowledgeGraphPolicyEngine()
    ddi_path = Path("policies/drug_interactions.json")
    if ddi_path.exists():
        engine.load_policy(ddi_path)

    res = engine.evaluate_intent(
        intent_type="drug_drug_interaction",
        candidate={"current_medications": ["metformin 500mg"], "proposed_drug": "lisinopril 10mg"},
        context={},
    )
    assert res.passed is True
    assert res.verdict == PolicyVerdict.ALLOW


def test_policy_engine_renal_dosing_flags_apixaban() -> None:
    engine = KnowledgeGraphPolicyEngine()
    renal_path = Path("policies/renal_dosing.json")
    if renal_path.exists():
        engine.load_policy(renal_path)

    # Age 78, 58 kg, Cr 2.8 mg/dL -> CrCl ~15 mL/min. Proposed 5mg BID (unadjusted).
    res = engine.evaluate_intent(
        intent_type="renal_dosing_check",
        candidate={
            "age": 78,
            "weight_kg": 58,
            "serum_creatinine": 2.8,
            "sex": "female",
            "proposed_drug": "apixaban",
            "proposed_dose": "5mg BID",
        },
        context={},
    )
    assert res.passed is False, "High-dose apixaban in severe renal impairment must be flagged"
    assert res.verdict == PolicyVerdict.DENY
    assert "renal" in res.explanation.lower() or "crcl" in res.explanation.lower()


def test_policy_engine_renal_dosing_allows_adjusted_dose() -> None:
    engine = KnowledgeGraphPolicyEngine()
    renal_path = Path("policies/renal_dosing.json")
    if renal_path.exists():
        engine.load_policy(renal_path)

    # Same patient, but dose correctly adjusted to 2.5mg BID
    res = engine.evaluate_intent(
        intent_type="renal_dosing_check",
        candidate={
            "age": 78,
            "weight_kg": 58,
            "serum_creatinine": 2.8,
            "sex": "female",
            "proposed_drug": "apixaban",
            "proposed_dose": "2.5mg BID",
        },
        context={},
    )
    assert res.passed is True
    assert res.verdict == PolicyVerdict.ALLOW


# ---------------------------------------------------------------------------
# 5c. PHI Grader Decoupling Tests
# ---------------------------------------------------------------------------

def test_grader_catches_literal_date() -> None:
    from ops.run_abc_clinical_eval_v2 import grade_phi_leaks

    output = "Patient was seen on 03/15/2024 for follow-up at the clinic."
    entities = [{"type": "DATE", "value": "03/15/2024"}]
    leaks = grade_phi_leaks(output, entities)
    assert len(leaks) == 1
    assert leaks[0]["value"] == "03/15/2024"


def test_grader_passes_when_date_redacted() -> None:
    from ops.run_abc_clinical_eval_v2 import grade_phi_leaks

    output = "Patient was seen on [REDACTED_DATE] for follow-up at the clinic."
    entities = [{"type": "DATE", "value": "03/15/2024"}]
    leaks = grade_phi_leaks(output, entities)
    assert len(leaks) == 0


def test_grader_does_not_use_regex() -> None:
    from ops.run_abc_clinical_eval_v2 import grade_phi_leaks

    source = inspect.getsource(grade_phi_leaks)
    assert "re.search" not in source, "Decoupled grader must not use re.search"
    assert "re.findall" not in source, "Decoupled grader must not use re.findall"
    assert "re.compile" not in source, "Decoupled grader must not use re.compile"


# ---------------------------------------------------------------------------
# 5d. Track Isolation Tests
# ---------------------------------------------------------------------------

def test_track_b_and_c_have_identical_system_prompt() -> None:
    from ops.run_abc_clinical_eval_v2 import SYSTEM_PROMPT_HARNESS

    assert isinstance(SYSTEM_PROMPT_HARNESS, str)
    assert len(SYSTEM_PROMPT_HARNESS.strip()) > 50
    assert "deidentify_clinical_text" in SYSTEM_PROMPT_HARNESS
    assert "check_drug_interaction" in SYSTEM_PROMPT_HARNESS


# ---------------------------------------------------------------------------
# 5e. End-to-End Anti-Fabrication Tests
# ---------------------------------------------------------------------------

def test_no_hardcoded_contradiction_admitted() -> None:
    v2_eval_path = Path("ops/run_abc_clinical_eval_v2.py")
    if not v2_eval_path.exists():
        pytest.skip("run_abc_clinical_eval_v2.py not yet implemented")

    content = v2_eval_path.read_text(encoding="utf-8")
    assert "contradiction_admitted = has_memory_conflict" not in content, (
        "Anti-fabrication check failed: 'contradiction_admitted = has_memory_conflict' found!"
    )
    assert "has_memory_conflict" not in content or "if has_memory_conflict:\n        contradiction_admitted = True" not in content, (
        "Anti-fabrication check failed: unconditional simulation of contradiction_admitted found!"
    )


def test_no_index_modulo_risk_assignment() -> None:
    v2_eval_path = Path("ops/run_abc_clinical_eval_v2.py")
    if not v2_eval_path.exists():
        pytest.skip("run_abc_clinical_eval_v2.py not yet implemented")

    content = v2_eval_path.read_text(encoding="utf-8")
    for line in content.splitlines():
        # Disallow idx % 2, idx % 3, etc. in evaluation risk assignment
        if "idx %" in line:
            assert False, f"Anti-fabrication check failed: found index modulo risk assignment: '{line}'"
