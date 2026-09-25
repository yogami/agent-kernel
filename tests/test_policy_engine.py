"""Unit tests for KnowledgeGraphPolicyEngine and PolicyEnginePort."""

from pathlib import Path
import pytest
from core.policy_engine import (
    KnowledgeGraphPolicyEngine,
    PolicyEnginePort,
    PolicyVerdict,
)


def test_protocol_conformance():
    """Verify that KnowledgeGraphPolicyEngine satisfies PolicyEnginePort protocol."""
    engine = KnowledgeGraphPolicyEngine()
    assert isinstance(engine, PolicyEnginePort)


def test_clinical_contraindication_intent():
    """Verify structured intent evaluation for clinical contraindications."""
    engine = KnowledgeGraphPolicyEngine()

    # Case 1: Penicillin allergy with proposed Cephalosporin (Cefazolin) -> DENY
    res_deny = engine.evaluate_intent(
        intent_type="drug_allergy_conflict",
        candidate={"prescribed_drug": "cefazolin 1g IV"},
        context={"patient_allergies": ["penicillin"]},
    )
    assert res_deny.passed is False
    assert res_deny.verdict == PolicyVerdict.DENY
    assert res_deny.category == "beta_lactam"
    assert res_deny.rule_id == "RULE-RX-001"
    assert res_deny.severity == "HIGH"
    assert "cross-reactive" in res_deny.explanation

    # Case 2: Aspirin allergy with proposed Toradol (NSAID) -> DENY
    res_nsaid = engine.evaluate_intent(
        intent_type="drug_allergy_conflict",
        candidate={"prescribed_drug": "toradol 30mg"},
        context={"patient_allergies": ["aspirin allergy"]},
    )
    assert res_nsaid.passed is False
    assert res_nsaid.verdict == PolicyVerdict.DENY
    assert res_nsaid.category == "nsaid"

    # Case 3: Penicillin allergy with safe proposed drug (Vancomycin) -> ALLOW
    res_allow = engine.evaluate_intent(
        intent_type="drug_allergy_conflict",
        candidate={"prescribed_drug": "vancomycin 1g IV"},
        context={"patient_allergies": ["penicillin"]},
    )
    assert res_allow.passed is True
    assert res_allow.verdict == PolicyVerdict.ALLOW
    assert res_allow.severity == "NONE"


def test_quantitative_vital_bounds():
    """Verify physiological vital bounds assertions."""
    engine = KnowledgeGraphPolicyEngine()

    # Normal vitals
    res_ok = engine.evaluate_intent(
        intent_type="vital_bounds",
        candidate={"systolic_bp": 120, "diastolic_bp": 80, "heart_rate": 72},
        context={},
    )
    assert res_ok.passed is True
    assert res_ok.verdict == PolicyVerdict.ALLOW

    # Hypertensive crisis out of bound
    res_err = engine.evaluate_intent(
        intent_type="vital_bounds",
        candidate={"systolic_bp": 350, "heart_rate": 20},
        context={},
    )
    assert res_err.passed is False
    assert res_err.verdict == PolicyVerdict.DENY
    assert len(res_err.matched_constraints) == 2
    assert any("systolic_bp" in c for c in res_err.matched_constraints)
    assert any("heart_rate" in c for c in res_err.matched_constraints)


def test_cross_domain_fintech_policy():
    """Verify that the same engine loads and enforces fintech capital markets policies."""
    fintech_policy_path = Path(__file__).resolve().parent.parent / "policies" / "fintech_trading_rules.json"
    engine = KnowledgeGraphPolicyEngine(policy_source=fintech_policy_path)

    assert engine.domain == "fintech_capital_markets"

    # 1. Compliant order within notional limits
    res_compliant = engine.evaluate_intent(
        intent_type="trade_order",
        candidate={"ticker": "AAPL", "notional_usd": 150000.0},
        context={},
    )
    assert res_compliant.passed is True
    assert res_compliant.verdict == PolicyVerdict.ALLOW

    # 2. Blacklisted ticker -> DENY
    res_sanctioned = engine.evaluate_intent(
        intent_type="trade_order",
        candidate={"ticker": "SANCTIONED_CO", "notional_usd": 10000.0},
        context={},
    )
    assert res_sanctioned.passed is False
    assert res_sanctioned.verdict == PolicyVerdict.DENY
    assert res_sanctioned.rule_id == "RULE-FIN-001"
    assert res_sanctioned.severity == "CRITICAL"
    assert "blacklist" in res_sanctioned.explanation.lower()

    # 3. Exceeded notional risk cap (,500,000 > ,000,000) -> QUARANTINE
    res_oversized = engine.evaluate_intent(
        intent_type="trade_order",
        candidate={"ticker": "MSFT", "notional_usd": 1500000.0},
        context={},
    )
    assert res_oversized.passed is False
    assert res_oversized.verdict == PolicyVerdict.QUARANTINE
    assert res_oversized.rule_id == "RULE-FIN-002"
    assert "Risk Budget Violation" in res_oversized.explanation


def test_free_text_evaluation_negation_awareness():
    """Verify negation-aware policy scanning in unstructured text."""
    engine = KnowledgeGraphPolicyEngine()

    # Unsafe output: recommends contraindicated drug without negation
    unsafe_text = "Recommend initiating therapy with amoxicillin 500mg PO TID for 7 days."
    violations = engine.evaluate_free_text(unsafe_text, {"patient_allergies": ["penicillin"]})
    assert len(violations) == 1
    assert violations[0].passed is False
    assert violations[0].verdict == PolicyVerdict.DENY
    assert violations[0].category == "beta_lactam"
    assert "STRICTLY CONTRAINDICATED" in violations[0].explanation

    # Safe output: model explicitly withheld or noted contraindication
    safe_negated_text = (
        "Patient has a documented penicillin allergy. Beta-lactam antibiotics including amoxicillin "
        "are strictly contraindicated and withheld. Starting azithromycin instead."
    )
    negated_violations = engine.evaluate_free_text(safe_negated_text, {"patient_allergies": ["penicillin"]})
    assert len(negated_violations) == 0

    # Safe output: completely different medication class
    safe_unrelated_text = "Recommend initiating ciprofloxacin 500mg PO BID for acute uncomplicated UTI."
    unrelated_violations = engine.evaluate_free_text(safe_unrelated_text, {"patient_allergies": ["penicillin"]})
    assert len(unrelated_violations) == 0
