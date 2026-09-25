"""Tests for Clinical Curation Pipeline tools."""

from tools.clinical_tools import (
    ClinicalAssertionCheckerTool,
    DeidentifyTextTool,
    FHIRValidatorTool,
    MedicalOntologyMapperTool,
)


def test_deidentification_scrubber():
    """Verify German/English PII is redacted cleanly."""
    tool = DeidentifyTextTool()
    raw_text = "Patient: Herr Schmidt (PAT-9876), geb. 14.05.1982. Behandelt an der Charité Berlin."
    result = tool.execute({"text": raw_text})

    output = result.output
    sanitized = output["sanitized_text"]

    assert "[REDACTED_PATIENT]" in sanitized
    assert "[REDACTED_DATE]" in sanitized
    assert "[REDACTED_HOSPITAL]" in sanitized
    assert "Herr Schmidt" not in sanitized
    assert "14.05.1982" not in sanitized
    assert output["pii_leakage_risk"] == 0.0


def test_hybrid_ner_deidentification_unstructured():
    """Verify statistical NER catches unstructured entities (hospitals, doctors, patients, dates)."""
    tool = DeidentifyTextTool()
    unstructured_note = (
        "5 y/o boy admitted 10/17/92. Transferred to UIHC by Dr. Peterson. "
        "Mr. ABC was also evaluated at General Hospital in Baltimore."
    )
    result = tool.execute({"text": unstructured_note})
    sanitized = result.output["sanitized_text"]

    assert "[REDACTED_DATE]" in sanitized
    assert "[REDACTED_HOSPITAL]" in sanitized
    assert "[REDACTED_PATIENT]" in sanitized
    assert "UIHC" not in sanitized
    assert "Dr. Peterson" not in sanitized
    assert "10/17/92" not in sanitized
    assert result.output["pii_leakage_risk"] == 0.0


def test_ontology_mapping():
    """Verify medical entity extraction maps to SNOMED-CT and ICD-10."""
    tool = MedicalOntologyMapperTool()

    res_t2d = tool.execute({"entity_text": "Type 2 diabetes", "domain": "condition"})
    assert res_t2d.output["matched"] is True
    assert res_t2d.output["mapping"]["icd10"] == "E11.9"
    assert res_t2d.output["mapping"]["snomed"] == "44054006"

    res_stemi = tool.execute({"entity_text": "Z.n. STEMI", "domain": "condition"})
    assert res_stemi.output["matched"] is True
    assert res_stemi.output["mapping"]["icd10"] == "I25.2"


def test_fhir_validator():
    """Verify FHIR resource validation against HL7 FHIR (R4) specifications."""
    tool = FHIRValidatorTool()

    valid_condition = {
        "resourceType": "Condition",
        "subject": {"reference": "Patient/1001"},
        "code": {"coding": [{"system": "http://hl7.org/fhir/sid/icd-10", "code": "I10"}]},
    }
    res_valid = tool.execute({"resource_type": "Condition", "resource_json": valid_condition})
    assert res_valid.output["valid"] is True

    invalid_condition = {
        "resourceType": "Condition",
        # Missing subject and code
    }
    res_invalid = tool.execute({"resource_type": "Condition", "resource_json": invalid_condition})
    assert res_invalid.output["valid"] is False
    assert res_invalid.is_error is True


def test_clinical_assertions():
    """Verify biological range assertions and drug-allergy contraindication checks."""
    tool = ClinicalAssertionCheckerTool()

    # Drug allergy conflict
    res_conflict = tool.execute({
        "assertion_type": "drug_allergy_conflict",
        "parameters": {
            "patient_allergies": ["penicillin"],
            "prescribed_drug": "amoxicillin",
        },
    })
    assert res_conflict.output["passed"] is False
    assert res_conflict.output["conflict_detected"] is True
    assert res_conflict.output["severity"] == "HIGH"

    # Physiological vital bounds
    res_vitals = tool.execute({
        "assertion_type": "vital_bounds",
        "parameters": {"systolic_bp": 120, "diastolic_bp": 80},
    })
    assert res_vitals.output["passed"] is True

    res_bad_vitals = tool.execute({
        "assertion_type": "vital_bounds",
        "parameters": {"systolic_bp": 450, "diastolic_bp": 20},
    })
    assert res_bad_vitals.output["passed"] is False
    assert res_bad_vitals.output["bounds_violated"] is True


def test_pharmacology_knowledge_base_multi_class():
    """Verify multi-class cross-reactivity checks (Cephalosporins, NSAIDs, Opioids)."""
    tool = ClinicalAssertionCheckerTool()

    # Cephalosporin cross-reactivity with penicillin allergy
    res_ceph = tool.check_drug_contraindication(["penicillin allergy"], "cefazolin 1g IV")
    assert res_ceph["conflict_detected"] is True
    assert res_ceph["drug_class"] == "beta_lactam"

    # NSAID conflict
    res_nsaid = tool.check_drug_contraindication(["ibuprofen allergy"], "toradol 30mg")
    assert res_nsaid["conflict_detected"] is True
    assert res_nsaid["drug_class"] == "nsaid"

    # Safe drug
    res_safe = tool.check_drug_contraindication(["penicillin allergy"], "vancomycin 1g IV")
    assert res_safe["conflict_detected"] is False


def test_output_guardrail_contraindication_firewall():
    """Verify ClinicalOutputGuardrails intercepts unsafe outputs deterministically."""
    from core.guardrails import ClinicalOutputGuardrails

    guardrail = ClinicalOutputGuardrails(patient_allergies=["penicillin allergy"])
    unsafe_text = "The patient may receive standard beta-lactam antibiotics such as amoxicillin 500mg."
    sanitized = guardrail.validate_text_output(unsafe_text)

    assert guardrail.last_contraindication_blocked is True
    assert "STRICTLY CONTRAINDICATED" in sanitized
    assert "Kernel Assertion Gate" in sanitized
