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
