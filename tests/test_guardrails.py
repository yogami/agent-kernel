import pytest
import json
from core.guardrails import OutputGuardrails
from domain.state_machine import GuardrailViolationError

def test_validate_text_output_empty():
    guardrails = OutputGuardrails()
    assert guardrails.validate_text_output("") == ""
    assert guardrails.validate_text_output(None) == ""

def test_validate_text_output_phone_violation():
    guardrails = OutputGuardrails()
    with pytest.raises(GuardrailViolationError, match="phone number"):
        guardrails.validate_text_output("My number is 123-456-7890")
    with pytest.raises(GuardrailViolationError, match="phone number"):
        guardrails.validate_text_output("Call me at 0151/1234567")

def test_validate_text_output_id_violation():
    guardrails = OutputGuardrails()
    with pytest.raises(GuardrailViolationError, match="national identity"):
        guardrails.validate_text_output("Tax ID: 12/345/67890")
    with pytest.raises(GuardrailViolationError, match="national identity"):
        guardrails.validate_text_output("My SSN: 123-45-6789")

def test_validate_text_output_patient_id_violation():
    guardrails = OutputGuardrails()
    with pytest.raises(GuardrailViolationError, match="patient identifier"):
        guardrails.validate_text_output("Patient PAT-2004 summary notes")
    with pytest.raises(GuardrailViolationError, match="patient identifier"):
        guardrails.validate_text_output("Record for MRN: 987654")


def test_validate_json_schema_success():
    guardrails = OutputGuardrails()
    valid_json = '{"name": "test", "value": 1}'
    res = guardrails.validate_json_schema(valid_json, ["name", "value"])
    assert res == {"name": "test", "value": 1}

def test_validate_json_schema_decode_error():
    guardrails = OutputGuardrails()
    with pytest.raises(GuardrailViolationError, match="JSON parsing"):
        guardrails.validate_json_schema("{invalid json", ["key"])

def test_validate_json_schema_type_error():
    guardrails = OutputGuardrails()
    with pytest.raises(GuardrailViolationError, match="dictionary object"):
        guardrails.validate_json_schema("[1, 2, 3]", ["key"])

def test_validate_json_schema_missing_keys():
    guardrails = OutputGuardrails()
    with pytest.raises(GuardrailViolationError, match="missing required keys"):
        guardrails.validate_json_schema('{"name": "test"}', ["name", "value"])
