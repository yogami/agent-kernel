"""Guardrails and output firewall for deterministic validation and safety."""

from __future__ import annotations

import json
import re
from typing import Any
from domain.state_machine import GuardrailViolationError


class OutputGuardrails:
    """Output firewall ensuring privacy, structural schema conformance, and clinical safety."""

    def __init__(self, strict_pii: bool = True, deidentify_clinical: bool = False) -> None:
        self.strict_pii = strict_pii
        self.deidentify_clinical = deidentify_clinical

    def _check_phone_leak(self, text: str) -> None:
        pattern = r"(\+49\s?\d{2,4}\s?\d{4,8}|\b0\d{2,4}[\s/-]?\d{4,8}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b)"
        if self.strict_pii and re.search(pattern, text):
            raise GuardrailViolationError(
                "Potential unredacted phone number detected in model response.",
                {"pattern": "PHONE_NUMBER"},
            )

    def _check_national_id_leak(self, text: str) -> None:
        pattern = r"\b(DE\d{9}|\b\d{2}/\d{3}/\d{5}\b|\bSSN:\s*\d{3}-\d{2}-\d{4}\b)"
        if self.strict_pii and re.search(pattern, text):
            raise GuardrailViolationError(
                "Potential national identity number detected in model response.",
                {"pattern": "NATIONAL_ID"},
            )

    def _check_patient_id_leak(self, text: str) -> None:
        pattern = r"\b(PAT-[\w\d-]+|PID[:\s]+[\w\d-]+|MRN[:\s]*[\w\d-]+)\b"
        if self.strict_pii and re.search(pattern, text, flags=re.IGNORECASE):
            raise GuardrailViolationError(
                "Potential patient identifier detected in model response.",
                {"pattern": "PATIENT_ID"},
            )

    def _deidentify_if_enabled(self, text: str) -> str:
        if not self.deidentify_clinical:
            return text
        from tools.clinical_tools import DeidentifyTextTool
        scrubbed, _ = DeidentifyTextTool.scrub(text)
        return scrubbed

    def validate_text_output(self, text: str) -> str:
        """Inspect output text for potential privacy leaks or prohibited content, sanitizing if enabled."""
        if not text:
            return ""

        text = self._deidentify_if_enabled(text)
        self._check_phone_leak(text)
        self._check_national_id_leak(text)
        self._check_patient_id_leak(text)
        return text



    def _parse_json_dict(self, output_str: str) -> dict[str, Any]:
        try:
            data = json.loads(output_str)
        except json.JSONDecodeError as exc:
            raise GuardrailViolationError(
                f"Model response failed JSON parsing: {exc}",
                {"raw_output": output_str},
            )
        if not isinstance(data, dict):
            raise GuardrailViolationError(
                "JSON root element must be a dictionary object.",
                {"parsed_type": type(data).__name__},
            )
        return data

    def _verify_required_keys(self, data: dict[str, Any], required_keys: list[str]) -> None:
        missing = [k for k in required_keys if k not in data]
        if missing:
            raise GuardrailViolationError(
                f"Output JSON missing required keys: {missing}",
                {"missing_keys": missing, "present_keys": list(data.keys())},
            )

    def validate_json_schema(self, output_str: str, required_keys: list[str]) -> dict[str, Any]:
        """Verify that string output is valid JSON and contains required keys."""
        data = self._parse_json_dict(output_str)
        self._verify_required_keys(data, required_keys)
        return data



class ClinicalOutputGuardrails(OutputGuardrails):
    """Output firewall pre-configured for clinical environments with hybrid statistical NER de-identification
    and deterministic pharmacological contraindication interception."""

    def __init__(
        self,
        strict_pii: bool = False,
        patient_allergies: list[str] | None = None,
        policy_engine: Any | None = None,
    ) -> None:
        super().__init__(strict_pii=strict_pii, deidentify_clinical=True)
        self.patient_allergies: list[str] = list(patient_allergies) if patient_allergies else []
        self.last_contraindication_blocked: bool = False
        if policy_engine is None:
            from core.policy_engine import KnowledgeGraphPolicyEngine
            self.policy_engine = KnowledgeGraphPolicyEngine()
        else:
            self.policy_engine = policy_engine

    def set_patient_allergies(self, allergies: list[str]) -> None:
        self.patient_allergies = list(allergies)

    def _format_violation_notices(self, violations: list[Any]) -> str:
        notices = [
            f"[SAFETY INTERCEPTION]: {v.explanation} {v.remediation or ''} Medication withheld by Kernel Assertion Gate."
            for v in violations
        ]
        return "\n\n" + "\n".join(notices)

    def _intercept_contraindications(self, text: str) -> str:
        if not self.patient_allergies or not text:
            return text
        violations = self.policy_engine.evaluate_free_text(
            text, {"patient_allergies": self.patient_allergies}
        )
        if not violations:
            return text
        self.last_contraindication_blocked = True
        return text + self._format_violation_notices(violations)

    def validate_text_output(self, text: str) -> str:
        text = super().validate_text_output(text)
        self.last_contraindication_blocked = False
        return self._intercept_contraindications(text)

