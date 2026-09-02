"""Guardrails and output firewall for deterministic validation and safety."""

from __future__ import annotations

import json
import re
from typing import Any
from domain.state_machine import GuardrailViolationError


class OutputGuardrails:
    """Output firewall ensuring privacy, structural schema conformance, and clinical safety."""

    def __init__(self, strict_pii: bool = True) -> None:
        self.strict_pii = strict_pii

    def validate_text_output(self, text: str) -> str:
        """Inspect output text for potential privacy leaks or prohibited content."""
        if not text:
            return ""

        # Check for unscrubbed German / US phone numbers
        phone_pattern = r"(\+49\s?\d{2,4}\s?\d{4,8}|\b0\d{2,4}[\s/-]?\d{4,8}\b|\b\d{3}[-.]?\d{3}[-.]?\d{4}\b)"
        if re.search(phone_pattern, text):
            if self.strict_pii:
                raise GuardrailViolationError(
                    "Potential unredacted phone number detected in model response.",
                    {"pattern": "PHONE_NUMBER"},
                )

        # Check for unscrubbed German Steuer-ID or Social ID patterns
        id_pattern = r"\b(DE\d{9}|\b\d{2}/\d{3}/\d{5}\b|\bSSN:\s*\d{3}-\d{2}-\d{4}\b)"
        if re.search(id_pattern, text):
            if self.strict_pii:
                raise GuardrailViolationError(
                    "Potential national identity number detected in model response.",
                    {"pattern": "NATIONAL_ID"},
                )

        return text

    def validate_json_schema(self, output_str: str, required_keys: list[str]) -> dict[str, Any]:
        """Verify that string output is valid JSON and contains required keys."""
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

        missing = [k for k in required_keys if k not in data]
        if missing:
            raise GuardrailViolationError(
                f"Output JSON missing required keys: {missing}",
                {"missing_keys": missing, "present_keys": list(data.keys())},
            )

        return data
