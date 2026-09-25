"""Causal Pre-Flight Gate.

Intercepts agent tool executions that rely on unverified or falsified Structural Causal Models.
Runs observational verification against empirical tables before allowing intervention actions.
"""

from __future__ import annotations

from typing import Any
from core.causal_graph import StructuralCausalModel
from core.causal_verifier import CausalVerificationReport, ObservationalCausalVerifier


class CausalFalsificationViolationError(PermissionError):
    """Raised when an agent attempts an intervention on an empirically falsified causal mechanism."""
    pass


class CausalPreFlightGate:
    """Evaluates DAG observational consistency before allowing causal tool actions."""

    def __init__(self, alpha: float = 0.05, apply_fdr: bool = True) -> None:
        self.alpha = alpha
        self.apply_fdr = apply_fdr
        self.verifier = ObservationalCausalVerifier(alpha=alpha)

    def evaluate_scm_consistency(
        self,
        scm: StructuralCausalModel,
        observational_data: list[dict[str, Any]],
        intervention_var: str | None = None,
        outcome_var: str | None = None,
    ) -> CausalVerificationReport:
        """Verify empirical observational consistency of an SCM before allowing interventions."""
        report = self.verifier.verify_dataset(
            scm=scm,
            data=observational_data,
            check_direct_edges=True,
            apply_fdr=self.apply_fdr,
        )
        return report

    def pre_flight_check(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        scm: StructuralCausalModel | None,
        observational_data: list[dict[str, Any]] | None,
        intervention_var: str | None = None,
        outcome_var: str | None = None,
    ) -> tuple[bool, str | None, CausalVerificationReport | None]:
        """Perform pre-flight verification on a proposed tool execution.

        Returns:
            (is_admitted, failure_reason, verification_report)
        """
        if scm is None or observational_data is None:
            # If no causal model or observational baseline is bound, allow standard pass-through
            return True, None, None

        report = self.evaluate_scm_consistency(
            scm=scm,
            observational_data=observational_data,
            intervention_var=intervention_var,
            outcome_var=outcome_var,
        )

        if not report.is_valid:
            violations_summary = "; ".join(
                v.violation_detail for v in report.violations if v.violation_detail
            )
            reason = (
                f"Causal Pre-Flight Violation: Tool '{tool_name}' relies on falsified mechanism "
                f"in SCM '{scm.name}'. Violations: [{violations_summary}]."
            )
            return False, reason, report

        return True, None, report

    def enforce_pre_flight(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        scm: StructuralCausalModel | None,
        observational_data: list[dict[str, Any]] | None,
        intervention_var: str | None = None,
        outcome_var: str | None = None,
    ) -> CausalVerificationReport | None:
        """Enforce pre-flight verification, raising CausalFalsificationViolationError if falsified."""
        is_admitted, reason, report = self.pre_flight_check(
            tool_name=tool_name,
            arguments=arguments,
            scm=scm,
            observational_data=observational_data,
            intervention_var=intervention_var,
            outcome_var=outcome_var,
        )

        if not is_admitted:
            raise CausalFalsificationViolationError(reason)

        return report
