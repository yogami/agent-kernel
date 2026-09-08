"""Causal reasoning tools for interventional simulation and counterfactual attribution."""

from __future__ import annotations

from typing import Any
from core.causal_reasoner import CausalReasoner
from domain.models import ToolResult
from domain.ports import ToolPort


class SimulateCausalInterventionTool(ToolPort):
    """Pearl's Level 2: Simulates the causal downstream effect of an action do(X)."""

    name: str = "simulate_causal_intervention"
    description: str = (
        "Simulates setting variables directly via Pearl's do-operator on a mutilated causal DAG, "
        "checking downstream outcomes and adverse events before an action is executed."
    )

    def __init__(self, reasoner: CausalReasoner | None = None) -> None:
        self.reasoner = reasoner or CausalReasoner()

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "intervention": {
                    "type": "object",
                    "description": "Variables to set via do(X), e.g. {'drug_prescription': 'amoxicillin', 'dosage_mg': 500}.",
                },
                "patient_context": {
                    "type": "object",
                    "description": "Baseline observed patient variables, e.g. {'penicillin_allergy': true, 'baseline_gfr': 55.0}.",
                },
            },
            "required": ["intervention"],
        }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        intervention = arguments.get("intervention", {})
        context = arguments.get("patient_context", {})

        if not intervention:
            return ToolResult(
                tool_id="sim_causal_err",
                tool_name=self.name,
                output={"error": "Empty intervention dictionary provided."},
                is_error=True,
                error_message="Empty intervention.",
            )

        sim_res = self.reasoner.simulate_intervention_safety(
            proposed_action=intervention,
            patient_evidence=context,
        )

        return ToolResult(
            tool_id="sim_causal_ok",
            tool_name=self.name,
            output={
                "is_safe": sim_res["is_safe"],
                "simulated_outcomes": sim_res["simulated_outcomes"],
                "warnings": sim_res["warnings"],
                "intervention_evaluated": intervention,
            },
        )


class ExplainCounterfactualAttributionTool(ToolPort):
    """Pearl's Level 3: Evaluates counterfactual 'what-if' queries to attribute failure causes."""

    name: str = "explain_counterfactual_attribution"
    description: str = (
        "Answers what-if counterfactual questions using the 3-step Pearlian algorithm (Abduction, "
        "Action, Prediction) to determine whether an alternative action would have prevented an adverse outcome."
    )

    def __init__(self, reasoner: CausalReasoner | None = None) -> None:
        self.reasoner = reasoner or CausalReasoner()

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "factual_evidence": {
                    "type": "object",
                    "description": "What actually occurred in the factual world (actions + baseline conditions).",
                },
                "hypothetical_action": {
                    "type": "object",
                    "description": "The alternative action that could have been taken, e.g. {'drug_prescription': 'ramipril'}.",
                },
                "adverse_outcome_target": {
                    "type": "string",
                    "description": "Target outcome to analyze, e.g. 'anaphylaxis_reaction' or 'acute_kidney_injury'.",
                },
            },
            "required": ["factual_evidence", "hypothetical_action", "adverse_outcome_target"],
        }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        factual = arguments.get("factual_evidence", {})
        hypothetical = arguments.get("hypothetical_action", {})
        target = arguments.get("adverse_outcome_target", "")

        if not target:
            return ToolResult(
                tool_id="cf_eval_err",
                tool_name=self.name,
                output={"error": "Missing adverse_outcome_target parameter."},
                is_error=True,
                error_message="Missing target.",
            )

        attribution = self.reasoner.explain_counterfactual_attribution(
            actual_evidence=factual,
            hypothetical_alternative=hypothetical,
            observed_bad_outcome=target,
        )

        return ToolResult(
            tool_id="cf_eval_ok",
            tool_name=self.name,
            output={
                "target_outcome": target,
                "outcome_prevented_by_alternative": attribution["outcome_prevented"],
                "factual_outcome": attribution["factual_outcome"],
                "counterfactual_outcome": attribution["counterfactual_outcome"],
                "causal_explanation": attribution["explanation"],
            },
        )


class CausalGraphQueryTool(ToolPort):
    """Inspects causal pathways, backdoor confounding paths, and topological dependencies."""

    name: str = "query_causal_graph"
    description: str = (
        "Inspects causal relationships between domain variables, identifying direct mechanisms "
        "and potential backdoor confounding paths."
    )

    def __init__(self, reasoner: CausalReasoner | None = None) -> None:
        self.reasoner = reasoner or CausalReasoner()

    @property
    def schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "treatment_node": {"type": "string", "description": "Action or treatment variable (X)."},
                "outcome_node": {"type": "string", "description": "Outcome variable (Y)."},
            },
            "required": ["treatment_node", "outcome_node"],
        }

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        treatment = arguments.get("treatment_node", "")
        outcome = arguments.get("outcome_node", "")

        scm = self.reasoner.scm
        if treatment not in scm.nodes or outcome not in scm.nodes:
            return ToolResult(
                tool_id="causal_query_err",
                tool_name=self.name,
                output={"error": f"One or both nodes '{treatment}', '{outcome}' not found in SCM."},
                is_error=True,
                error_message="Nodes not found.",
            )

        backdoor = scm.find_backdoor_paths(treatment, outcome)
        direct_edges = [
            {"target": e.target, "mechanism": e.mechanism_description}
            for e in scm.edges.get(treatment, [])
        ]

        return ToolResult(
            tool_id="causal_query_ok",
            tool_name=self.name,
            output={
                "treatment": treatment,
                "outcome": outcome,
                "direct_mechanisms": direct_edges,
                "backdoor_confounding_paths": backdoor,
                "has_confounding": len(backdoor) > 0,
            },
        )
