"""Causal Reasoning Engine for Agentic Inferencing.

Integrates Pearlian Structural Causal Models into the agent's decision loop:
- Interventional Simulation (do(Action)) before tool execution
- Counterfactual Root-Cause Analysis after unexpected outcomes or failures
- Confounder and Backdoor Risk Identification
"""

from __future__ import annotations

from typing import Any
from core.causal_graph import CausalNodeType, StructuralCausalModel


class CausalReasoner:
    """Orchestrates causal interventions and counterfactual diagnostics for the Agent Kernel."""

    def __init__(self, scm: StructuralCausalModel | None = None) -> None:
        self.scm = scm or self.build_clinical_pharmacotherapy_scm()

    @staticmethod
    def build_clinical_pharmacotherapy_scm() -> StructuralCausalModel:
        """Construct standard benchmark SCM for medication, allergies, and organ function."""
        scm = StructuralCausalModel(name="clinical_pharmacotherapy")

        # Exogenous / Patient Baseline Variables
        scm.add_node("patient_age", "Patient age in years", CausalNodeType.EXOGENOUS, baseline_value=62)
        scm.add_node("baseline_gfr", "Baseline Glomerular Filtration Rate", CausalNodeType.EXOGENOUS, baseline_value=75.0)
        scm.add_node("penicillin_allergy", "History of beta-lactam hypersensitivity", CausalNodeType.EXOGENOUS, baseline_value=False)

        # Interventions (Agent Tool Decisions)
        scm.add_node("drug_prescription", "Administered medication", CausalNodeType.INTERVENTION, baseline_value="none")
        scm.add_node("dosage_mg", "Medication dosage in mg", CausalNodeType.INTERVENTION, baseline_value=0.0)

        # Endogenous Intermediate States
        scm.add_node("drug_class", "Pharmacological class of prescribed drug", CausalNodeType.ENDOGENOUS, baseline_value="none")
        scm.add_node("renal_clearance", "Active kidney filtration rate", CausalNodeType.ENDOGENOUS, baseline_value=75.0)

        # Outcomes
        scm.add_node("blood_pressure_reduction", "Reduction in systolic BP mmHg", CausalNodeType.OUTCOME, baseline_value=0.0)
        scm.add_node("anaphylaxis_reaction", "Severe allergic or adverse event", CausalNodeType.OUTCOME, baseline_value=False)
        scm.add_node("acute_kidney_injury", "Risk of renal injury", CausalNodeType.OUTCOME, baseline_value=False)

        # Causal Mechanisms (Edges)
        scm.add_edge("drug_prescription", "drug_class", "Drug name dictates chemical classification.")
        scm.add_edge("drug_class", "anaphylaxis_reaction", "Beta-lactams trigger allergy if hypersensitive.")
        scm.add_edge("penicillin_allergy", "anaphylaxis_reaction", "Sensitized immune system triggers response.")

        scm.add_edge("drug_prescription", "blood_pressure_reduction", "Antihypertensives lower blood pressure.")
        scm.add_edge("dosage_mg", "blood_pressure_reduction", "Dose-dependent clinical effect.")

        scm.add_edge("baseline_gfr", "renal_clearance", "Baseline kidney health sets clearance capacity.")
        scm.add_edge("dosage_mg", "acute_kidney_injury", "Excessive dose relative to clearance causes nephrotoxicity.")
        scm.add_edge("renal_clearance", "acute_kidney_injury", "Low clearance amplifies toxic build-up.")

        # Register Structural Equations
        def calc_drug_class(parents: dict[str, Any], curr: Any) -> str:
            drug = str(parents.get("drug_prescription", "")).lower()
            if any(w in drug for w in ["amoxicillin", "penicillin", "ampicillin"]):
                return "beta_lactam"
            if any(w in drug for w in ["ramipril", "lisinopril", "enalapril"]):
                return "ace_inhibitor"
            if any(w in drug for w in ["metformin"]):
                return "biguanide"
            return "other"

        def calc_anaphylaxis(parents: dict[str, Any], curr: Any) -> bool:
            drug_class = parents.get("drug_class")
            allergy = parents.get("penicillin_allergy", False)
            return bool(drug_class == "beta_lactam" and allergy)

        def calc_bp_reduction(parents: dict[str, Any], curr: Any) -> float:
            drug = str(parents.get("drug_prescription", "")).lower()
            dose = float(parents.get("dosage_mg", 0.0))
            if "ramipril" in drug:
                return min(25.0, dose * 2.5)
            return 0.0

        def calc_aki(parents: dict[str, Any], curr: Any) -> bool:
            dose = float(parents.get("dosage_mg", 0.0))
            clearance = float(parents.get("renal_clearance", 75.0))
            # Nephrotoxicity if high dose with compromised clearance
            return dose > 40.0 and clearance < 40.0

        scm.register_mechanism("drug_class", calc_drug_class)
        scm.register_mechanism("anaphylaxis_reaction", calc_anaphylaxis)
        scm.register_mechanism("blood_pressure_reduction", calc_bp_reduction)
        scm.register_mechanism("acute_kidney_injury", calc_aki)

        return scm

    def simulate_intervention_safety(
        self,
        proposed_action: dict[str, Any],
        patient_evidence: dict[str, Any],
    ) -> dict[str, Any]:
        """Pearl's Level 2: Simulate do(Action) and evaluate safety outcomes.

        Prevents agent execution if the mutilated causal graph predicts
        severe adverse reactions or contraindications.
        """
        # Set patient observations
        self.scm.set_evidence(patient_evidence)

        # Apply do(Action) intervention
        intervened_scm = self.scm.do_intervention(proposed_action)
        simulated_state = intervened_scm.forward_simulate()

        is_safe = True
        warnings: list[str] = []

        if simulated_state.get("anaphylaxis_reaction"):
            is_safe = False
            warnings.append("Intervention do(drug_prescription) causes ANAPHYLAXIS due to active drug allergy.")

        if simulated_state.get("acute_kidney_injury"):
            is_safe = False
            warnings.append("Intervention causes ACUTE KIDNEY INJURY due to excessive dose over impaired clearance.")

        return {
            "is_safe": is_safe,
            "simulated_outcomes": {
                "blood_pressure_reduction": simulated_state.get("blood_pressure_reduction", 0.0),
                "anaphylaxis_reaction": simulated_state.get("anaphylaxis_reaction", False),
                "acute_kidney_injury": simulated_state.get("acute_kidney_injury", False),
            },
            "warnings": warnings,
            "full_state": simulated_state,
        }

    def explain_counterfactual_attribution(
        self,
        actual_evidence: dict[str, Any],
        hypothetical_alternative: dict[str, Any],
        observed_bad_outcome: str,
    ) -> dict[str, Any]:
        """Pearl's Level 3: Counterfactual root-cause diagnostic.

        Answers: "Would the adverse outcome have been prevented if the agent had chosen the alternative action?"
        """
        analysis = self.scm.counterfactual_analysis(
            factual_evidence=actual_evidence,
            hypothetical_intervention=hypothetical_alternative,
            target_outcome=observed_bad_outcome,
        )

        factual_val = analysis["factual_outcome"]
        cf_val = analysis["counterfactual_outcome"]
        prevented = (factual_val is True and cf_val is False)

        attribution_explanation = (
            f"Counterfactual Proof: If the agent had executed {hypothetical_alternative}, "
            f"the outcome '{observed_bad_outcome}' would have changed from {factual_val} to {cf_val}. "
            f"Therefore, the factual action was a necessary cause of the outcome."
            if prevented
            else f"The alternative intervention {hypothetical_alternative} would NOT have changed outcome '{observed_bad_outcome}'."
        )

        return {
            "target_outcome": observed_bad_outcome,
            "outcome_prevented": prevented,
            "factual_outcome": factual_val,
            "counterfactual_outcome": cf_val,
            "explanation": attribution_explanation,
            "details": analysis,
        }
