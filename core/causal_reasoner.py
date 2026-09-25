"""Causal Reasoning Engine for Agentic Inferencing.

Integrates Pearlian Structural Causal Models into the agent's decision loop:
- Interventional Simulation (do(Action)) before tool execution
- Counterfactual Root-Cause Analysis after unexpected outcomes or failures
- Confounder and Backdoor Risk Identification
"""

from __future__ import annotations

from typing import Any
from core.causal_graph import CausalNodeType, StructuralCausalModel
from core.causal_verifier import CausalVerificationReport, ObservationalCausalVerifier


def _check_warning(state: dict[str, Any], key: str, msg: str) -> str | None:
    return msg if state.get(key) else None


def _evaluate_safety_warnings(state: dict[str, Any]) -> tuple[bool, list[str]]:
    checks = [
        _check_warning(state, "anaphylaxis_reaction", "Intervention causes ANAPHYLAXIS due to active drug allergy."),
        _check_warning(state, "acute_kidney_injury", "Intervention causes ACUTE KIDNEY INJURY due to excessive dose over impaired clearance."),
        _check_warning(state, "hemorrhagic_stroke_risk", "Intervention causes FATAL HEMORRHAGE due to active bleeding contraindication."),
    ]
    warnings = [w for w in checks if w is not None]
    return (len(warnings) == 0, warnings)


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
        scm.add_node("active_bleeding", "Active intracranial or systemic hemorrhage", CausalNodeType.EXOGENOUS, baseline_value=False)

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
        scm.add_node("hemorrhagic_stroke_risk", "Fatal bleeding progression risk", CausalNodeType.OUTCOME, baseline_value=False)

        # Causal Mechanisms (Edges)
        scm.add_edge("drug_prescription", "drug_class", "Drug name dictates chemical classification.")
        scm.add_edge("drug_class", "anaphylaxis_reaction", "Beta-lactams trigger allergy if hypersensitive.")
        scm.add_edge("penicillin_allergy", "anaphylaxis_reaction", "Sensitized immune system triggers response.")

        scm.add_edge("drug_prescription", "blood_pressure_reduction", "Antihypertensives lower blood pressure.")
        scm.add_edge("dosage_mg", "blood_pressure_reduction", "Dose-dependent clinical effect.")

        scm.add_edge("baseline_gfr", "renal_clearance", "Baseline kidney health sets clearance capacity.")
        scm.add_edge("dosage_mg", "acute_kidney_injury", "Excessive dose relative to clearance causes nephrotoxicity.")
        scm.add_edge("renal_clearance", "acute_kidney_injury", "Low clearance amplifies toxic build-up.")

        scm.add_edge("drug_class", "hemorrhagic_stroke_risk", "Anticoagulants amplify active hemorrhage.")
        scm.add_edge("active_bleeding", "hemorrhagic_stroke_risk", "Active bleeding exacerbated by anticoagulants.")

        # Register Structural Equations
        def calc_drug_class(parents: dict[str, Any], curr: Any) -> str:
            drug = str(parents.get("drug_prescription", "")).lower()
            if any(w in drug for w in ["amoxicillin", "penicillin", "ampicillin"]):
                return "beta_lactam"
            if any(w in drug for w in ["ramipril", "lisinopril", "enalapril"]):
                return "ace_inhibitor"
            if any(w in drug for w in ["metformin"]):
                return "biguanide"
            if any(w in drug for w in ["warfarin", "heparin", "apixaban"]):
                return "anticoagulant"
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

        def calc_clearance(parents: dict[str, Any], curr: Any) -> float:
            gfr = parents.get("baseline_gfr")
            if gfr is not None:
                return float(gfr)
            return float(curr or 75.0)

        def calc_aki(parents: dict[str, Any], curr: Any) -> bool:
            dose = float(parents.get("dosage_mg", 0.0))
            clearance = float(parents.get("renal_clearance", 75.0))
            return dose > 40.0 and clearance < 40.0

        def calc_hemorrhage(parents: dict[str, Any], curr: Any) -> bool:
            drug_class = parents.get("drug_class")
            bleeding = parents.get("active_bleeding", False)
            return bool(drug_class == "anticoagulant" and bleeding)

        scm.register_mechanism("drug_class", calc_drug_class)
        scm.register_mechanism("renal_clearance", calc_clearance)
        scm.register_mechanism("anaphylaxis_reaction", calc_anaphylaxis)
        scm.register_mechanism("blood_pressure_reduction", calc_bp_reduction)
        scm.register_mechanism("acute_kidney_injury", calc_aki)
        scm.register_mechanism("hemorrhagic_stroke_risk", calc_hemorrhage)

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
        self.scm.set_evidence(patient_evidence)
        intervened_scm = self.scm.do_intervention(proposed_action)
        simulated_state = intervened_scm.forward_simulate()
        is_safe, warnings = _evaluate_safety_warnings(simulated_state)

        return {
            "is_safe": is_safe,
            "simulated_outcomes": {
                "blood_pressure_reduction": simulated_state.get("blood_pressure_reduction", 0.0),
                "anaphylaxis_reaction": simulated_state.get("anaphylaxis_reaction", False),
                "acute_kidney_injury": simulated_state.get("acute_kidney_injury", False),
                "hemorrhagic_stroke_risk": simulated_state.get("hemorrhagic_stroke_risk", False),
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

    def load_scm_from_json(self, json_str_or_path: str) -> StructuralCausalModel:
        """Load and set active SCM from JSON string or file path."""
        self.scm = StructuralCausalModel.from_json(json_str_or_path)
        return self.scm

    def load_scm_from_graphml(self, xml_str_or_path: str) -> StructuralCausalModel:
        """Load and set active SCM from GraphML XML string or file path."""
        self.scm = StructuralCausalModel.from_graphml(xml_str_or_path)
        return self.scm

    def export_scm_to_json(self, filepath: str | None = None) -> str:
        """Export current active SCM to JSON string or file."""
        return self.scm.to_json(filepath=filepath)

    def export_scm_to_graphml(self, filepath: str | None = None) -> str:
        """Export current active SCM to GraphML XML string or file."""
        return self.scm.to_graphml(filepath=filepath)

    def verify_observational_data(
        self,
        data: list[dict[str, Any]],
        alpha: float = 0.05,
        custom_tests: list[dict[str, Any]] | None = None,
        check_direct_edges: bool = True,
    ) -> CausalVerificationReport:
        """Verify observational dataset against current active SCM's d-separation structure."""
        verifier = ObservationalCausalVerifier(alpha=alpha)
        return verifier.verify_dataset(
            scm=self.scm,
            data=data,
            custom_tests=custom_tests,
            check_direct_edges=check_direct_edges,
        )
