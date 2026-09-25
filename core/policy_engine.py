"""Declarative Policy-as-Code and Knowledge Graph Assertion Engine.

Decouples deterministic constraint validation, taxonomy hierarchies, and safety firewalls
from the core Agent Kernel execution loop. Domain-specific rules (pharmacology, fintech,
aerospace, supply chain) are loaded from external declarative policy files.
"""

from __future__ import annotations

from enum import Enum
import json
from pathlib import Path
import re
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field


class PolicyVerdict(str, Enum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    QUARANTINE = "QUARANTINE"


class PolicyEvaluationResult(BaseModel):
    passed: bool = Field(..., description="Whether the evaluation satisfied all active policy constraints.")
    verdict: PolicyVerdict = Field(..., description="Actionable policy verdict: ALLOW, DENY, or QUARANTINE.")
    rule_id: str = Field(..., description="Identifier of the specific policy rule evaluated or triggered.")
    domain: str = Field(..., description="Policy domain, e.g. 'healthcare_pharmacology' or 'fintech_capital_markets'.")
    severity: str = Field("NONE", description="Severity level: NONE, LOW, MEDIUM, HIGH, CRITICAL.")
    category: str | None = Field(None, description="Category or taxonomy class matched.")
    matched_constraints: list[str] = Field(default_factory=list, description="Specific terms or constraints matched.")
    explanation: str = Field(..., description="Clear human-readable explanation of the policy decision.")
    remediation: str | None = Field(None, description="Recommended remediation action if denied.")


@runtime_checkable
class PolicyEnginePort(Protocol):
    """Protocol interface for domain-agnostic declarative policy assertion engines."""

    def evaluate_intent(
        self, intent_type: str, candidate: dict[str, Any], context: dict[str, Any]
    ) -> PolicyEvaluationResult:
        """Evaluate a structured candidate intent against active domain policies."""
        ...

    def evaluate_free_text(
        self, text: str, context: dict[str, Any]
    ) -> list[PolicyEvaluationResult]:
        """Scan unstructured text output against active domain policies with negation awareness."""
        ...

    def load_policy(self, policy_source: str | Path | dict[str, Any]) -> None:
        """Load or hot-reload an external declarative policy document."""
        ...


class KnowledgeGraphPolicyEngine:
    """Graph-backed declarative policy engine supporting multi-domain constraint checking."""

    def __init__(self, policy_source: str | Path | dict[str, Any] | None = None) -> None:
        self.domain: str = "unspecified"
        self.version: str = "1.0.0"
        self.raw_policy: dict[str, Any] = {}
        self.taxonomy_trees: dict[str, Any] = {}
        self.rules: list[dict[str, Any]] = []
        self.quantitative_bounds: dict[str, Any] = {}
        self.negation_keywords: list[str] = [
            "contraindicated", "avoid", "do not administer", "do not give", "allergic",
            "allergy to", "not recommend", "unsafe", "withhold", "prohibit", "withheld", "prohibited"
        ]
        self.interaction_rules: list[dict[str, Any]] = []
        self.renal_rules: list[dict[str, Any]] = []
        self.hepatic_rules: list[dict[str, Any]] = []

        # Auto-load standard policies from repository
        policy_dir = Path(__file__).resolve().parent.parent / "policies"
        default_files = [
            policy_dir / "clinical_contraindications.json",
            policy_dir / "drug_interactions.json",
            policy_dir / "renal_dosing.json",
        ]
        for f in default_files:
            if f.exists():
                self.load_policy(f)

        if policy_source is not None:
            self.load_policy(policy_source)

    def load_policy(self, policy_source: str | Path | dict[str, Any]) -> None:
        """Load policy specification from file path, JSON string, or dictionary."""
        if isinstance(policy_source, (str, Path)):
            path = Path(policy_source)
            if path.is_file():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = json.loads(str(policy_source))
        elif isinstance(policy_source, dict):
            data = policy_source
        else:
            raise ValueError(f"Unsupported policy source type: {type(policy_source)}")

        self.raw_policy.update(data)
        if self.domain == "unspecified" or data.get("domain") != "generic":
            self.domain = data.get("domain", self.domain)
        self.version = data.get("version", self.version)

        if "taxonomy_trees" in data:
            self.taxonomy_trees.update(data["taxonomy_trees"])
        if "contraindication_rules" in data:
            self.rules.extend(data["contraindication_rules"])
        elif "compliance_rules" in data:
            self.rules.extend(data["compliance_rules"])
        if "quantitative_bounds" in data:
            self.quantitative_bounds.update(data["quantitative_bounds"])
        if "interaction_rules" in data:
            self.interaction_rules.extend(data["interaction_rules"])
        if "renal_rules" in data:
            self.renal_rules.extend(data["renal_rules"])
        if "hepatic_rules" in data:
            self.hepatic_rules.extend(data["hepatic_rules"])
        if "negation_keywords" in data:
            for kw in data["negation_keywords"]:
                if kw not in self.negation_keywords:
                    self.negation_keywords.append(kw)

    def evaluate_intent(
        self, intent_type: str, candidate: dict[str, Any], context: dict[str, Any]
    ) -> PolicyEvaluationResult:
        """Evaluate a structured candidate action against the knowledge graph."""
        # 1. Clinical Drug-Allergy Intent Check
        if intent_type in ("drug_allergy_conflict", "medication_administration"):
            allergies = [a.lower() for a in candidate.get("patient_allergies", context.get("patient_allergies", []))]
            drug = candidate.get("prescribed_drug", "").lower()

            for class_name, tree in self.taxonomy_trees.items():
                # Check allergen match in taxonomy
                matched_allergens = [
                    al for al in allergies
                    if any(kw in al for kw in tree.get("allergens", []))
                ]
                if not matched_allergens:
                    continue

                # Check drug match in taxonomy
                matched_drugs = [
                    m for m in tree.get("members", [])
                    if m in drug
                ]
                if matched_drugs:
                    matching_rule = next((r for r in self.rules if r.get("class") == class_name), {})
                    return PolicyEvaluationResult(
                        passed=False,
                        verdict=PolicyVerdict.DENY,
                        rule_id=matching_rule.get("rule_id", f"RULE-{class_name.upper()}"),
                        domain=self.domain,
                        severity=matching_rule.get("severity", "CRITICAL"),
                        category=class_name,
                        matched_constraints=[f"Allergy: {matched_allergens}", f"Proposed: {matched_drugs}"],
                        explanation=(
                            f"Contraindication Policy Violation: Patient has documented allergy '{matched_allergens}' "
                            f"and proposed drug '{matched_drugs}' belongs to cross-reactive class '{class_name}'."
                        ),
                        remediation=matching_rule.get("remediation", "Withhold medication and select alternative."),
                    )

            return PolicyEvaluationResult(
                passed=True,
                verdict=PolicyVerdict.ALLOW,
                rule_id="RULE-ALLOW-DEFAULT",
                domain=self.domain,
                severity="NONE",
                explanation="No active policy contraindication found in knowledge graph.",
            )

        # 2. Quantitative Bounds (e.g. Vitals or Financial Caps)
        if intent_type in ("vital_bounds", "physiological_vitals"):
            errors = []
            for metric, val in candidate.items():
                bounds = self.quantitative_bounds.get(metric)
                if bounds and isinstance(val, (int, float)):
                    min_v = bounds.get("min")
                    max_v = bounds.get("max")
                    if min_v is not None and val < min_v:
                        errors.append(f"{metric} value {val} below minimum bound ({min_v} {bounds.get('unit', '')})")
                    elif max_v is not None and val > max_v:
                        errors.append(f"{metric} value {val} above maximum bound ({max_v} {bounds.get('unit', '')})")

            if errors:
                return PolicyEvaluationResult(
                    passed=False,
                    verdict=PolicyVerdict.DENY,
                    rule_id="RULE-QUANT-BOUNDS",
                    domain=self.domain,
                    severity="HIGH",
                    matched_constraints=errors,
                    explanation=f"Quantitative bounds violation: {'; '.join(errors)}",
                    remediation="Re-verify measurements or trigger critical clinical alert.",
                )
            return PolicyEvaluationResult(
                passed=True,
                verdict=PolicyVerdict.ALLOW,
                rule_id="RULE-QUANT-OK",
                domain=self.domain,
                severity="NONE",
                explanation="All quantitative parameters within allowed bounds.",
            )

        # 3. Drug-Drug Interaction Check (CYP450 / P-gp)
        if intent_type in ("drug_drug_interaction", "ddi_check"):
            current_meds = [m.lower() for m in candidate.get("current_medications", context.get("current_medications", []))]
            proposed_drug = candidate.get("proposed_drug", "").lower()

            for rule in self.interaction_rules:
                perps = [p.lower() for p in rule.get("perpetrators", [])]
                substrates = [s.lower() for s in rule.get("substrates", [])]

                perp_in_current = [p for p in perps if any(p in med for med in current_meds)]
                sub_in_proposed = [s for s in substrates if s in proposed_drug]

                perp_in_proposed = [p for p in perps if p in proposed_drug]
                sub_in_current = [s for s in substrates if any(s in med for med in current_meds)]

                if (perp_in_current and sub_in_proposed) or (perp_in_proposed and sub_in_current):
                    active_perp = perp_in_current or perp_in_proposed
                    active_sub = sub_in_proposed or sub_in_current
                    mech = rule.get("mechanism", "CYP450_interaction")
                    return PolicyEvaluationResult(
                        passed=False,
                        verdict=PolicyVerdict.DENY,
                        rule_id=rule.get("rule_id", "RULE-DDI-DENY"),
                        domain="healthcare_drug_interactions",
                        severity=rule.get("severity", "CRITICAL"),
                        category=mech,
                        matched_constraints=[f"Perpetrator: {active_perp}", f"Substrate: {active_sub}", f"Mechanism: {mech}"],
                        explanation=f"Severe Drug-Drug Interaction ({mech}): {rule.get('explanation', '')}",
                        remediation=rule.get("remediation", "Withhold interacting agent and select alternative."),
                    )

            return PolicyEvaluationResult(
                passed=True,
                verdict=PolicyVerdict.ALLOW,
                rule_id="RULE-DDI-OK",
                domain="healthcare_drug_interactions",
                severity="NONE",
                explanation="No documented severe drug-drug interactions detected.",
            )

        # 4. Renal & Hepatic Dosing Boundary Check
        if intent_type in ("renal_dosing_check", "renal_clearance", "hepatic_dosing_check"):
            age = candidate.get("age", 60)
            weight_kg = candidate.get("weight_kg", 70.0)
            scr = candidate.get("serum_creatinine", 1.0)
            sex = str(candidate.get("sex", "male")).lower()
            proposed_drug = str(candidate.get("proposed_drug", "")).lower()
            proposed_dose = str(candidate.get("proposed_dose", "")).lower()

            crcl = 100.0
            if scr and float(scr) > 0:
                crcl = ((140.0 - float(age)) * float(weight_kg)) / (72.0 * float(scr))
                if "f" in sex:
                    crcl *= 0.85

            for rule in self.renal_rules:
                target_drug = rule.get("drug", "").lower()
                if target_drug in proposed_drug:
                    thresh = rule.get("crcl_threshold", 30.0)
                    if crcl < thresh:
                        reduction_dose = rule.get("dose_reduction_dose", "").lower()
                        if reduction_dose and reduction_dose in proposed_dose:
                            return PolicyEvaluationResult(
                                passed=True,
                                verdict=PolicyVerdict.ALLOW,
                                rule_id=f"{rule.get('rule_id')}-ADJUSTED-OK",
                                domain="healthcare_renal_dosing",
                                severity="NONE",
                                explanation=f"Dose is appropriately adjusted ({proposed_dose}) for renal impairment (CrCl {crcl:.1f} mL/min < {thresh} mL/min).",
                            )

                        return PolicyEvaluationResult(
                            passed=False,
                            verdict=PolicyVerdict.DENY,
                            rule_id=rule.get("rule_id", "RULE-RENAL-DENY"),
                            domain="healthcare_renal_dosing",
                            severity=rule.get("severity", "HIGH"),
                            category="renal_dosing_contraindication",
                            matched_constraints=[f"Calculated CrCl: {crcl:.1f} mL/min < Threshold: {thresh} mL/min", f"Ordered Dose: {proposed_dose}"],
                            explanation=f"Renal Dosing Policy Violation: {rule.get('explanation', '')} (Calculated CrCl: {crcl:.1f} mL/min).",
                            remediation=rule.get("remediation", "Adjust dose according to renal protocol."),
                        )

            # Hepatic checks
            if intent_type == "hepatic_dosing_check" or "acetaminophen" in proposed_drug:
                for rule in self.hepatic_rules:
                    target_drug = rule.get("drug", "").lower()
                    if target_drug in proposed_drug:
                        max_cirrhosis = rule.get("max_daily_dose_mg_cirrhosis", 2000)
                        if "1000" in proposed_dose and ("q6h" in proposed_dose or "4000" in proposed_dose or "prn" in proposed_dose):
                            return PolicyEvaluationResult(
                                passed=False,
                                verdict=PolicyVerdict.DENY,
                                rule_id=rule.get("rule_id", "RULE-HEPATIC-DENY"),
                                domain="healthcare_renal_dosing",
                                severity=rule.get("severity", "HIGH"),
                                category="hepatic_dosing_contraindication",
                                matched_constraints=[f"Proposed daily dose ~4000mg > Max allowed in severe cirrhosis: {max_cirrhosis}mg"],
                                explanation=f"Hepatic Dosing Policy Violation: {rule.get('explanation', '')}",
                                remediation=rule.get("remediation", "Limit daily acetaminophen to maximum 2000 mg/day."),
                            )

            return PolicyEvaluationResult(
                passed=True,
                verdict=PolicyVerdict.ALLOW,
                rule_id="RULE-RENAL-OK",
                domain="healthcare_renal_dosing",
                severity="NONE",
                explanation=f"Dosing parameters comply with renal clearance bounds (CrCl {crcl:.1f} mL/min).",
            )

        # 5. FinTech Capital Markets Compliance Check
        if self.domain == "fintech_capital_markets" or intent_type == "trade_order":
            ticker = candidate.get("ticker", "").upper()
            notional = candidate.get("notional_usd", 0.0)

            blacklisted = self.raw_policy.get("blacklisted_tickers", [])
            if ticker in blacklisted:
                return PolicyEvaluationResult(
                    passed=False,
                    verdict=PolicyVerdict.DENY,
                    rule_id="RULE-FIN-001",
                    domain=self.domain,
                    severity="CRITICAL",
                    matched_constraints=[f"Ticker: {ticker}"],
                    explanation=f"Compliance Violation: Security '{ticker}' is on the restricted trading blacklist.",
                    remediation="Immediately cancel order and notify legal compliance.",
                )

            max_notional = self.raw_policy.get("single_order_max_notional_usd", 1000000)
            if notional > max_notional:
                return PolicyEvaluationResult(
                    passed=False,
                    verdict=PolicyVerdict.QUARANTINE,
                    rule_id="RULE-FIN-002",
                    domain=self.domain,
                    severity="HIGH",
                    matched_constraints=[f"Notional: ${notional:,.2f} > Max: ${max_notional:,.2f}"],
                    explanation=f"Risk Budget Violation: Order value ${notional:,.2f} exceeds single-order limit of ${max_notional:,.2f}.",
                    remediation="Route order to Human-in-the-Loop desk for supervisor sign-off.",
                )

            return PolicyEvaluationResult(
                passed=True,
                verdict=PolicyVerdict.ALLOW,
                rule_id="RULE-FIN-OK",
                domain=self.domain,
                severity="NONE",
                explanation="Trade order complies with autonomous execution policies.",
            )

        return PolicyEvaluationResult(
            passed=True,
            verdict=PolicyVerdict.ALLOW,
            rule_id="RULE-PASSTHROUGH",
            domain=self.domain,
            severity="NONE",
            explanation=f"No policy defined for intent type '{intent_type}'. Defaulting to passthrough.",
        )

    def evaluate_free_text(
        self, text: str, context: dict[str, Any]
    ) -> list[PolicyEvaluationResult]:
        """Scan unstructured text output against active knowledge graph entities with negation awareness."""
        if not text:
            return []

        violations: list[PolicyEvaluationResult] = []
        text_lower = text.lower()
        allergies = [a.lower() for a in context.get("patient_allergies", [])]

        if not allergies:
            return []

        # Build negation pattern from declarative keywords
        neg_terms = "|".join(re.escape(k) for k in self.negation_keywords)

        for class_name, tree in self.taxonomy_trees.items():
            matched_allergens = [
                al for al in allergies
                if any(kw in al for kw in tree.get("allergens", []))
            ]
            if not matched_allergens:
                continue

            for drug in tree.get("members", []):
                pattern = rf"\b{re.escape(drug)}\b"
                if re.search(pattern, text_lower):
                    # Check if negated/blocked in text
                    neg_window = rf"({neg_terms}).*?\b{re.escape(drug)}\b|\b{re.escape(drug)}\b.*?({neg_terms})"
                    if not re.search(neg_window, text_lower):
                        matching_rule = next((r for r in self.rules if r.get("class") == class_name), {})
                        violations.append(
                            PolicyEvaluationResult(
                                passed=False,
                                verdict=PolicyVerdict.DENY,
                                rule_id=matching_rule.get("rule_id", f"RULE-TEXT-{class_name.upper()}"),
                                domain=self.domain,
                                severity=matching_rule.get("severity", "CRITICAL"),
                                category=class_name,
                                matched_constraints=[f"Allergen: {matched_allergens}", f"Unblocked Drug: {drug}"],
                                explanation=(
                                    f"Administration of '{drug}' is STRICTLY CONTRAINDICATED due to documented patient allergy to '{matched_allergens}' "
                                    f"({class_name} cross-reactivity)."
                                ),
                                remediation=matching_rule.get("remediation", "Medication withheld by Kernel Assertion Gate."),
                            )
                        )

        return violations
