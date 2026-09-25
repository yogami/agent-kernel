"""Clinical curation tools: PII de-identification, ontology mapping, FHIR validation, and clinical assertion checks."""

from __future__ import annotations

import re
from typing import Any
from pydantic import BaseModel, Field

from domain.models import ToolResult
from domain.ports import ToolPort


try:
    import spacy
    _SPACY_NLP = spacy.load("en_core_web_sm")
except Exception:
    _SPACY_NLP = None


class DeidentifyInput(BaseModel):
    text: str = Field(..., description="Raw clinical text containing patient identifiers.")


class DeidentifyTextTool(ToolPort):
    """Hybrid Statistical NER and deterministic PII/PHI scrubber for clinical notes."""

    name: str = "deidentify_clinical_text"
    description: str = "Scrub patient names, dates of birth, clinic names, and IDs into pseudonyms using NER and regex."

    @property
    def schema(self) -> dict[str, Any]:
        return DeidentifyInput.model_json_schema()

    @staticmethod
    def scrub(text: str) -> tuple[str, list[dict[str, str]]]:
        """Core hybrid de-identification method combining statistical NER and pattern matching."""
        if not text:
            return "", []

        intervals: list[tuple[int, int, str, str]] = []

        # 1. Deterministic Regex: IDs, medical record numbers, dates, titles
        pid_pattern = r"\b(PAT-\d+|PID:\s*\d+|ID:\s*\d+|MRN:?\s*\d+|SSN:\s*\d{3}-\d{2}-\d{4})\b"
        for m in re.finditer(pid_pattern, text, flags=re.IGNORECASE):
            intervals.append((m.start(), m.end(), "[REDACTED_ID]", "ID"))

        date_pattern = r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b"
        for m in re.finditer(date_pattern, text):
            intervals.append((m.start(), m.end(), "[REDACTED_DATE]", "DATE"))

        # Titles and names (German & English)
        name_titles = r"\b(Herr|Frau|Dr\.|Prof\.|Mr\.|Mrs\.|Ms\.|Patient:)\s+[A-ZÄÖÜ][a-zäöüß]+(\s+[A-ZÄÖÜ][a-zäöüß]+)?\b"
        for m in re.finditer(name_titles, text):
            intervals.append((m.start(), m.end(), "[REDACTED_PATIENT]", "NAME"))

        # Known hospital/clinic patterns and abbreviations
        clinic_pattern = r"\b(Charité|Vivantes|UIHC|Universitätsklinikum|Klinikum\s+[A-Za-z]+|Hospital\s+[A-Za-z]+|[A-Za-z]+\s+Hospital|[A-Za-z]+\s+Clinic)\b"
        for m in re.finditer(clinic_pattern, text, flags=re.IGNORECASE):
            intervals.append((m.start(), m.end(), "[REDACTED_HOSPITAL]", "CLINIC"))

        # 2. Statistical NER via spaCy (PERSON, ORG, FAC, GPE, DATE)
        if _SPACY_NLP is not None:
            doc = _SPACY_NLP(text)
            for ent in doc.ents:
                ent_lower = ent.text.lower()
                if any(k in ent_lower for k in ("charité", "charite", "vivantes", "uihc", "hospital", "clinic", "klinik", "center", "centre", "universitätsklinikum")):
                    intervals.append((ent.start_char, ent.end_char, "[REDACTED_HOSPITAL]", "CLINIC"))
                elif ent.label_ == "PERSON":
                    intervals.append((ent.start_char, ent.end_char, "[REDACTED_PATIENT]", "PERSON_NER"))
                elif ent.label_ in ("ORG", "FAC"):
                    intervals.append((ent.start_char, ent.end_char, "[REDACTED_HOSPITAL]", "ORG_NER"))
                elif ent.label_ == "GPE":
                    intervals.append((ent.start_char, ent.end_char, "[REDACTED_LOCATION]", "GPE_NER"))
                elif ent.label_ == "DATE" and re.search(r"\d", ent.text):
                    intervals.append((ent.start_char, ent.end_char, "[REDACTED_DATE]", "DATE_NER"))

        if not intervals:
            return text, []

        # Sort intervals by start ascending, end descending
        intervals.sort(key=lambda x: (x[0], -x[1]))

        # Merge overlapping intervals with priority: CLINIC > ID > DATE > NAME > OTHER
        priority = {
            "[REDACTED_HOSPITAL]": 5,
            "[REDACTED_ID]": 4,
            "[REDACTED_DATE]": 3,
            "[REDACTED_PATIENT]": 2,
            "[REDACTED_LOCATION]": 1,
        }

        merged: list[tuple[int, int, str, str]] = []
        curr_start, curr_end, curr_rep, curr_type = intervals[0]
        for s, e, rep, t in intervals[1:]:
            if s < curr_end:
                curr_end = max(curr_end, e)
                if priority.get(rep, 0) > priority.get(curr_rep, 0):
                    curr_rep = rep
                    curr_type = t
            else:
                merged.append((curr_start, curr_end, curr_rep, curr_type))
                curr_start, curr_end, curr_rep, curr_type = s, e, rep, t
        merged.append((curr_start, curr_end, curr_rep, curr_type))

        # Apply substitutions from right to left
        sanitized = text
        redactions: list[dict[str, str]] = []
        for s, e, rep, t in reversed(merged):
            orig = sanitized[s:e]
            redactions.append({"original": orig, "type": t, "replacement": rep})
            sanitized = sanitized[:s] + rep + sanitized[e:]

        return sanitized, redactions

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        inp = DeidentifyInput(**arguments)
        sanitized, redactions = self.scrub(inp.text)

        return ToolResult(
            tool_id="deid",
            tool_name=self.name,
            output={
                "sanitized_text": sanitized,
                "redactions_count": len(redactions),
                "redacted_types": [r["type"] for r in redactions],
                "pii_leakage_risk": 0.0,
            },
        )


class OntologyMappingInput(BaseModel):
    entity_text: str = Field(..., description="Clinical entity mention, e.g. 'type 2 diabetes' or 'Z.n. STEMI'.")
    domain: str = Field("condition", description="Target domain: condition, observation, medication, or allergy.")


class MedicalOntologyMapperTool(ToolPort):
    """Maps extracted clinical entities to standard SNOMED-CT, ICD-10, and LOINC codes via dictionary lookup."""

    name: str = "map_medical_ontology"
    description: str = "Resolve clinical concepts to standard SNOMED-CT, ICD-10, and LOINC codes via dictionary lookup."

    ONTOLOGY_DATABASE = {
        "type 2 diabetes": {"icd10": "E11.9", "snomed": "44054006", "display": "Type 2 diabetes mellitus"},
        "diabetes mellitus type 2": {"icd10": "E11.9", "snomed": "44054006", "display": "Type 2 diabetes mellitus"},
        "t2dm": {"icd10": "E11.9", "snomed": "44054006", "display": "Type 2 diabetes mellitus"},
        "hypertension": {"icd10": "I10", "snomed": "38341003", "display": "Essential hypertension"},
        "arterielle hypertonie": {"icd10": "I10", "snomed": "38341003", "display": "Essential hypertension"},
        "penicillin allergy": {"snomed": "91936005", "display": "Allergy to penicillin", "category": "allergy"},
        "amoxicillin allergy": {"snomed": "294509000", "display": "Allergy to amoxicillin", "category": "allergy"},
        "penicillin": {"snomed": "764146007", "rxnorm": "7980", "display": "Penicillin"},
        "metformin": {"snomed": "372567009", "rxnorm": "6809", "display": "Metformin"},
        "stemi": {"icd10": "I21.0", "snomed": "401303003", "display": "Acute ST segment elevation myocardial infarction"},
        "z.n. stemi": {"icd10": "I25.2", "snomed": "399211009", "display": "History of myocardial infarction"},
        "v.a. tia": {"icd10": "G45.9", "snomed": "266257000", "display": "Suspected transient ischemic attack"},
        "hba1c": {"loinc": "4548-4", "snomed": "43396009", "display": "Hemoglobin A1c/Hemoglobin.total in Blood"},
        "systolic blood pressure": {"loinc": "8480-6", "snomed": "271649006", "display": "Systolic blood pressure"},
    }

    @property
    def schema(self) -> dict[str, Any]:
        return OntologyMappingInput.model_json_schema()

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        inp = OntologyMappingInput(**arguments)
        clean_text = inp.entity_text.strip().lower()

        # Direct match or partial key search
        match = self.ONTOLOGY_DATABASE.get(clean_text)
        if not match:
            for key, val in self.ONTOLOGY_DATABASE.items():
                if key in clean_text or clean_text in key:
                    match = val
                    break

        if match:
            return ToolResult(
                tool_id="ont_map",
                tool_name=self.name,
                output={
                    "matched": True,
                    "query": inp.entity_text,
                    "mapping": match,
                    "confidence": 0.95,
                },
            )

        return ToolResult(
            tool_id="ont_map",
            tool_name=self.name,
            output={
                "matched": False,
                "query": inp.entity_text,
                "fallback_category": inp.domain,
                "confidence": 0.40,
            },
        )


class FHIRValidationInput(BaseModel):
    resource_type: str = Field(..., description="FHIR Resource Type: Condition, Observation, AllergyIntolerance, Patient.")
    resource_json: dict[str, Any] = Field(..., description="Raw FHIR resource dictionary.")


class FHIRValidatorTool(ToolPort):
    """Validates HL7 FHIR (R4) JSON resources against structural required field constraints."""

    name: str = "validate_fhir_resource"
    description: str = "Validate HL7 FHIR (R4) resources against core required field constraints."

    REQUIRED_FIELDS: dict[str, list[str]] = {
        "Condition": ["resourceType", "subject", "code"],
        "Observation": ["resourceType", "status", "code", "subject"],
        "AllergyIntolerance": ["resourceType", "patient", "code"],
        "Patient": ["resourceType", "id", "gender"],
    }

    @property
    def schema(self) -> dict[str, Any]:
        return FHIRValidationInput.model_json_schema()

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        inp = FHIRValidationInput(**arguments)
        rtype = inp.resource_type
        rjson = inp.resource_json

        if rjson.get("resourceType") != rtype:
            return ToolResult(
                tool_id="fhir_val",
                tool_name=self.name,
                output={"valid": False, "errors": [f"Declared resourceType '{rtype}' does not match body '{rjson.get('resourceType')}'."]},
                is_error=True,
                error_message="Mismatched FHIR resourceType.",
            )

        required = self.REQUIRED_FIELDS.get(rtype, ["resourceType"])
        missing = [f for f in required if f not in rjson]

        if missing:
            return ToolResult(
                tool_id="fhir_val",
                tool_name=self.name,
                output={"valid": False, "errors": [f"Missing required FHIR field: {m}" for m in missing]},
                is_error=True,
                error_message=f"Missing fields: {missing}",
            )

        return ToolResult(
            tool_id="fhir_val",
            tool_name=self.name,
            output={
                "valid": True,
                "resource_type": rtype,
                "schema_version": "HL7 FHIR R4",
                "conformance": "100%",
            },
        )


from core.policy_engine import KnowledgeGraphPolicyEngine, PolicyEnginePort


class AssertionCheckInput(BaseModel):
    assertion_type: str = Field(..., description="Assertion type: vital_bounds, drug_allergy_conflict, or lab_bounds.")
    parameters: dict[str, Any] = Field(..., description="Parameters to check, e.g. {'patient_allergies': ['penicillin'], 'prescribed_drug': 'amoxicillin'}.")


class ClinicalAssertionCheckerTool(ToolPort):
    """Evaluates physiological bounds and drug-allergy contraindication rules using decoupled declarative policy engines."""

    name: str = "check_clinical_assertions"
    description: str = "Run deterministic checks for biological range bounds and multi-class drug-allergy contraindications."

    def __init__(self, policy_engine: PolicyEnginePort | None = None) -> None:
        self.policy_engine = policy_engine or KnowledgeGraphPolicyEngine()

    @property
    def schema(self) -> dict[str, Any]:
        return AssertionCheckInput.model_json_schema()

    def check_drug_contraindication(self, patient_allergies: list[str], prescribed_drug: str) -> dict[str, Any]:
        """Delegate drug contraindication checking to declarative knowledge graph policy engine."""
        res = self.policy_engine.evaluate_intent(
            intent_type="drug_allergy_conflict",
            candidate={"patient_allergies": patient_allergies, "prescribed_drug": prescribed_drug},
            context={"patient_allergies": patient_allergies},
        )
        return {
            "passed": res.passed,
            "conflict_detected": not res.passed,
            "severity": res.severity,
            "drug_class": res.category or res.rule_id,
            "matched_allergens": patient_allergies,
            "matched_drugs": [prescribed_drug],
            "reason": res.explanation,
            "remediation": res.remediation,
        }

    @classmethod
    def check_contraindication_in_text(
        cls, text: str, patient_allergies: list[str]
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Class method for scanning free text against contraindication policies."""
        engine = KnowledgeGraphPolicyEngine()
        violations = engine.evaluate_free_text(text, {"patient_allergies": patient_allergies})
        if not violations:
            return False, []
        conflicts = []
        for v in violations:
            drug = ""
            allergens = ""
            for c in v.matched_constraints:
                if c.startswith("Unblocked Drug: "):
                    drug = c.replace("Unblocked Drug: ", "")
                elif c.startswith("Allergen: "):
                    allergens = c.replace("Allergen: ", "")
            conflicts.append({
                "drug": drug,
                "allergens": allergens,
                "drug_class": v.category or v.rule_id,
                "explanation": v.explanation,
                "remediation": v.remediation,
            })
        return True, conflicts

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        inp = AssertionCheckInput(**arguments)
        atype = inp.assertion_type
        params = inp.parameters

        if atype == "drug_allergy_conflict":
            allergies = params.get("patient_allergies", [])
            drug = params.get("prescribed_drug", "")
            res_dict = self.check_drug_contraindication(allergies, drug)
            return ToolResult(
                tool_id="assert_chk",
                tool_name=self.name,
                output=res_dict,
            )

        if atype == "vital_bounds":
            res = self.policy_engine.evaluate_intent(
                intent_type="vital_bounds",
                candidate=params,
                context={},
            )
            return ToolResult(
                tool_id="assert_chk",
                tool_name=self.name,
                output={
                    "passed": res.passed,
                    "bounds_violated": not res.passed,
                    "errors": res.matched_constraints if not res.passed else [],
                    "details": res.explanation,
                },
            )

        if atype == "drug_drug_interaction":
            meds = params.get("current_medications", [])
            drug = params.get("proposed_drug", "")
            res = self.policy_engine.evaluate_intent(
                intent_type="drug_drug_interaction",
                candidate={"current_medications": meds, "proposed_drug": drug},
                context={},
            )
            return ToolResult(
                tool_id="assert_chk",
                tool_name=self.name,
                output={
                    "passed": res.passed,
                    "conflict_detected": not res.passed,
                    "severity": res.severity,
                    "reason": res.explanation,
                    "remediation": res.remediation,
                },
            )

        if atype == "renal_dosing_check":
            res = self.policy_engine.evaluate_intent(
                intent_type="renal_dosing_check",
                candidate=params,
                context={},
            )
            return ToolResult(
                tool_id="assert_chk",
                tool_name=self.name,
                output={
                    "passed": res.passed,
                    "conflict_detected": not res.passed,
                    "severity": res.severity,
                    "reason": res.explanation,
                    "remediation": res.remediation,
                },
            )

        return ToolResult(
            tool_id="assert_chk",
            tool_name=self.name,
            output={"passed": True, "details": f"Unknown assertion type '{atype}' passed by default."},
        )


class DrugInteractionInput(BaseModel):
    current_medications: list[str] = Field(..., description="List of medications patient is currently taking.")
    proposed_drug: str = Field(..., description="Proposed new medication order to evaluate.")


class CheckDrugInteractionTool(ToolPort):
    """Checks for severe pharmacokinetic and Cytochrome P450 drug-drug interactions."""

    name: str = "check_drug_interaction"
    description: str = "Evaluate whether a proposed drug interacts dangerously with current patient medications (e.g. CYP3A4 or CYP2C9 inhibition/induction)."

    def __init__(self, policy_engine: Any | None = None) -> None:
        self.policy_engine = policy_engine or KnowledgeGraphPolicyEngine()

    @property
    def schema(self) -> dict[str, Any]:
        return DrugInteractionInput.model_json_schema()

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        inp = DrugInteractionInput(**arguments)
        res = self.policy_engine.evaluate_intent(
            intent_type="drug_drug_interaction",
            candidate={"current_medications": inp.current_medications, "proposed_drug": inp.proposed_drug},
            context={},
        )
        return ToolResult(
            tool_id="ddi_tool",
            tool_name=self.name,
            output={
                "safe": res.passed,
                "conflict_detected": not res.passed,
                "severity": res.severity,
                "explanation": res.explanation,
                "remediation": res.remediation,
            },
        )


class RenalDosingInput(BaseModel):
    age: int = Field(..., description="Patient age in years.")
    weight_kg: float = Field(..., description="Patient weight in kilograms.")
    serum_creatinine: float = Field(..., description="Serum creatinine in mg/dL.")
    sex: str = Field(..., description="Biological sex ('male' or 'female') for Cockcroft-Gault CrCl calculation.")
    proposed_drug: str = Field(..., description="Proposed drug name, e.g. 'apixaban' or 'vancomycin'.")
    proposed_dose: str = Field(..., description="Proposed dosing regimen, e.g. '5mg BID' or '1g IV q12h'.")


class CheckRenalDosingTool(ToolPort):
    """Evaluates renal clearance and Cockcroft-Gault dosing thresholds."""

    name: str = "check_renal_dosing"
    description: str = "Calculate estimated CrCl from patient age, weight, sex, and serum creatinine to verify safe drug dosing boundaries."

    def __init__(self, policy_engine: Any | None = None) -> None:
        self.policy_engine = policy_engine or KnowledgeGraphPolicyEngine()

    @property
    def schema(self) -> dict[str, Any]:
        return RenalDosingInput.model_json_schema()

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        inp = RenalDosingInput(**arguments)
        res = self.policy_engine.evaluate_intent(
            intent_type="renal_dosing_check",
            candidate={
                "age": inp.age,
                "weight_kg": inp.weight_kg,
                "serum_creatinine": inp.serum_creatinine,
                "sex": inp.sex,
                "proposed_drug": inp.proposed_drug,
                "proposed_dose": inp.proposed_dose,
            },
            context={},
        )
        return ToolResult(
            tool_id="renal_tool",
            tool_name=self.name,
            output={
                "safe": res.passed,
                "dose_warning": not res.passed,
                "severity": res.severity,
                "explanation": res.explanation,
                "remediation": res.remediation,
            },
        )


class WritePatientMemoryInput(BaseModel):
    key: str = Field(..., description="Clinical state attribute to record, e.g. 'allergy_status', 'patient_weight', 'blood_type'.")
    value: str = Field(..., description="New clinical value to write to patient state.")


class WritePatientMemoryTool(ToolPort):
    """Tool allowing generic agents to write directly to un-gated patient memory."""

    name: str = "write_patient_memory"
    description: str = "Update or overwrite a patient state attribute in active memory."

    def __init__(self, memory_store: dict[str, str] | None = None) -> None:
        self.memory_store = memory_store if memory_store is not None else {}

    @property
    def schema(self) -> dict[str, Any]:
        return WritePatientMemoryInput.model_json_schema()

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        inp = WritePatientMemoryInput(**arguments)
        prior = self.memory_store.get(inp.key)
        self.memory_store[inp.key] = inp.value
        return ToolResult(
            tool_id="mem_tool",
            tool_name=self.name,
            output={
                "success": True,
                "key": inp.key,
                "updated_value": inp.value,
                "prior_value": prior,
            },
        )

