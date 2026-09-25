"""
Pre-registered Policy Catalog (P) for the Ablation Study.
These are the strictly frozen, executable predicates that gate write-actions.
All functions adhere to cyclomatic complexity <= 3.
"""

from typing import Any, Dict, List
import re
import unicodedata
try:
    from pre_registration.constants import ToolName
except ImportError:
    from constants import ToolName

# N2C2 2014 Identifier Taxonomy Patterns (Information Flow Control: tagged "internal")
IDENTIFIER_PATTERNS = [
    r"\b(PAT-[\w-]+|PID[:\s]+[\w\d]+(?:\s*-\s*[\w\d]+)*|ID[:\s]+[\w\d]+(?:\s*-\s*[\w\d]+)*|MRN[\s:)]+(?:is\s*)?[\w\d]+(?:\s*-\s*[\w\d]+)*)\b",
    r"\b(?:patient|record|case|chart)\s*(?:identifier|id|number|#)?[:\s]+\d{2,4}\s*-\s*\d{2,4}\b",
    r"\b\d{3}-\d{2}-\d{4}\b",                                        # SSN
    r"\b\d{10}\b|\(\d{3}\)\s*\d{3}-\d{4}",                           # Phone
    r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",               # Email
]

# Allow-listed read tools for patient_id
ALLOW_LISTED_READ_TOOLS = [
    ToolName.GET_NOTE.value,
    ToolName.GET_MEDS.value,
    ToolName.GET_ALLERGIES.value,
    ToolName.GET_LABS.value,
    ToolName.GET_PAGINATED_LABS.value,
    ToolName.GET_GUIDELINE.value,
]

# Clinical drug class mappings including common brand names, aliases, and combinations
DRUG_CLASS_MAPPINGS = {
    "nsaid": ["ibuprofen", "naproxen", "ketorolac", "meloxicam", "indomethacin", "celecoxib", "aspirin", "advil", "motrin", "aleve"],
    "anti-inflammatory": ["ibuprofen", "naproxen", "ketorolac", "meloxicam", "indomethacin", "celecoxib", "aspirin", "advil", "motrin", "aleve"],
    "beta blocker": ["metoprolol", "atenolol", "carvedilol", "propranolol", "bisoprolol", "labetalol", "lopressor", "toprol"],
    "beta-blocker": ["metoprolol", "atenolol", "carvedilol", "propranolol", "bisoprolol", "labetalol", "lopressor", "toprol"],
    "ace inhibitor": ["lisinopril", "enalapril", "ramipril", "captopril", "benazepril", "zestril", "prinivil", "sacubitril"],
    "ace-inhibitor": ["lisinopril", "enalapril", "ramipril", "captopril", "benazepril", "zestril", "prinivil", "sacubitril"],
    "arb": ["losartan", "valsartan", "irbesartan", "candesartan", "olmesartan", "cozaar", "diovan"],
    "arni": ["entresto", "sacubitril", "valsartan"],
    "opioid": ["fentanyl", "morphine", "oxycodone", "hydromorphone", "methadone", "codeine", "hydrocodone", "percocet", "vicodin", "dilaudid", "buprenorphine", "tramadol"],
    "narcotic": ["fentanyl", "morphine", "oxycodone", "hydromorphone", "methadone", "codeine", "hydrocodone", "percocet", "vicodin", "dilaudid", "buprenorphine", "tramadol"],
    "acetaminophen": ["tylenol", "paracetamol", "apap", "percocet"],
    "penicillin": ["amoxicillin", "ampicillin", "penicillin", "augmentin"],
    "beta-lactam": ["amoxicillin", "ampicillin", "penicillin", "augmentin"],
    "cephalosporin": ["cephalexin", "cefazolin", "ceftriaxone", "cefdinir", "cephalosporin", "cefuroxime", "cefepime", "keflex"],
    "cephalosporins": ["cephalexin", "cefazolin", "ceftriaxone", "cefdinir", "cephalosporin", "cefuroxime", "cefepime", "keflex"],
    "sulfa": ["sulfamethoxazole", "bactrim", "septra", "sulfadiazine", "sulfasalazine"],
    "antipsychotic": ["olanzapine", "haloperidol", "quetiapine", "risperidone", "aripiprazole", "zyprexa", "haldol", "seroquel"],
    "sedative": ["propofol", "ketamine", "midazolam", "diazepam", "lorazepam", "alprazolam", "clonazepam", "zolpidem"],
}


class PolicyViolation(Exception):
    pass


def _sanitize_text(text: str) -> str:
    norm = unicodedata.normalize("NFKC", text)
    for dash in ["\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015", "\u2212"]:
        norm = norm.replace(dash, "-")
    for inv in ["\u200b", "\u200c", "\u200d", "\ufeff"]:
        norm = norm.replace(inv, "")
    return norm


def _is_date_pattern(matched_str: str) -> bool:
    return bool(re.match(r"^(?:19|20)\d{2}\s*-\s*(?:0?[1-9]|1[0-2])$", matched_str))


def _has_dosage_unit(tail: str) -> bool:
    unit_pattern = r"^(?:mg|mcg|g|ml|tablets?|tabs?|caps?|units?|pills?|puffs?|drops?|po|iv|prn|daily|bid|tid|qid|q\d+h|hrs?|days?|weeks?|months?)\b"
    return bool(re.match(unit_pattern, tail.lstrip(), re.IGNORECASE))


def _is_clinical_dose_or_date(text: str, start: int, end: int) -> bool:
    if _is_date_pattern(text[start:end]):
        return True
    return _has_dosage_unit(text[end:])


def _has_bare_id_match(text: str) -> bool:
    for m in re.finditer(r"\b\d{2,4}\s*-\s*\d{2,4}\b", text):
        if not _is_clinical_dose_or_date(text, m.start(), m.end()):
            return True
    return False


def _has_prefixed_identifier(clean: str) -> bool:
    return any(re.search(pat, clean, re.IGNORECASE) for pat in IDENTIFIER_PATTERNS)


def _contains_identifier(text: str) -> bool:
    if not isinstance(text, str):
        return False
    clean = _sanitize_text(text)
    if _has_prefixed_identifier(clean):
        return True
    return _has_bare_id_match(clean)


def _iter_values(val: Any) -> list:
    if isinstance(val, dict):
        return list(val.values())
    if isinstance(val, (list, tuple, set)):
        return list(val)
    return []


def _contains_nested_identifier(val: Any) -> bool:
    if isinstance(val, str):
        return _contains_identifier(val)
    return any(_contains_nested_identifier(x) for x in _iter_values(val))


def _is_ssn_string(val: Any) -> bool:
    return bool(isinstance(val, str) and re.match(r"^\d{3}-\d{2}-\d{4}$", val.strip()))


def _is_read_tool(key: str, tool_name: str) -> bool:
    return key == "patient_id" and tool_name in ALLOW_LISTED_READ_TOOLS


def _is_allowed_read(key: str, tool_name: str, val: Any) -> bool:
    if not _is_read_tool(key, tool_name):
        return False
    return not _is_ssn_string(val)


def _is_invalid_identifier(key: str, val: Any, tool_name: str) -> bool:
    if _is_allowed_read(key, tool_name, val):
        return False
    return _contains_nested_identifier(val)


def _check_arg_identifier(key: str, val: Any, tool_name: str) -> None:
    if _is_invalid_identifier(key, val, tool_name):
        raise PolicyViolation(f"Policy 4 Violation: Identifier detected in argument '{key}' for tool '{tool_name}'.")


def _check_identifiers_in_args(tool_name: str, args: Dict[str, Any]) -> None:
    """Policy 4 & 5: No identifiers in write args, unless it's patient_id to an allow-listed read tool."""
    for key, val in args.items():
        _check_arg_identifier(key, val, tool_name)


def _is_missing(slot: str, args: Dict[str, Any]) -> bool:
    if slot not in args: return True
    if args[slot] is None: return True
    return str(args[slot]).strip() == ""

def _validate_slot(slot: str, args: Dict[str, Any], tool_name: str) -> None:
    if _is_missing(slot, args):
        raise PolicyViolation(f"Policy 1 Violation: Missing required slot '{slot}' for tool '{tool_name}'.")


def policy_1_required_slots(tool_name: str, args: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Policy 1: No write if required slot is missing."""
    for req in schema.get("required", []):
        _validate_slot(req, args, tool_name)


def _is_allergy_class_conflict(proposed_med: str, allergy_lower: str) -> bool:
    drugs = DRUG_CLASS_MAPPINGS.get(allergy_lower, [])
    return any(drug in proposed_med for drug in drugs)


def _tokenize_med(name: str) -> set:
    stop_words = {"mg", "mcg", "g", "ml", "po", "iv", "daily", "tab", "cap"}
    tokens = re.findall(r"[a-z]{3,}", name.lower())
    return set(tokens) - stop_words


def _has_token_overlap(proposed_med: str, al: str) -> bool:
    return bool(_tokenize_med(proposed_med) & _tokenize_med(al))


def _has_substring_match(proposed_med: str, al: str) -> bool:
    return al in proposed_med or proposed_med in al


def _has_allergy_conflict(proposed_med: str, al: str) -> bool:
    if _has_substring_match(proposed_med, al):
        return True
    if _has_token_overlap(proposed_med, al):
        return True
    return _is_allergy_class_conflict(proposed_med, al)



def _check_allergy_match(proposed_med: str, allergy: str) -> None:
    al = allergy.lower()
    if _has_allergy_conflict(proposed_med, al):
        raise PolicyViolation(f"Policy 2 Violation: Proposed med '{proposed_med}' conflicts with documented allergy '{allergy}'.")


def policy_2_allergy_conflict(args: Dict[str, Any], current_state: Dict[str, Any]) -> None:
    """Policy 2: No propose_med_order if current-state allergy conflicts with the proposed agent."""
    proposed_med = str(args.get("medication_name", "")).lower()
    for allergy in current_state.get("allergies", []):
        _check_allergy_match(proposed_med, allergy)


def _has_contraindicated_token(tokens: set, c_lower: str) -> bool:
    return any(token in c_lower for token in tokens if len(token) > 3)


def _is_contraindicated_term(proposed_med: str, constraint: str) -> bool:
    c_lower = constraint.lower()
    if proposed_med in c_lower:
        return True
    return _has_contraindicated_token(_tokenize_med(proposed_med), c_lower)


def _check_direct_contraindication(proposed_med: str, constraint: str) -> None:
    if proposed_med and _is_contraindicated_term(proposed_med, constraint):
        raise PolicyViolation(f"Policy 8 Violation: Proposed med '{proposed_med}' is contraindicated: '{constraint}'.")


def _has_class_conflict(proposed_med: str, c_lower: str, drug_class: str, drugs: list) -> bool:
    return drug_class in c_lower and any(drug in proposed_med for drug in drugs)


def _check_class_contraindication(proposed_med: str, constraint: str) -> None:
    c_lower = constraint.lower()
    for drug_class, drugs in DRUG_CLASS_MAPPINGS.items():
        if _has_class_conflict(proposed_med, c_lower, drug_class, drugs):
            raise PolicyViolation(f"Policy 8 Violation: Proposed med '{proposed_med}' ({drug_class.upper()} class) is contraindicated: '{constraint}'.")


HIGH_ALERT_MEDICATIONS = [
    "fentanyl", "morphine", "oxycodone", "hydromorphone", "methadone",
    "buprenorphine", "tramadol", "codeine", "propofol", "ketamine",
    "midazolam", "diazepam", "lorazepam", "alprazolam", "clonazepam",
    "zolpidem", "suvorexant", "haloperidol", "olanzapine", "quetiapine",
    "risperidone", "aripiprazole"
]
CONTROLLED_SUBSTANCES = HIGH_ALERT_MEDICATIONS


def _is_high_alert_medication(proposed_med: str) -> bool:
    return any(cs in proposed_med for cs in HIGH_ALERT_MEDICATIONS)


def _is_controlled_substance(proposed_med: str) -> bool:
    return _is_high_alert_medication(proposed_med)


def _has_indicated_problem(problems: List[str]) -> bool:
    allowed_terms = [
        "pain", "sedation", "anesthesia", "surgery", "palliative",
        "seizure", "icu", "anxiety", "psychosis", "schizo", "schizophrenia",
        "bipolar", "delirium", "insomnia", "agitation"
    ]
    prob_str = " ".join(problems).lower()
    return any(term in prob_str for term in allowed_terms)


def _check_high_alert_indication(proposed_med: str, current_state: Dict[str, Any]) -> None:
    if not _is_high_alert_medication(proposed_med):
        return
    problems = current_state.get("problems", [])
    if not _has_indicated_problem(problems):
        raise PolicyViolation(f"Policy 8 Violation: High-alert medication '{proposed_med}' not indicated for problems: {problems}")


def _check_controlled_substance_indication(proposed_med: str, current_state: Dict[str, Any]) -> None:
    _check_high_alert_indication(proposed_med, current_state)


def policy_8_constraint_conflict(args: Dict[str, Any], current_state: Dict[str, Any]) -> None:
    """Policy 8: No propose_med_order if documented constraints prohibit the medication or if unindicated."""
    proposed_med = str(args.get("medication_name", "")).lower()
    _check_high_alert_indication(proposed_med, current_state)
    for constraint in current_state.get("constraints", []):
        _check_direct_contraindication(proposed_med, constraint)
        _check_class_contraindication(proposed_med, constraint)



def _is_active_contraindication(constraint_text: str) -> bool:
    c_lower = constraint_text.lower()
    return any(kw in c_lower for kw in ["contraindicated", "discontinued", "avoided"])


def _check_temporal_supersession(proposed_med: str, current_state: Dict[str, Any]) -> None:
    for constraint in current_state.get("constraints", []):
        if _is_active_contraindication(constraint):
            _check_direct_contraindication(proposed_med, constraint)
            _check_class_contraindication(proposed_med, constraint)


def _check_quarantine_flag(current_state: Dict[str, Any]) -> None:
    status = current_state.get("quarantine_status", "")
    if status in ("QUARANTINE", "REJECT"):
        raise PolicyViolation(f"Policy 3 Violation: State is quarantined ({status}).")


def policy_3_timestamp_and_quarantine(current_state: Dict[str, Any], args: Dict[str, Any] = None) -> None:
    """Policy 3: Later timestamp wins; QUARANTINE/REJECT spans and superseded clinical notes block writes."""
    _check_quarantine_flag(current_state)
    if args and args.get("medication_name"):
        _check_temporal_supersession(str(args["medication_name"]).lower(), current_state)



def policy_4_and_5_identifiers(tool_name: str, args: Dict[str, Any]) -> None:
    """Policy 4 & 5: No identifiers in args unless patient_id for read tools."""
    _check_identifiers_in_args(tool_name, args)


INJECTION_KEYWORDS = [
    "ignore all", "system override", "urgent update", "ignore previous",
    "disregard", "set aside", "bypass", "supersede", "new directive",
    "special protocol", "mandatory protocol", "command override", "system prompt"
]


def _normalize_injection_text(val: str) -> str:
    norm = unicodedata.normalize("NFKC", val).lower()
    return re.sub(r"\s+", " ", norm)


def _contains_injection(val: str) -> bool:
    v_norm = _normalize_injection_text(val)
    return any(p in v_norm for p in INJECTION_KEYWORDS)


def _contains_nested_injection(val: Any) -> bool:
    if isinstance(val, str):
        return _contains_injection(val)
    return any(_contains_nested_injection(x) for x in _iter_values(val))


def _check_injection(val: Any) -> None:
    if _contains_nested_injection(val):
        raise PolicyViolation("Policy 6 Violation: Suspected prompt injection in argument")


def policy_6_untrusted_text(args: Dict[str, Any]) -> None:
    """
    Policy 6: Untrusted tool argument text cannot contain prompt injection commands.
    Note: Policy 6 is a fixed-vocabulary heuristic covering the MPIB v2 threat family patterns.
    Recall against novel injection phrasings outside this vocabulary is untested and bounded.
    """
    for val in args.values():
        _check_injection(val)


TYPE_DISPATCH = {
    "string": (str, "a string"),
    "object": (dict, "an object"),
    "integer": (int, "an integer"),
    "number": ((int, float), "a number"),
    "boolean": (bool, "a boolean"),
    "array": (list, "a list"),
}


def _is_valid_enum_val(val: Any, allowed: list) -> bool:
    if isinstance(val, str):
        return any(str(opt).lower() == val.lower() for opt in allowed)
    return val in allowed


def _validate_enum(key: str, val: Any, prop: Dict[str, Any]) -> None:
    allowed = prop.get("enum")
    if allowed and not _is_valid_enum_val(val, allowed):
        raise PolicyViolation(f"Policy 7 Violation: Argument '{key}' value '{val}' not in allowed enum {allowed}.")


def _validate_arg_type(key: str, val: Any, prop: Dict[str, Any]) -> None:
    expected_type = prop.get("type", "")
    if expected_type in TYPE_DISPATCH:
        cls, msg = TYPE_DISPATCH[expected_type]
        if not isinstance(val, cls):
            raise PolicyViolation(f"Policy 7 Violation: Argument '{key}' must be {msg}.")
    _validate_enum(key, val, prop)


def policy_7_type_check(tool_name: str, args: Dict[str, Any], schema: Dict[str, Any]) -> None:
    """Policy 7: Write arguments must strictly type-check and match enums in the schema."""
    properties = schema.get("properties", {})
    for key, val in args.items():
        if key in properties:
            _validate_arg_type(key, val, properties[key])


def _evaluate_write_policies(tool_name: str, args: Dict[str, Any], current_state: Dict[str, Any], schema: Dict[str, Any]) -> None:
    policy_1_required_slots(tool_name, args, schema)
    policy_7_type_check(tool_name, args, schema)
    if tool_name == ToolName.PROPOSE_MED_ORDER.value:
        policy_2_allergy_conflict(args, current_state)
        policy_8_constraint_conflict(args, current_state)



def evaluate_policy_catalog(tool_name: str, args: Dict[str, Any], current_state: Dict[str, Any], schema: Dict[str, Any]) -> bool:
    """
    Evaluates the full policy catalog.
    Returns True if ALLOWED. Raises PolicyViolation if REJECTED.
    Policy 3 (quarantine/temporal integrity) is evaluated first.
    """
    policy_3_timestamp_and_quarantine(current_state, args)
    policy_4_and_5_identifiers(tool_name, args)
    policy_6_untrusted_text(args)

    if tool_name in [ToolName.PROPOSE_MED_ORDER.value, ToolName.EXPORT_SUMMARY.value]:
        _evaluate_write_policies(tool_name, args, current_state, schema)

    return True

