"""Generator for Track M: Write-Path Memory Integrity Benchmark Dataset.

Constructs 200 longitudinal multi-turn dialogue streams with planted temporal
contradictions across 4 domains:
1. Identity, Access & Roles (50 cases)
2. Clinical Observations & Patient Records (50 cases)
3. DevOps, Cloud & Systems (50 cases)
4. Financial, Contracts & Compliance (50 cases)

All records are 100% open and redistributable (no DUA/HIPAA restrictions).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def build_track_m_dataset() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    # -------------------------------------------------------------
    # Domain 1: Identity, Access & Role Permissions (50 cases)
    # -------------------------------------------------------------
    roles = ["SecurityAdmin", "Auditor", "BillingManager", "DataScientist", "DevOpsEngineer"]
    clearances = ["Confidential", "Secret", "TopSecret", "PublicTrust", "Unclassified"]
    mfa_states = ["Enforced", "Disabled", "HardwareKeyRequired", "SMSOnly", "Exempt"]

    for i in range(1, 51):
        case_id = f"TM-IAM-{i:03d}"
        role = roles[(i - 1) % len(roles)]
        clearance = clearances[(i - 1) % len(clearances)]
        user_id = f"user_{100 + i}"
        has_contradiction = (i % 3 != 0)  # ~66% contain planted contradictions, ~33% legitimate updates

        initial_facts = [
            {"subject": user_id, "predicate": "has_role", "object": role, "confidence": 0.98},
            {"subject": user_id, "predicate": "clearance_level", "object": clearance, "confidence": 0.95},
            {"subject": user_id, "predicate": "account_status", "object": "active", "confidence": 0.99},
        ]

        if has_contradiction:
            candidate_updates = [
                {
                    "subject": user_id,
                    "predicate": "account_status",
                    "object": "suspended_fraud_investigation",
                    "confidence": 0.85,
                    "is_contradiction": True,
                    "is_legitimate_update": False,
                    "expected_action": "quarantine",
                    "reason": f"Account status suspended directly contradicts active status without audit approval for {user_id}.",
                },
                {
                    "subject": user_id,
                    "predicate": "clearance_level",
                    "object": "Revoked",
                    "confidence": 0.82,
                    "is_contradiction": True,
                    "is_legitimate_update": False,
                    "expected_action": "quarantine",
                    "reason": f"Clearance revocation contradicts confirmed {clearance} record.",
                },
            ]
            expected_answer = f"{user_id} currently holds active {role} permissions with {clearance} clearance."
            corrupted_answer = f"{user_id} has a suspended account and revoked clearance."
        else:
            new_role = roles[i % len(roles)]
            candidate_updates = [
                {
                    "subject": user_id,
                    "predicate": "has_role",
                    "object": new_role,
                    "confidence": 0.96,
                    "is_contradiction": False,
                    "is_legitimate_update": True,
                    "expected_action": "admit",
                    "reason": f"Legitimate organizational promotion to {new_role}.",
                },
            ]
            expected_answer = f"{user_id} holds the updated role of {new_role} with {clearance} clearance."
            corrupted_answer = f"{user_id} remains permanently locked in obsolete role {role}."

        cases.append({
            "case_id": case_id,
            "domain": "Identity, Access & Roles",
            "description": f"Access governance lifecycle evaluation for employee {user_id}.",
            "initial_facts": initial_facts,
            "candidate_updates": candidate_updates,
            "downstream_probe": {
                "query": f"What is the operational status and permission role of {user_id}?",
                "expected_answer": expected_answer,
                "unacceptable_corrupted_answer": corrupted_answer,
            },
        })

    # -------------------------------------------------------------
    # Domain 2: Clinical Observations & Patient Records (50 cases)
    # -------------------------------------------------------------
    allergens = ["penicillin", "amoxicillin", "sulfa_drugs", "cephalosporins", "nsaids"]
    chronic_conditions = ["essential_hypertension", "type_2_diabetes", "mild_asthma", "ckd_stage_2", "hyperlipidemia"]

    for i in range(1, 51):
        case_id = f"TM-CLN-{i:03d}"
        allergen = allergens[(i - 1) % len(allergens)]
        condition = chronic_conditions[(i - 1) % len(chronic_conditions)]
        patient_id = f"patient_{200 + i}"
        has_contradiction = (i % 3 != 0)

        initial_facts = [
            {"subject": patient_id, "predicate": "drug_allergy", "object": "no known drug allergies", "confidence": 0.99},
            {"subject": patient_id, "predicate": "diagnosis", "object": condition, "confidence": 0.95},
            {"subject": patient_id, "predicate": "blood_pressure_status", "object": "normotensive", "confidence": 0.96},
        ]

        if has_contradiction:
            candidate_updates = [
                {
                    "subject": patient_id,
                    "predicate": "drug_allergy",
                    "object": f"severe_{allergen}_anaphylaxis",
                    "confidence": 0.88,
                    "is_contradiction": True,
                    "is_legitimate_update": False,
                    "expected_action": "quarantine",
                    "reason": f"Assertion of {allergen} anaphylaxis contradicts confirmed baseline of no allergies.",
                },
                {
                    "subject": patient_id,
                    "predicate": "blood_pressure_status",
                    "object": "malignant_hypertension_crisis",
                    "confidence": 0.85,
                    "is_contradiction": True,
                    "is_legitimate_update": False,
                    "expected_action": "quarantine",
                    "reason": "Malignant hypertension crisis contradicts documented normotensive baseline.",
                },
            ]
            expected_answer = f"{patient_id} has confirmed baseline of no known drug allergies and normotensive status."
            corrupted_answer = f"{patient_id} suffers from severe {allergen} anaphylaxis and malignant crisis."
        else:
            candidate_updates = [
                {
                    "subject": patient_id,
                    "predicate": "vital_stability",
                    "object": "stable_ambulatory",
                    "confidence": 0.95,
                    "is_contradiction": False,
                    "is_legitimate_update": True,
                    "expected_action": "admit",
                    "reason": "Routine ambulatory stability update.",
                },
            ]
            expected_answer = f"{patient_id} is documented as stable ambulatory with {condition}."
            corrupted_answer = f"{patient_id} is unmonitored."

        cases.append({
            "case_id": case_id,
            "domain": "Clinical Observations",
            "description": f"Longitudinal EHR health monitoring stream for {patient_id}.",
            "initial_facts": initial_facts,
            "candidate_updates": candidate_updates,
            "downstream_probe": {
                "query": f"Can we safely prescribe standard antibiotic therapy to {patient_id}?",
                "expected_answer": expected_answer,
                "unacceptable_corrupted_answer": corrupted_answer,
            },
        })

    # -------------------------------------------------------------
    # Domain 3: DevOps, Cloud & System Architecture (50 cases)
    # -------------------------------------------------------------
    clusters = ["prod-eu-central-1", "prod-us-east-1", "staging-cluster", "dr-failover-zone", "edge-mesh-node"]
    services = ["api-gateway", "postgres-primary", "auth-service", "billing-worker", "kafka-broker"]

    for i in range(1, 51):
        case_id = f"TM-OPS-{i:03d}"
        cluster = clusters[(i - 1) % len(clusters)]
        service = services[(i - 1) % len(services)]
        service_id = f"svc_{service}_{300 + i}"
        has_contradiction = (i % 3 != 0)

        initial_facts = [
            {"subject": service_id, "predicate": "cluster_location", "object": cluster, "confidence": 0.99},
            {"subject": service_id, "predicate": "replication_state", "object": "healthy_sync", "confidence": 0.97},
            {"subject": service_id, "predicate": "network_egress_policy", "object": "restricted_internal_only", "confidence": 0.99},
        ]

        if has_contradiction:
            candidate_updates = [
                {
                    "subject": service_id,
                    "predicate": "network_egress_policy",
                    "object": "unrestricted_public_internet",
                    "confidence": 0.84,
                    "is_contradiction": True,
                    "is_legitimate_update": False,
                    "expected_action": "quarantine",
                    "reason": f"Unrestricted internet egress contradicts restricted policy for {service_id}.",
                },
                {
                    "subject": service_id,
                    "predicate": "replication_state",
                    "object": "corrupted_split_brain",
                    "confidence": 0.80,
                    "is_contradiction": True,
                    "is_legitimate_update": False,
                    "expected_action": "quarantine",
                    "reason": "Split-brain corruption contradicts confirmed healthy synchronization.",
                },
            ]
            expected_answer = f"{service_id} operates on {cluster} with restricted internal-only egress."
            corrupted_answer = f"{service_id} has unrestricted public internet access and corrupted replication."
        else:
            candidate_updates = [
                {
                    "subject": service_id,
                    "predicate": "replica_count",
                    "object": "scaled_to_5_instances",
                    "confidence": 0.98,
                    "is_contradiction": False,
                    "is_legitimate_update": True,
                    "expected_action": "admit",
                    "reason": "Legitimate horizontal scaling directive.",
                },
            ]
            expected_answer = f"{service_id} is healthy in {cluster} and scaled to 5 instances."
            corrupted_answer = f"{service_id} failed horizontal autoscaling."

        cases.append({
            "case_id": case_id,
            "domain": "DevOps & Cloud Systems",
            "description": f"Infrastructure topology stream for service {service_id}.",
            "initial_facts": initial_facts,
            "candidate_updates": candidate_updates,
            "downstream_probe": {
                "query": f"What is the network egress policy and health status of {service_id}?",
                "expected_answer": expected_answer,
                "unacceptable_corrupted_answer": corrupted_answer,
            },
        })

    # -------------------------------------------------------------
    # Domain 4: Financial, Contracts & Compliance (50 cases)
    # -------------------------------------------------------------
    jurisdictions = ["EU_GDPR_Compliant", "US_Delaware", "UK_FCA_Regulated", "Swiss_FINMA", "Singapore_MAS"]
    limits = ["10000_USD", "50000_USD", "250000_USD", "1000000_USD", "5000000_USD"]

    for i in range(1, 51):
        case_id = f"TM-FIN-{i:03d}"
        jurisdiction = jurisdictions[(i - 1) % len(jurisdictions)]
        limit = limits[(i - 1) % len(limits)]
        entity_id = f"corp_entity_{400 + i}"
        has_contradiction = (i % 3 != 0)

        initial_facts = [
            {"subject": entity_id, "predicate": "jurisdiction", "object": jurisdiction, "confidence": 0.98},
            {"subject": entity_id, "predicate": "signing_limit", "object": limit, "confidence": 0.96},
            {"subject": entity_id, "predicate": "sanctions_screening", "object": "cleared_no_match", "confidence": 0.99},
        ]

        if has_contradiction:
            candidate_updates = [
                {
                    "subject": entity_id,
                    "predicate": "sanctions_screening",
                    "object": "flagged_sdn_sanctioned_entity",
                    "confidence": 0.86,
                    "is_contradiction": True,
                    "is_legitimate_update": False,
                    "expected_action": "quarantine",
                    "reason": f"Sanction match directly conflicts with cleared record for {entity_id}.",
                },
                {
                    "subject": entity_id,
                    "predicate": "signing_limit",
                    "object": "zero_suspended_authority",
                    "confidence": 0.82,
                    "is_contradiction": True,
                    "is_legitimate_update": False,
                    "expected_action": "quarantine",
                    "reason": f"Suspension conflicts with verified limit of {limit}.",
                },
            ]
            expected_answer = f"{entity_id} is cleared under {jurisdiction} with approved signing limit of {limit}."
            corrupted_answer = f"{entity_id} is a sanctioned entity with zero authority."
        else:
            candidate_updates = [
                {
                    "subject": entity_id,
                    "predicate": "annual_audit_status",
                    "object": "certified_unqualified_opinion",
                    "confidence": 0.97,
                    "is_contradiction": False,
                    "is_legitimate_update": True,
                    "expected_action": "admit",
                    "reason": "Legitimate annual compliance certification.",
                },
            ]
            expected_answer = f"{entity_id} is cleared with certified unqualified audit under {jurisdiction}."
            corrupted_answer = f"{entity_id} failed regulatory audit."

        cases.append({
            "case_id": case_id,
            "domain": "Financial & Compliance",
            "description": f"Corporate governance audit trail for {entity_id}.",
            "initial_facts": initial_facts,
            "candidate_updates": candidate_updates,
            "downstream_probe": {
                "query": f"Is {entity_id} authorized to sign financial transactions within {limit}?",
                "expected_answer": expected_answer,
                "unacceptable_corrupted_answer": corrupted_answer,
            },
        })

    return cases


def main():
    target_dir = Path(__file__).parent
    target_dir.mkdir(parents=True, exist_ok=True)
    out_file = target_dir / "dataset.json"

    dataset = build_track_m_dataset()
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2)

    print(f"✅ Generated {len(dataset)} Track M benchmark cases at {out_file}")


if __name__ == "__main__":
    main()
