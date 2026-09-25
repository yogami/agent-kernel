"""Dataset and Structural Causal Model Generator for Track C Benchmark.

Curates 20 open GraphML Structural Causal Models across 3 domains:
1. Operational & Cloud Infrastructure (7 SCMs)
2. Financial & Governance Systems (6 SCMs)
3. Biomedical & Clinical Care (7 SCMs)

Pairs each SCM with N=500 observational records:
- 10 valid datasets adhering strictly to the DAG's d-separation topology.
- 10 corrupted datasets with planted confounding or broken mechanisms.
"""

from __future__ import annotations

import json
from pathlib import Path
import random
from typing import Any

from core.causal_graph import CausalNodeType, StructuralCausalModel


def create_all_20_scms() -> list[tuple[StructuralCausalModel, bool, str | None, dict[str, str]]]:
    """Build the 20 benchmark SCMs with validity flags and proposed interventions."""
    scms_info = []

    # =============================================================
    # Domain 1: Operational & Infrastructure (7 SCMs)
    # =============================================================
    # 1. cluster_autoscaling (Valid)
    scm1 = StructuralCausalModel("cluster_autoscaling")
    scm1.add_node("traffic_load", node_type=CausalNodeType.EXOGENOUS)
    scm1.add_node("cpu_utilization", node_type=CausalNodeType.ENDOGENOUS)
    scm1.add_node("network_io", node_type=CausalNodeType.ENDOGENOUS)
    scm1.add_node("replica_count", node_type=CausalNodeType.INTERVENTION)
    scm1.add_node("p99_latency", node_type=CausalNodeType.OUTCOME)
    scm1.add_edge("traffic_load", "cpu_utilization", weight=0.8)
    scm1.add_edge("traffic_load", "network_io", weight=0.7)
    scm1.add_edge("cpu_utilization", "replica_count", weight=0.6)
    scm1.add_edge("replica_count", "p99_latency", weight=-0.4)
    scm1.add_edge("network_io", "p99_latency", weight=0.8)
    scms_info.append((
        scm1, False, None,
        {"intervention": "replica_count", "outcome": "p99_latency", "effect": "reduce latency"},
    ))

    # 2. db_connection_pool (Valid)
    scm2 = StructuralCausalModel("db_connection_pool")
    scm2.add_node("request_volume", node_type=CausalNodeType.EXOGENOUS)
    scm2.add_node("active_connections", node_type=CausalNodeType.ENDOGENOUS)
    scm2.add_node("queue_depth", node_type=CausalNodeType.ENDOGENOUS)
    scm2.add_node("pool_size", node_type=CausalNodeType.INTERVENTION)
    scm2.add_node("db_latency", node_type=CausalNodeType.OUTCOME)
    scm2.add_edge("request_volume", "active_connections", weight=0.7)
    scm2.add_edge("active_connections", "queue_depth", weight=0.8)
    scm2.add_edge("pool_size", "queue_depth", weight=-0.7)
    scm2.add_edge("queue_depth", "db_latency", weight=0.85)
    scms_info.append((
        scm2, False, None,
        {"intervention": "pool_size", "outcome": "db_latency", "effect": "drain connection queue"},
    ))

    # 3. api_circuit_breaker (Valid)
    scm3 = StructuralCausalModel("api_circuit_breaker")
    scm3.add_node("upstream_outage", node_type=CausalNodeType.EXOGENOUS)
    scm3.add_node("error_rate", node_type=CausalNodeType.ENDOGENOUS)
    scm3.add_node("trip_threshold", node_type=CausalNodeType.INTERVENTION)
    scm3.add_node("circuit_state", node_type=CausalNodeType.ENDOGENOUS)
    scm3.add_node("fallback_traffic", node_type=CausalNodeType.OUTCOME)
    scm3.add_edge("upstream_outage", "error_rate", weight=0.9)
    scm3.add_edge("error_rate", "circuit_state", weight=0.8)
    scm3.add_edge("trip_threshold", "circuit_state", weight=-0.6)
    scm3.add_edge("circuit_state", "fallback_traffic", weight=0.85)
    scms_info.append((
        scm3, False, None,
        {"intervention": "trip_threshold", "outcome": "fallback_traffic", "effect": "protect service"},
    ))

    # 4. cache_invalidation (Valid)
    scm4 = StructuralCausalModel("cache_invalidation")
    scm4.add_node("mutation_rate", node_type=CausalNodeType.EXOGENOUS)
    scm4.add_node("invalidation_rate", node_type=CausalNodeType.INTERVENTION)
    scm4.add_node("cache_miss_rate", node_type=CausalNodeType.ENDOGENOUS)
    scm4.add_node("backend_cpu_load", node_type=CausalNodeType.OUTCOME)
    scm4.add_edge("mutation_rate", "invalidation_rate", weight=0.7)
    scm4.add_edge("invalidation_rate", "cache_miss_rate", weight=0.75)
    scm4.add_edge("cache_miss_rate", "backend_cpu_load", weight=0.8)
    scms_info.append((
        scm4, False, None,
        {"intervention": "invalidation_rate", "outcome": "backend_cpu_load", "effect": "balance freshness"},
    ))

    # 5. deploy_canary (Valid)
    scm5 = StructuralCausalModel("deploy_canary")
    scm5.add_node("canary_traffic_split", node_type=CausalNodeType.INTERVENTION)
    scm5.add_node("canary_error_rate", node_type=CausalNodeType.ENDOGENOUS)
    scm5.add_node("burn_rate", node_type=CausalNodeType.ENDOGENOUS)
    scm5.add_node("incident_severity", node_type=CausalNodeType.OUTCOME)
    scm5.add_edge("canary_traffic_split", "canary_error_rate", weight=0.65)
    scm5.add_edge("canary_error_rate", "burn_rate", weight=0.8)
    scm5.add_edge("burn_rate", "incident_severity", weight=0.7)
    scms_info.append((
        scm5, False, None,
        {"intervention": "canary_traffic_split", "outcome": "incident_severity", "effect": "prevent blast radius"},
    ))

    # 6. network_latency_routing (CORRUPTED: Planted Confounder U -> link_congestion, rtt_latency)
    scm6 = StructuralCausalModel("network_latency_routing")
    scm6.add_node("link_congestion", node_type=CausalNodeType.ENDOGENOUS)
    scm6.add_node("packet_loss", node_type=CausalNodeType.ENDOGENOUS)
    scm6.add_node("routing_weight", node_type=CausalNodeType.INTERVENTION)
    scm6.add_node("rtt_latency", node_type=CausalNodeType.OUTCOME)
    scm6.add_edge("link_congestion", "packet_loss", weight=0.8)
    scm6.add_edge("packet_loss", "rtt_latency", weight=0.7)
    scm6.add_edge("routing_weight", "link_congestion", weight=-0.6)
    scms_info.append((
        scm6, True, "Planted unobserved hardware degradation inducing spurious dependency between routing_weight and rtt_latency.",
        {"intervention": "routing_weight", "outcome": "rtt_latency", "effect": "reroute link"},
    ))

    # 7. kafka_consumer_lag (CORRUPTED: Broken conditional independence partition_backlog _||_ processing_delay | consumer_lag)
    scm7 = StructuralCausalModel("kafka_consumer_lag")
    scm7.add_node("produce_rate", node_type=CausalNodeType.EXOGENOUS)
    scm7.add_node("partition_backlog", node_type=CausalNodeType.ENDOGENOUS)
    scm7.add_node("assigned_workers", node_type=CausalNodeType.INTERVENTION)
    scm7.add_node("consumer_lag", node_type=CausalNodeType.ENDOGENOUS)
    scm7.add_node("processing_delay", node_type=CausalNodeType.OUTCOME)
    scm7.add_edge("produce_rate", "partition_backlog", weight=0.75)
    scm7.add_edge("partition_backlog", "consumer_lag", weight=0.8)
    scm7.add_edge("assigned_workers", "consumer_lag", weight=-0.7)
    scm7.add_edge("consumer_lag", "processing_delay", weight=0.85)
    scms_info.append((
        scm7, True, "Direct unmodeled disk queue coupling directly linking partition_backlog to processing_delay.",
        {"intervention": "assigned_workers", "outcome": "processing_delay", "effect": "scale workers"},
    ))

    # =============================================================
    # Domain 2: Financial & Governance (6 SCMs)
    # =============================================================
    # 8. fraud_scoring (Valid)
    scm8 = StructuralCausalModel("fraud_scoring")
    scm8.add_node("device_reputation", node_type=CausalNodeType.ENDOGENOUS)
    scm8.add_node("velocity_score", node_type=CausalNodeType.ENDOGENOUS)
    scm8.add_node("mfa_challenge_policy", node_type=CausalNodeType.INTERVENTION)
    scm8.add_node("fraud_probability", node_type=CausalNodeType.OUTCOME)
    scm8.add_edge("device_reputation", "fraud_probability", weight=-0.7)
    scm8.add_edge("velocity_score", "fraud_probability", weight=0.75)
    scm8.add_edge("mfa_challenge_policy", "fraud_probability", weight=-0.8)
    scms_info.append((
        scm8, False, None,
        {"intervention": "mfa_challenge_policy", "outcome": "fraud_probability", "effect": "suppress account takeover"},
    ))

    # 9. credit_default_risk (Valid)
    scm9 = StructuralCausalModel("credit_default_risk")
    scm9.add_node("debt_to_income", node_type=CausalNodeType.EXOGENOUS)
    scm9.add_node("credit_score", node_type=CausalNodeType.ENDOGENOUS)
    scm9.add_node("collateral_ratio", node_type=CausalNodeType.INTERVENTION)
    scm9.add_node("default_risk", node_type=CausalNodeType.OUTCOME)
    scm9.add_edge("debt_to_income", "credit_score", weight=-0.65)
    scm9.add_edge("credit_score", "default_risk", weight=-0.75)
    scm9.add_edge("collateral_ratio", "default_risk", weight=-0.7)
    scms_info.append((
        scm9, False, None,
        {"intervention": "collateral_ratio", "outcome": "default_risk", "effect": "mitigate credit loss"},
    ))

    # 10. kyc_verification_flow (Valid)
    scm10 = StructuralCausalModel("kyc_verification_flow")
    scm10.add_node("sanctions_screening", node_type=CausalNodeType.ENDOGENOUS)
    scm10.add_node("document_confidence", node_type=CausalNodeType.ENDOGENOUS)
    scm10.add_node("manual_review_decision", node_type=CausalNodeType.INTERVENTION)
    scm10.add_node("regulatory_approval", node_type=CausalNodeType.OUTCOME)
    scm10.add_edge("sanctions_screening", "regulatory_approval", weight=-0.9)
    scm10.add_edge("document_confidence", "manual_review_decision", weight=-0.6)
    scm10.add_edge("manual_review_decision", "regulatory_approval", weight=0.75)
    scms_info.append((
        scm10, False, None,
        {"intervention": "manual_review_decision", "outcome": "regulatory_approval", "effect": "complete onboarding"},
    ))

    # 11. settlement_reconciliation (CORRUPTED: Confounder U -> ledger_mismatch, dispute_flag)
    scm11 = StructuralCausalModel("settlement_reconciliation")
    scm11.add_node("transaction_volume", node_type=CausalNodeType.EXOGENOUS)
    scm11.add_node("ledger_mismatch", node_type=CausalNodeType.ENDOGENOUS)
    scm11.add_node("auto_reconcile_tolerance", node_type=CausalNodeType.INTERVENTION)
    scm11.add_node("break_amount", node_type=CausalNodeType.ENDOGENOUS)
    scm11.add_node("dispute_flag", node_type=CausalNodeType.OUTCOME)
    scm11.add_edge("transaction_volume", "ledger_mismatch", weight=0.6)
    scm11.add_edge("ledger_mismatch", "break_amount", weight=0.8)
    scm11.add_edge("auto_reconcile_tolerance", "break_amount", weight=-0.7)
    scm11.add_edge("break_amount", "dispute_flag", weight=0.75)
    scms_info.append((
        scm11, True, "Hidden exchange timing skew inducing unmodeled correlation between ledger_mismatch and dispute_flag.",
        {"intervention": "auto_reconcile_tolerance", "outcome": "dispute_flag", "effect": "resolve breaks"},
    ))

    # 12. trade_execution_slippage (CORRUPTED: Unobserved liquidity pool drain)
    scm12 = StructuralCausalModel("trade_execution_slippage")
    scm12.add_node("order_size", node_type=CausalNodeType.EXOGENOUS)
    scm12.add_node("orderbook_depth", node_type=CausalNodeType.ENDOGENOUS)
    scm12.add_node("smart_router_split", node_type=CausalNodeType.INTERVENTION)
    scm12.add_node("market_impact", node_type=CausalNodeType.ENDOGENOUS)
    scm12.add_node("realized_slippage", node_type=CausalNodeType.OUTCOME)
    scm12.add_edge("order_size", "orderbook_depth", weight=-0.6)
    scm12.add_edge("orderbook_depth", "market_impact", weight=-0.75)
    scm12.add_edge("smart_router_split", "market_impact", weight=-0.7)
    scm12.add_edge("market_impact", "realized_slippage", weight=0.85)
    scms_info.append((
        scm12, True, "Unobserved high-frequency latency arbitrage breaking smart_router_split independence from slippage.",
        {"intervention": "smart_router_split", "outcome": "realized_slippage", "effect": "minimize slippage"},
    ))

    # 13. insider_trading_signal (CORRUPTED: Direct mechanism severed)
    scm13 = StructuralCausalModel("insider_trading_signal")
    scm13.add_node("abnormal_call_volume", node_type=CausalNodeType.EXOGENOUS)
    scm13.add_node("options_skew", node_type=CausalNodeType.ENDOGENOUS)
    scm13.add_node("investigation_priority", node_type=CausalNodeType.INTERVENTION)
    scm13.add_node("finra_referral", node_type=CausalNodeType.OUTCOME)
    scm13.add_edge("abnormal_call_volume", "options_skew", weight=0.75)
    scm13.add_edge("options_skew", "investigation_priority", weight=0.8)
    scm13.add_edge("investigation_priority", "finra_referral", weight=0.85)
    scms_info.append((
        scm13, True, "Posited direct transmission options_skew -> investigation_priority absent in observational logs.",
        {"intervention": "investigation_priority", "outcome": "finra_referral", "effect": "escalate case"},
    ))

    # =============================================================
    # Domain 3: Biomedical & Clinical Care (7 SCMs)
    # =============================================================
    # 14. sepsis_early_warning (Valid)
    scm14 = StructuralCausalModel("sepsis_early_warning")
    scm14.add_node("infection_source", node_type=CausalNodeType.EXOGENOUS)
    scm14.add_node("inflammatory_response", node_type=CausalNodeType.ENDOGENOUS)
    scm14.add_node("antibiotic_timing_hours", node_type=CausalNodeType.INTERVENTION)
    scm14.add_node("lactate_clearance", node_type=CausalNodeType.ENDOGENOUS)
    scm14.add_node("septic_shock_mortality", node_type=CausalNodeType.OUTCOME)
    scm14.add_edge("infection_source", "inflammatory_response", weight=0.8)
    scm14.add_edge("inflammatory_response", "lactate_clearance", weight=-0.7)
    scm14.add_edge("antibiotic_timing_hours", "lactate_clearance", weight=-0.75)
    scm14.add_edge("lactate_clearance", "septic_shock_mortality", weight=-0.8)
    scms_info.append((
        scm14, False, None,
        {"intervention": "antibiotic_timing_hours", "outcome": "septic_shock_mortality", "effect": "accelerate therapy"},
    ))

    # 15. icu_mortality_risk (CORRUPTED: Broken conditional independence)
    scm15 = StructuralCausalModel("icu_mortality_risk")
    scm15.add_node("sofa_score", node_type=CausalNodeType.EXOGENOUS)
    scm15.add_node("multiorgan_failure", node_type=CausalNodeType.ENDOGENOUS)
    scm15.add_node("ventilator_peep_setting", node_type=CausalNodeType.INTERVENTION)
    scm15.add_node("arterial_po2", node_type=CausalNodeType.ENDOGENOUS)
    scm15.add_node("icu_mortality", node_type=CausalNodeType.OUTCOME)
    scm15.add_edge("sofa_score", "multiorgan_failure", weight=0.8)
    scm15.add_edge("multiorgan_failure", "icu_mortality", weight=0.75)
    scm15.add_edge("ventilator_peep_setting", "arterial_po2", weight=0.7)
    scm15.add_edge("arterial_po2", "icu_mortality", weight=-0.65)
    scms_info.append((
        scm15, True, "Hidden pre-existing COPD pathology coupling sofa_score directly to arterial_po2.",
        {"intervention": "ventilator_peep_setting", "outcome": "icu_mortality", "effect": "titrate PEEP"},
    ))

    # 16. drug_nephrotoxicity (CORRUPTED: Confounder U -> vancomycin_trough, acute_kidney_injury)
    scm16 = StructuralCausalModel("drug_nephrotoxicity")
    scm16.add_node("vancomycin_trough", node_type=CausalNodeType.INTERVENTION)
    scm16.add_node("serum_creatinine", node_type=CausalNodeType.ENDOGENOUS)
    scm16.add_node("glomerular_filtration_rate", node_type=CausalNodeType.ENDOGENOUS)
    scm16.add_node("acute_kidney_injury", node_type=CausalNodeType.OUTCOME)
    scm16.add_edge("vancomycin_trough", "serum_creatinine", weight=0.7)
    scm16.add_edge("serum_creatinine", "glomerular_filtration_rate", weight=-0.8)
    scm16.add_edge("glomerular_filtration_rate", "acute_kidney_injury", weight=-0.85)
    scms_info.append((
        scm16, True, "Omitted nephrotoxic NSAID co-administration confounding vancomycin_trough and acute_kidney_injury.",
        {"intervention": "vancomycin_trough", "outcome": "acute_kidney_injury", "effect": "adjust dosing"},
    ))

    # 17. glucose_insulin_dynamics (Valid)
    scm17 = StructuralCausalModel("glucose_insulin_dynamics")
    scm17.add_node("carbohydrate_intake", node_type=CausalNodeType.EXOGENOUS)
    scm17.add_node("plasma_glucose", node_type=CausalNodeType.ENDOGENOUS)
    scm17.add_node("insulin_infusion_units", node_type=CausalNodeType.INTERVENTION)
    scm17.add_node("target_glucose", node_type=CausalNodeType.OUTCOME)
    scm17.add_edge("carbohydrate_intake", "plasma_glucose", weight=0.8)
    scm17.add_edge("plasma_glucose", "target_glucose", weight=0.6)
    scm17.add_edge("insulin_infusion_units", "target_glucose", weight=-0.8)
    scms_info.append((
        scm17, False, None,
        {"intervention": "insulin_infusion_units", "outcome": "target_glucose", "effect": "stabilize glucose"},
    ))

    # 18. hypertensive_crisis (CORRUPTED: Posited arterial_stiffness link missing)
    scm18 = StructuralCausalModel("hypertensive_crisis")
    scm18.add_node("sodium_intake", node_type=CausalNodeType.EXOGENOUS)
    scm18.add_node("arterial_stiffness", node_type=CausalNodeType.ENDOGENOUS)
    scm18.add_node("antihypertensive_dose", node_type=CausalNodeType.INTERVENTION)
    scm18.add_node("blood_pressure_mmhg", node_type=CausalNodeType.OUTCOME)
    scm18.add_edge("sodium_intake", "arterial_stiffness", weight=0.7)
    scm18.add_edge("arterial_stiffness", "blood_pressure_mmhg", weight=0.8)
    scm18.add_edge("antihypertensive_dose", "blood_pressure_mmhg", weight=-0.85)
    scms_info.append((
        scm18, True, "Observational data exhibits zero correlation between sodium_intake and arterial_stiffness.",
        {"intervention": "antihypertensive_dose", "outcome": "blood_pressure_mmhg", "effect": "lower blood pressure"},
    ))

    # 19. antibiotic_resistance (CORRUPTED: Confounder U -> antibiotic_exposure, treatment_failure)
    scm19 = StructuralCausalModel("antibiotic_resistance")
    scm19.add_node("antibiotic_exposure", node_type=CausalNodeType.INTERVENTION)
    scm19.add_node("selective_pressure", node_type=CausalNodeType.ENDOGENOUS)
    scm19.add_node("resistance_gene_density", node_type=CausalNodeType.ENDOGENOUS)
    scm19.add_node("treatment_failure", node_type=CausalNodeType.OUTCOME)
    scm19.add_edge("antibiotic_exposure", "selective_pressure", weight=0.75)
    scm19.add_edge("selective_pressure", "resistance_gene_density", weight=0.8)
    scm19.add_edge("resistance_gene_density", "treatment_failure", weight=0.7)
    scms_info.append((
        scm19, True, "Hidden prior hospitalization duration confounding antibiotic_exposure with treatment_failure.",
        {"intervention": "antibiotic_exposure", "outcome": "treatment_failure", "effect": "optimize regimen"},
    ))

    # 20. postop_pulmonary_complications (CORRUPTED: Direct link atelectasis -> icu_readmission)
    scm20 = StructuralCausalModel("postop_pulmonary_complications")
    scm20.add_node("anesthesia_hours", node_type=CausalNodeType.EXOGENOUS)
    scm20.add_node("atelectasis_severity", node_type=CausalNodeType.ENDOGENOUS)
    scm20.add_node("incentive_spirometry", node_type=CausalNodeType.INTERVENTION)
    scm20.add_node("postop_hypoxemia", node_type=CausalNodeType.ENDOGENOUS)
    scm20.add_node("icu_readmission", node_type=CausalNodeType.OUTCOME)
    scm20.add_edge("anesthesia_hours", "atelectasis_severity", weight=0.75)
    scm20.add_edge("atelectasis_severity", "postop_hypoxemia", weight=0.8)
    scm20.add_edge("incentive_spirometry", "postop_hypoxemia", weight=-0.65)
    scm20.add_edge("postop_hypoxemia", "icu_readmission", weight=0.85)
    scms_info.append((
        scm20, True, "Uncontrolled pleural effusion directly driving icu_readmission independent of hypoxemia.",
        {"intervention": "incentive_spirometry", "outcome": "icu_readmission", "effect": "prevent readmission"},
    ))

    return scms_info


def generate_scm_data(
    scm: StructuralCausalModel,
    is_falsified: bool,
    n_samples: int = 500,
    seed: int = 42,
) -> list[dict[str, float]]:
    """Synthesize N observational tabular records from an SCM.

    If is_falsified is False:
        Data generated strictly in topological order using linear equations and Gaussian noise.
    If is_falsified is True:
        Planted hidden confounder U injects empirical dependency into DAG d-separated pairs.
    """
    rng = random.Random(seed)
    nodes = scm.topological_sort()

    confounder_pair: tuple[str, str] | None = None
    if is_falsified:
        implied = scm.find_implied_independencies(max_conditioning_size=2)
        uncond = [ci for ci in implied if len(ci["conditioning_set"]) == 0]
        if uncond:
            confounder_pair = (uncond[0]["var_x"], uncond[0]["var_y"])
        elif implied:
            confounder_pair = (implied[0]["var_x"], implied[0]["var_y"])
        elif len(nodes) >= 2:
            confounder_pair = (nodes[0], nodes[-1])

    data: list[dict[str, float]] = []

    for _ in range(n_samples):
        row: dict[str, float] = {}
        u = rng.gauss(0.0, 1.0) if is_falsified else 0.0

        for node_name in nodes:
            parents = scm.incoming_edges.get(node_name, [])
            if not parents:
                val = rng.gauss(10.0, 2.0)
            else:
                val = 0.0
                for edge in parents:
                    parent_val = row.get(edge.source, 10.0)
                    val += edge.weight * parent_val
                val += rng.gauss(0.0, 0.2)

            if is_falsified and confounder_pair and node_name in confounder_pair:
                val += 3.5 * u

            row[node_name] = round(val, 3)

        data.append(row)

    return data


def build_track_c_dataset() -> list[dict[str, Any]]:
    project_root = Path(__file__).parent
    models_dir = project_root / "models"
    data_dir = project_root / "data"
    models_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    scms_info = create_all_20_scms()
    manifest: list[dict[str, Any]] = []

    domain_mapping = {
        "cluster_autoscaling": "Operational",
        "db_connection_pool": "Operational",
        "api_circuit_breaker": "Operational",
        "cache_invalidation": "Operational",
        "deploy_canary": "Operational",
        "network_latency_routing": "Operational",
        "kafka_consumer_lag": "Operational",
        "fraud_scoring": "Financial",
        "credit_default_risk": "Financial",
        "kyc_verification_flow": "Financial",
        "settlement_reconciliation": "Financial",
        "trade_execution_slippage": "Financial",
        "insider_trading_signal": "Financial",
        "sepsis_early_warning": "Clinical",
        "icu_mortality_risk": "Clinical",
        "drug_nephrotoxicity": "Clinical",
        "glucose_insulin_dynamics": "Clinical",
        "hypertensive_crisis": "Clinical",
        "antibiotic_resistance": "Clinical",
        "postop_pulmonary_complications": "Clinical",
    }

    for idx, (scm, is_falsified, violation_detail, intervention_info) in enumerate(scms_info, start=1):
        domain = domain_mapping.get(scm.name, "Operational")
        case_id = f"TC-{domain[:3].upper()}-{idx:03d}"

        # 1. Export GraphML file
        graphml_path = models_dir / f"{scm.name}.graphml"
        scm.to_graphml(str(graphml_path))

        # 2. Generate and save observational data
        data = generate_scm_data(scm, is_falsified=is_falsified, n_samples=500, seed=42 + idx)
        data_path = data_dir / f"{scm.name}_data.json"
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        manifest.append({
            "case_id": case_id,
            "domain": domain,
            "scm_name": scm.name,
            "graphml_file": str(graphml_path.relative_to(project_root)),
            "data_file": str(data_path.relative_to(project_root)),
            "sample_size": len(data),
            "is_falsified": is_falsified,
            "falsified_mechanism_detail": violation_detail,
            "proposed_intervention": intervention_info,
            "expected_decision": "intercept" if is_falsified else "allow",
        })

    valid_cases = [m for m in manifest if not m["is_falsified"]]
    falsified_cases = [m for m in manifest if m["is_falsified"]]
    interleaved: list[dict[str, Any]] = []
    for v, f_case in zip(valid_cases, falsified_cases):
        interleaved.append(v)
        interleaved.append(f_case)

    return interleaved


def main() -> None:
    manifest = build_track_c_dataset()
    out_file = Path(__file__).parent / "dataset.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    valid_count = sum(1 for m in manifest if not m["is_falsified"])
    falsified_count = sum(1 for m in manifest if m["is_falsified"])
    print(f"Generated {len(manifest)} Track C benchmark cases at {out_file}")
    print(f"Valid SCMs: {valid_count}, Falsified SCMs: {falsified_count}")


if __name__ == "__main__":
    main()
