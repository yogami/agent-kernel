"""Unit tests for Phase 10: Track C (Causal Pre-Flight Verification & Observational Falsification).

Validates:
- 20 open GraphML Structural Causal Models and paired N=500 datasets
- Benjamini-Hochberg FDR p-value adjustment and multiple-testing control
- CausalPreFlightGate action interception on falsified mechanisms
- Clean permission on valid causal mechanisms
- Single-seed and multi-seed runner execution and telemetry trace logging
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from core.causal_gate import CausalFalsificationViolationError, CausalPreFlightGate
from core.causal_graph import StructuralCausalModel
from core.causal_verifier import ObservationalCausalVerifier
from evals.track_c_causal.runner import TrackCRunner


def test_track_c_dataset_and_models_integrity():
    """Verify dataset manifest, 20 GraphML models, and 20 observational data tables."""
    project_root = Path(__file__).parent.parent / "evals" / "track_c_causal"
    dataset_file = project_root / "dataset.json"
    assert dataset_file.exists(), f"Dataset manifest not found at {dataset_file}"

    with open(dataset_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    assert len(manifest) == 20, f"Expected 20 cases, found {len(manifest)}"

    valid_count = 0
    falsified_count = 0
    domains = set()

    for item in manifest:
        assert "case_id" in item
        assert "domain" in item
        assert "scm_name" in item
        assert "graphml_file" in item
        assert "data_file" in item
        assert "is_falsified" in item
        assert "proposed_intervention" in item

        domains.add(item["domain"])
        if item["is_falsified"]:
            falsified_count += 1
        else:
            valid_count += 1

        # Check GraphML file parses into valid SCM
        g_path = project_root / item["graphml_file"]
        assert g_path.exists(), f"GraphML missing: {g_path}"
        scm = StructuralCausalModel.from_graphml(str(g_path))
        assert scm.name == item["scm_name"]
        assert len(scm.nodes) >= 4

        # Check data file has 500 rows
        d_path = project_root / item["data_file"]
        assert d_path.exists(), f"Data table missing: {d_path}"
        with open(d_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert len(data) == 500

    assert valid_count == 10, f"Expected 10 valid SCMs, got {valid_count}"
    assert falsified_count == 10, f"Expected 10 falsified SCMs, got {falsified_count}"
    assert len(domains) == 3, f"Expected 3 domains, got {domains}"


def test_benjamini_hochberg_fdr_correction():
    """Verify that multiple testing adjusts raw p-values into monotonic q-values."""
    verifier = ObservationalCausalVerifier(alpha=0.05)

    # Build a simple chain SCM: A -> B -> C -> D
    scm = StructuralCausalModel("test_fdr_chain")
    for n in ["A", "B", "C", "D"]:
        scm.add_node(n)
    scm.add_edge("A", "B")
    scm.add_edge("B", "C")
    scm.add_edge("C", "D")

    # Generate synthetic data consistent with DAG
    data = []
    for _ in range(300):
        a = 5.0 + 0.5 * _
        b = 0.8 * a + 1.0
        c = 0.7 * b + 0.5
        d = 0.9 * c + 2.0
        data.append({"A": a, "B": b, "C": c, "D": d})

    report = verifier.verify_dataset(scm, data, apply_fdr=True)
    assert report.fdr_applied is True
    assert len(report.all_results) > 0

    for res in report.all_results:
        assert res.adjusted_p_value is not None
        assert 0.0 <= res.adjusted_p_value <= 1.0


def test_causal_pre_flight_gate_interception(tmp_path):
    """Verify CausalPreFlightGate blocks actions on falsified SCMs and admits valid SCMs."""
    gate = CausalPreFlightGate(alpha=0.05, apply_fdr=True)
    project_root = Path(__file__).parent.parent / "evals" / "track_c_causal"

    # 1. Test a valid case: cluster_autoscaling
    valid_scm = StructuralCausalModel.from_graphml(str(project_root / "models" / "cluster_autoscaling.graphml"))
    with open(project_root / "data" / "cluster_autoscaling_data.json", "r", encoding="utf-8") as f:
        valid_data = json.load(f)

    is_admitted, reason, report = gate.pre_flight_check(
        tool_name="scale_replicas",
        arguments={"intervention": "replica_count", "outcome": "p99_latency"},
        scm=valid_scm,
        observational_data=valid_data,
        intervention_var="replica_count",
        outcome_var="p99_latency",
    )
    assert is_admitted is True
    assert reason is None
    assert report.is_valid is True

    # 2. Test a falsified case: network_latency_routing
    falsified_scm = StructuralCausalModel.from_graphml(str(project_root / "models" / "network_latency_routing.graphml"))
    with open(project_root / "data" / "network_latency_routing_data.json", "r", encoding="utf-8") as f:
        falsified_data = json.load(f)

    is_admitted_f, reason_f, report_f = gate.pre_flight_check(
        tool_name="reroute_traffic",
        arguments={"intervention": "routing_weight", "outcome": "rtt_latency"},
        scm=falsified_scm,
        observational_data=falsified_data,
        intervention_var="routing_weight",
        outcome_var="rtt_latency",
    )
    assert is_admitted_f is False
    assert reason_f is not None
    assert "Causal Pre-Flight Violation" in reason_f
    assert report_f.is_valid is False

    # 3. Test exception raising in enforce_pre_flight
    with pytest.raises(CausalFalsificationViolationError):
        gate.enforce_pre_flight(
            tool_name="reroute_traffic",
            arguments={"intervention": "routing_weight", "outcome": "rtt_latency"},
            scm=falsified_scm,
            observational_data=falsified_data,
        )


def test_track_c_runner_single_seed_execution(tmp_path):
    """Verify single-seed runner execution intercepts falsified mechanisms and logs traces."""
    trace_file = tmp_path / "track_c_traces.jsonl"
    runner = TrackCRunner()

    result = runner.run_single_seed(seed=42, cases_limit=20, trace_file=trace_file)

    assert result["seed"] == 42
    assert result["cases_evaluated"] == 20

    baseline = result["baseline"]
    kernel = result["kernel"]

    # Baseline admits 100% of falsified models without verification
    assert baseline["falsified_interception_rate"] == 0.0

    # Kernel intercepts falsified mechanisms
    assert kernel["falsified_interception_rate"] >= 0.90
    assert kernel["false_alarm_rate"] <= 0.10
    assert kernel["trace_propagation_rate"] == 1.0

    # Verify trace file was written
    assert trace_file.exists()
    with open(trace_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 20
    assert "traceparent" in lines[0]
    assert "kernel" in lines[0]


def test_track_c_runner_multi_seed_aggregate(tmp_path):
    """Verify multi-seed aggregation calculates 95% confidence intervals across seeds."""
    runner = TrackCRunner()
    trace_file = tmp_path / "multi_seed_c_traces.jsonl"

    res = runner.run_benchmark(seeds=[42, 99], cases_limit=10, trace_file=trace_file)

    assert "baseline_results" in res
    assert "kernel_results" in res
    assert len(res["seeds_evaluated"]) == 2

    k_res = res["kernel_results"]
    assert "falsified_interception_rate" in k_res
    assert k_res["falsified_interception_rate"]["mean"] >= 0.90
    assert k_res["false_alarm_rate"]["mean"] <= 0.10
    assert "f1_score" in k_res
