"""Unit tests for Phase 8: Track M (Write-Path Memory Integrity Benchmark).

Validates:
- Dataset schema, 200 cases balance, domain coverage
- FrozenNLIClassifier precision and deterministic behavior
- Single-seed and multi-seed runner execution
- Contradiction admission suppression and downstream factuality improvement
- JSONL telemetry trace generation
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from domain.models import NLILabel
from evals.track_m_memory.nli_evaluator import FrozenNLIClassifier
from evals.track_m_memory.runner import TrackMRunner


def test_track_m_dataset_integrity_and_distribution():
    """Verify dataset contains 200 well-structured cases evenly distributed across 4 domains."""
    dataset_file = Path(__file__).parent.parent / "evals" / "track_m_memory" / "dataset.json"
    assert dataset_file.exists(), f"Dataset file not found at {dataset_file}"

    with open(dataset_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert len(data) == 200, f"Expected 200 cases, found {len(data)}"

    domains = {}
    for case in data:
        assert "case_id" in case
        assert "domain" in case
        assert "initial_facts" in case
        assert "candidate_updates" in case
        assert "downstream_probe" in case

        dom = case["domain"]
        domains[dom] = domains.get(dom, 0) + 1

        # Check candidate ground truth
        for cand in case["candidate_updates"]:
            assert "is_contradiction" in cand
            assert "is_legitimate_update" in cand
            assert cand["expected_action"] in {"admit", "quarantine", "reject"}

        # Check probe
        probe = case["downstream_probe"]
        assert "query" in probe
        assert "expected_answer" in probe
        assert "unacceptable_corrupted_answer" in probe

    # 50 cases in each of the 4 domains
    assert len(domains) == 4
    for dom, count in domains.items():
        assert count == 50, f"Domain '{dom}' has {count} cases, expected 50"


def test_frozen_nli_classifier_cross_domain_accuracy():
    """Verify FrozenNLIClassifier catches semantic and polarity contradictions across domains."""
    clf = FrozenNLIClassifier(model_id="test/deberta-v3-large-mnli")

    # 1. Identity & Access: active vs suspended
    r_iam = clf.classify_pair(
        "Subject user_101 has account_status active.",
        "Subject user_101 has account_status suspended_fraud_investigation.",
    )
    assert r_iam.label == NLILabel.CONTRADICTION
    assert r_iam.contradiction_score >= 0.90

    # 2. Clinical: normotensive vs hypertension
    r_cln = clf.classify_pair(
        "Subject patient_201 has blood_pressure_status normotensive.",
        "Subject patient_201 has blood_pressure_status malignant_hypertension_crisis.",
    )
    assert r_cln.label == NLILabel.CONTRADICTION
    assert r_cln.contradiction_score >= 0.90

    # 3. DevOps: restricted internal only vs unrestricted public internet
    r_ops = clf.classify_pair(
        "Subject svc_api has network_egress_policy restricted_internal_only.",
        "Subject svc_api has network_egress_policy unrestricted_public_internet.",
    )
    assert r_ops.label == NLILabel.CONTRADICTION
    assert r_ops.contradiction_score >= 0.90

    # 4. Financial: cleared vs sanctioned
    r_fin = clf.classify_pair(
        "Subject corp_401 has sanctions_screening cleared_no_match.",
        "Subject corp_401 has sanctions_screening flagged_sdn_sanctioned_entity.",
    )
    assert r_fin.label == NLILabel.CONTRADICTION
    assert r_fin.contradiction_score >= 0.90

    # 5. Non-contradictory update: entailment / neutral
    r_neutral = clf.classify_pair(
        "Subject user_101 has clearance_level Secret.",
        "Subject user_101 has annual_review completed.",
    )
    assert r_neutral.label in {NLILabel.NEUTRAL, NLILabel.ENTAILMENT}
    assert r_neutral.contradiction_score < 0.30


def test_track_m_runner_single_seed_execution(tmp_path):
    """Verify single-seed runner execution correctly suppresses contradictions in Kernel vs Baseline."""
    trace_file = tmp_path / "test_traces.jsonl"
    runner = TrackMRunner()

    # Run on subset of 20 cases for fast unit testing
    result = runner.run_single_seed(seed=42, cases_limit=20, trace_file=trace_file)

    assert result["seed"] == 42
    assert result["cases_evaluated"] == 20

    baseline = result["baseline"]
    kernel = result["kernel"]

    # Baseline admits 100% of contradictions
    assert baseline["contradiction_admission_rate"] == 1.0

    # Kernel blocks virtually all contradictions
    assert kernel["contradiction_admission_rate"] <= 0.10
    assert kernel["quarantine_precision"] >= 0.85
    assert kernel["quarantine_recall"] >= 0.85

    # Kernel downstream factuality significantly exceeds baseline
    assert kernel["downstream_factuality"] > baseline["downstream_factuality"]

    # Verify trace file was written
    assert trace_file.exists()
    with open(trace_file, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) > 0
    assert "trace_id" in lines[0]
    assert "kernel_decision" in lines[0]


def test_track_m_runner_multi_seed_aggregate(tmp_path):
    """Verify multi-seed aggregation calculates 95% confidence intervals and summary metrics."""
    runner = TrackMRunner()
    trace_file = tmp_path / "multi_seed_traces.jsonl"

    res = runner.run_benchmark(seeds=[42, 99], cases_limit=15, trace_file=trace_file)

    assert "baseline_results" in res
    assert "kernel_results" in res
    assert len(res["seeds_evaluated"]) == 2

    k_res = res["kernel_results"]
    assert "contradiction_admission_rate" in k_res
    assert "ci_95" in k_res["contradiction_admission_rate"]
    assert "downstream_factuality" in k_res
    assert k_res["downstream_factuality"]["mean"] >= 0.80
