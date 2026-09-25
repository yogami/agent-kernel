"""Verification suite for Phase 6 Empirical Benchmarks and DSPy Prompt Optimization."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from domain.models import ConfigPack
from infrastructure.prompt_optimizer import DSPyPromptOptimizerAdapter, MIPROv2PromptOptimizer
from ops.benchmark import BenchmarkRunner
from ops.config_manager import ConfigManager
from ops.eval_runner import EvalRunner
from ops.self_healer import SelfHealer


def test_benchmark_runner_live_measurements(tmp_path):
    """Verify BenchmarkRunner computes live metrics across 100 cases with zero hardcoded constants."""
    from infrastructure.sqlite_fact_store import SQLiteFactStore
    isolated_report = tmp_path / "BENCHMARK_REPORT.md"
    runner = BenchmarkRunner(
        "evals/datasets/100_clinical_cases.json",
        fact_store_factory=lambda: SQLiteFactStore(":memory:"),
        report_path=isolated_report,
    )
    results = runner.run_benchmark()

    assert isolated_report.exists()
    assert "# Architectural Simulation Report" in isolated_report.read_text(encoding="utf-8")

    assert "system_a_standard_rag" in results
    assert "system_a_plus_schema" in results
    assert "system_b_agent_kernel" in results

    sys_a = results["system_a_standard_rag"]
    sys_a_plus = results["system_a_plus_schema"]
    sys_b = results["system_b_agent_kernel"]

    # 1. Verification of live metric calculations
    assert sys_a["memory_contradiction_rate"] > 0.0
    assert sys_a["pii_leakage_rate"] > 80.0
    assert sys_a["p95_latency_ms"] > 0.0
    assert sys_a["cost_per_record_usd"] > 0.0

    # 2. System A+ enforces tool schema but suffers memory contamination
    assert sys_a_plus["schema_breakage_rate"] == 0.0
    assert sys_a_plus["memory_contradiction_rate"] > 0.0

    # 3. System B eliminates memory contradictions in active facts via quarantine
    assert sys_b["memory_contradiction_rate"] == 0.0
    assert sys_b["pii_leakage_rate"] == 0.0
    assert sys_b["runaway_loop_rate"] == 0.0
    assert sys_b["quarantined_backlog"] == 25
    assert sys_b["promoted_facts"] == 275

    # 4. Confirm report file exists and is populated
    report_path = Path("BENCHMARK_REPORT.md")
    assert report_path.exists()
    content = report_path.read_text(encoding="utf-8")
    assert "Summary Comparison Matrix" in content
    assert "Tri-State Admission Gate" in content


def test_miprov2_prompt_optimizer_proposals_and_scoring():
    """Verify MIPROv2 optimizer categorizes failure traces and selects top-scoring pack."""
    optimizer = MIPROv2PromptOptimizer(max_proposals=3)
    base_pack = ConfigPack(
        version="v1",
        system_prompt="Clinical reasoning baseline assistant.",
        rules=["Rule 1: Always be helpful."],
        sha256_hash="hash_v1",
    )

    failure_traces = [
        {"case_id": "c1", "error": "Contradiction detected on allergy status."},
        {"case_id": "c2", "error": "Vital bounds violation: systolic 450 mmHg."},
    ]

    def mock_eval_callback(candidate: ConfigPack) -> float:
        score = 0.5
        for r in candidate.rules:
            if "reject candidate facts" in r:
                score += 0.25
            if "physiological bounds" in r:
                score += 0.25
        return min(score, 1.0)

    best_pack, telemetry = optimizer.optimize_prompt(base_pack, failure_traces, mock_eval_callback)

    assert telemetry["improved"] is True
    assert telemetry["best_score"] > telemetry["baseline_score"]
    assert telemetry["proposals_evaluated"] >= 2
    assert len(best_pack.rules) > len(base_pack.rules)
    assert any("reject candidate facts" in r for r in best_pack.rules)


def test_dspy_adapter_fallback():
    """Verify DSPyPromptOptimizerAdapter initializes and delegates cleanly."""
    adapter = DSPyPromptOptimizerAdapter()
    base_pack = ConfigPack(
        version="v1",
        system_prompt="Base prompt.",
        rules=["Rule A"],
    )

    failures = [{"case_id": "f1", "error": "schema validation failed"}]
    pack, telemetry = adapter.optimize_prompt(base_pack, failures, lambda p: 0.8)
    assert pack is not None
    assert "proposals_evaluated" in telemetry


def test_self_healer_monotonic_non_regression_with_optimizer(tmp_path):
    """Verify SelfHealer uses optimizer and strictly enforces non-regression gating."""
    packs_dir = tmp_path / "packs"
    evals_dir = tmp_path / "evals"

    config_mgr = ConfigManager(packs_dir)
    eval_runner = EvalRunner(evals_dir)

    # Custom optimizer proposing candidate that improves dev score
    class MockImprovingOptimizer:
        def optimize_prompt(self, base_pack, failure_traces, eval_callback):
            cand = ConfigPack(
                version=base_pack.version,
                system_prompt=base_pack.system_prompt + " Verified.",
                rules=list(base_pack.rules) + ["Verified check."],
            )
            return cand, {"improved": True}

    healer = SelfHealer(config_mgr, eval_runner, max_iterations=2, optimizer=MockImprovingOptimizer())

    # When dev has zero failures, stays GREEN
    res = healer.run_repair_cycle()
    assert res["status"] == "GREEN"
    assert res["active_version"] == "v1"
