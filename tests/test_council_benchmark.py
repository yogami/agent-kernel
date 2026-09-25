"""Unit tests for Unified Council Benchmark Harness and McNemar statistical tests."""

from __future__ import annotations

import math
from pathlib import Path
import pytest

from ops.mcnemar import calculate_mcnemar_test, McNemarResult
from ops.run_council_bench import CouncilBenchmarkRunner, compute_ci95, format_markdown_report


def test_mcnemar_exact_calculation_known_values() -> None:
    """Validate McNemar test implementation against textbook reference pairs."""
    # Small n (< 25): b=12, c=1 -> n=13. Exact two-sided binomial p = 2 * (1 + 13) / 2^13 = 28 / 8192 = 0.00341796875
    a_small = [True] * 12 + [False] * 1
    b_small = [False] * 12 + [True] * 1
    res_small = calculate_mcnemar_test(a_small, b_small)

    assert res_small.discordant_a_passes_b_fails == 12
    assert res_small.discordant_a_fails_b_passes == 1
    assert res_small.total_discordant == 13
    assert abs(res_small.p_value - 0.003418) < 1e-5
    assert res_small.is_statistically_significant is True
    assert "Exact Binomial Test" in res_small.method

    # Large n (>= 25): b=40, c=10 -> n=50. (|40 - 10| - 1)^2 / 50 = 29^2 / 50 = 841 / 50 = 16.82
    a_large = [True] * 40 + [False] * 10
    b_large = [False] * 40 + [True] * 10
    res_large = calculate_mcnemar_test(a_large, b_large)

    assert res_large.discordant_a_passes_b_fails == 40
    assert res_large.discordant_a_fails_b_passes == 10
    assert res_large.total_discordant == 50
    assert abs(res_large.test_statistic - 16.82) < 1e-4
    assert res_large.p_value < 0.0001
    assert res_large.is_statistically_significant is True
    assert "Edwards Continuity-Corrected" in res_large.method

    # Zero discordant pairs: b=0, c=0
    res_zero = calculate_mcnemar_test([True, False], [True, False])
    assert res_zero.total_discordant == 0
    assert res_zero.p_value == 1.0
    assert res_zero.is_statistically_significant is False


def test_mcnemar_result_schema_and_serialization() -> None:
    """Check that McNemarResult serializes correctly into dictionary format."""
    res = calculate_mcnemar_test([True, True, False], [False, True, True])
    d = res.to_dict()

    assert "contingency_table" in d
    assert "discordant_b" in d
    assert "discordant_c" in d
    assert "total_discordant" in d
    assert "test_statistic" in d
    assert "p_value" in d
    assert "is_statistically_significant" in d
    assert "odds_ratio" in d
    assert "method" in d
    assert "alpha" in d


def test_compute_ci95_calculation() -> None:
    """Validate 95% confidence interval calculations."""
    vals = [0.95, 0.96, 0.94]
    ci = compute_ci95(vals)
    assert ci["mean"] == 0.95
    assert ci["min"] == 0.94
    assert ci["max"] == 0.96
    assert ci["ci_95"] > 0.0

    # Single value check
    single = compute_ci95([0.80])
    assert single["mean"] == 0.80
    assert single["ci_95"] == 0.0


def test_factorial_matrix_runner_single_seed_execution() -> None:
    """Verify that CouncilBenchmarkRunner executes single-seed fast pass across all 5 configurations."""
    runner = CouncilBenchmarkRunner()
    res = runner.run_case_level_evaluation(seed=42, m_cases=10, t_cases=10, c_cases=4)

    assert res["seed"] == 42
    assert res["total_cases"] == 24
    assert "config_metrics" in res

    configs = res["config_metrics"]
    expected_configs = [
        "raw_baseline",
        "kernel_no_nli",
        "kernel_no_confinement",
        "kernel_no_causal",
        "full_kernel",
    ]
    for cfg in expected_configs:
        assert cfg in configs
        assert 0.0 <= configs[cfg]["overall_accuracy"] <= 1.0

    # Full kernel should reach near-perfect accuracy
    assert configs["full_kernel"]["overall_accuracy"] == 1.0
    # Raw baseline should have significantly lower accuracy
    assert configs["raw_baseline"]["overall_accuracy"] < configs["full_kernel"]["overall_accuracy"]

    # Open 8B with Kernel should beat Raw Frontier
    assert res["kernel_8b_accuracy"] > res["raw_frontier_accuracy"]
    assert res["mcnemar"].p_value < 0.05


def test_ablation_degradation_signatures() -> None:
    """Verify that ablating specific components produces expected failure signatures."""
    runner = CouncilBenchmarkRunner()
    res = runner.run_case_level_evaluation(seed=42, m_cases=10, t_cases=10, c_cases=4)
    configs = res["config_metrics"]

    # kernel_no_nli: Track M drops to 0.0, Track T and C remain 1.0
    assert configs["kernel_no_nli"]["track_m_pass_rate"] == 0.0
    assert configs["kernel_no_nli"]["track_t_pass_rate"] == 1.0
    assert configs["kernel_no_nli"]["track_c_pass_rate"] == 1.0

    # kernel_no_confinement: Track T drops below 1.0, Track M remains 1.0
    assert configs["kernel_no_confinement"]["track_m_pass_rate"] == 1.0
    assert configs["kernel_no_confinement"]["track_t_pass_rate"] < 1.0
    assert configs["kernel_no_confinement"]["track_c_pass_rate"] == 1.0

    # kernel_no_causal: Track C drops below 1.0, Track M and T remain 1.0
    assert configs["kernel_no_causal"]["track_m_pass_rate"] == 1.0
    assert configs["kernel_no_causal"]["track_t_pass_rate"] == 1.0
    assert configs["kernel_no_causal"]["track_c_pass_rate"] < 1.0


def test_markdown_report_formatting() -> None:
    """Verify that format_markdown_report produces clean, compliant Markdown."""
    runner = CouncilBenchmarkRunner()
    benchmark_data = runner.run_benchmark(seeds=2, m_cases=10, t_cases=10, c_cases=4)
    md_output = format_markdown_report(benchmark_data)

    assert "# 📊 agent_kernel-bench: Verified Benchmark Report" in md_output
    assert "McNemar Paired Exact Hypothesis Test" in md_output
    assert "Factorial Ablation Matrix" in md_output
    assert "Track M: Write-Path Memory Integrity" in md_output
    assert "Track T: Tool Policy & Confinement" in md_output
    assert "Track C: Causal Pre-Flight Verification" in md_output

    # Verify no em-dashes
    assert "—" not in md_output
    assert " -- " not in md_output
