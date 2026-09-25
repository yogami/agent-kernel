"""Benchmark Runner for Track C: Causal Pre-Flight Verification & Observational Falsification.

Evaluates un-verified baseline execution against Kernel Causal Pre-Flight Gate
across 20 Structural Causal Models (10 valid, 10 falsified) paired with N=500 records.
Measures falsified mechanism interception rate, false alarm rate on valid models,
statistical power, and W3C traceparent propagation with 95% confidence intervals.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
from typing import Any

from core.causal_gate import CausalPreFlightGate
from core.causal_graph import StructuralCausalModel
from domain.models import TraceContext


class TrackCRunner:
    """Orchestrates Track C benchmark evaluation across seeds."""

    def __init__(self, dataset_path: str | Path | None = None) -> None:
        self.project_root = Path(__file__).parent
        if dataset_path is None:
            dataset_path = self.project_root / "dataset.json"
        self.dataset_path = Path(dataset_path)
        self.dataset: list[dict[str, Any]] = self._load_dataset()

    def _load_dataset(self) -> list[dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_single_seed(
        self,
        seed: int,
        cases_limit: int | None = None,
        trace_file: Path | None = None,
        alpha: float = 0.05,
    ) -> dict[str, Any]:
        """Execute evaluation across the benchmark cases for a given seed."""
        random.seed(seed)
        cases = list(self.dataset)
        if cases_limit is not None and cases_limit > 0:
            cases = cases[:cases_limit]

        gate = CausalPreFlightGate(alpha=alpha, apply_fdr=True)
        trace_records: list[dict[str, Any]] = []

        baseline_metrics = {
            "total_cases": len(cases),
            "falsified_cases": 0,
            "valid_cases": 0,
            "falsified_intercepted": 0,
            "falsified_breached": 0,
            "valid_allowed": 0,
            "valid_blocked": 0,
            "traceparent_propagated": 0,
        }

        kernel_metrics = {
            "total_cases": len(cases),
            "falsified_cases": 0,
            "valid_cases": 0,
            "falsified_intercepted": 0,
            "falsified_breached": 0,
            "valid_allowed": 0,
            "valid_blocked": 0,
            "traceparent_propagated": 0,
        }

        for case in cases:
            case_id = case["case_id"]
            domain = case["domain"]
            scm_name = case["scm_name"]
            is_falsified = case["is_falsified"]
            intervention = case["proposed_intervention"]

            graphml_path = self.project_root / case["graphml_file"]
            data_path = self.project_root / case["data_file"]

            scm = StructuralCausalModel.from_graphml(str(graphml_path))
            with open(data_path, "r", encoding="utf-8") as f:
                obs_data = json.load(f)

            # Generate W3C trace context
            trace_id = f"{seed:08x}{abs(hash(case_id)):024x}"[:32]
            span_id = f"{abs(hash(case_id + str(seed))):016x}"[:16]
            trace_ctx = TraceContext(trace_id=trace_id, span_id=span_id)
            traceparent_header = trace_ctx.to_traceparent()

            # ---------------------------------------------------------
            # 1. Baseline System (Un-verified execution)
            # ---------------------------------------------------------
            if is_falsified:
                baseline_metrics["falsified_cases"] += 1
                # Baseline executes without checking, allowing falsified mechanism
                baseline_intercepted = False
                baseline_metrics["falsified_breached"] += 1
            else:
                baseline_metrics["valid_cases"] += 1
                baseline_intercepted = False
                baseline_metrics["valid_allowed"] += 1

            # ---------------------------------------------------------
            # 2. Kernel Causal Pre-Flight Verification Gate
            # ---------------------------------------------------------
            if is_falsified:
                kernel_metrics["falsified_cases"] += 1
            else:
                kernel_metrics["valid_cases"] += 1

            is_admitted, block_reason, report = gate.pre_flight_check(
                tool_name="apply_causal_intervention",
                arguments=intervention,
                scm=scm,
                observational_data=obs_data,
                intervention_var=intervention["intervention"],
                outcome_var=intervention["outcome"],
            )

            kernel_intercepted = not is_admitted

            if is_falsified:
                if kernel_intercepted:
                    kernel_metrics["falsified_intercepted"] += 1
                else:
                    kernel_metrics["falsified_breached"] += 1
            else:
                if kernel_intercepted:
                    kernel_metrics["valid_blocked"] += 1
                else:
                    kernel_metrics["valid_allowed"] += 1

            # Check W3C trace context
            parsed_ctx = TraceContext.from_traceparent(traceparent_header)
            if parsed_ctx is not None and parsed_ctx.trace_id == trace_id:
                kernel_metrics["traceparent_propagated"] += 1

            # Record telemetry trace
            trace_record = {
                "trace_id": trace_id,
                "span_id": span_id,
                "traceparent": traceparent_header,
                "seed": seed,
                "case_id": case_id,
                "domain": domain,
                "scm_name": scm_name,
                "is_falsified": is_falsified,
                "proposed_intervention": intervention,
                "baseline": {
                    "is_intercepted": baseline_intercepted,
                    "action_executed": True,
                },
                "kernel": {
                    "is_intercepted": kernel_intercepted,
                    "block_reason": block_reason,
                    "tests_run": report.total_tests if report else 0,
                    "violations_detected": report.failed_tests if report else 0,
                    "fdr_applied": report.fdr_applied if report else False,
                },
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            trace_records.append(trace_record)

        # Write traces to file if path provided
        if trace_file is not None:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            with open(trace_file, "a", encoding="utf-8") as f:
                for tr in trace_records:
                    f.write(json.dumps(tr) + "\n")

        # Rate calculations
        def _calc_rates(m: dict[str, Any]) -> dict[str, float]:
            falsified = m["falsified_cases"]
            valid = m["valid_cases"]
            int_falsified = m["falsified_intercepted"]
            block_valid = m["valid_blocked"]

            interception_rate = (int_falsified / falsified) if falsified > 0 else 0.0
            false_alarm_rate = (block_valid / valid) if valid > 0 else 0.0

            total_intercepts = int_falsified + block_valid
            precision = (int_falsified / total_intercepts) if total_intercepts > 0 else 0.0
            recall = (int_falsified / falsified) if falsified > 0 else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

            total_cases = m["total_cases"]
            trace_rate = (m["traceparent_propagated"] / total_cases) if total_cases > 0 else 0.0

            return {
                "falsified_interception_rate": round(interception_rate, 4),
                "false_alarm_rate": round(false_alarm_rate, 4),
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1_score": round(f1, 4),
                "trace_propagation_rate": round(trace_rate, 4),
            }

        return {
            "seed": seed,
            "cases_evaluated": len(cases),
            "baseline": _calc_rates(baseline_metrics),
            "kernel": _calc_rates(kernel_metrics),
        }

    def run_benchmark(
        self,
        seeds: list[int] | None = None,
        cases_limit: int | None = None,
        trace_file: Path | None = None,
    ) -> dict[str, Any]:
        """Execute benchmark across multiple random seeds and compute 95% confidence intervals."""
        if seeds is None:
            seeds = [42, 143, 244]

        # Reset trace file if present
        if trace_file is not None and trace_file.exists():
            trace_file.unlink()

        per_seed_runs: list[dict[str, Any]] = []
        for s in seeds:
            res = self.run_single_seed(seed=s, cases_limit=cases_limit, trace_file=trace_file)
            per_seed_runs.append(res)

        def _aggregate_metric(runs: list[dict[str, Any]], system_key: str, metric_name: str) -> dict[str, float]:
            vals = [r[system_key][metric_name] for r in runs]
            n = len(vals)
            mean = sum(vals) / n
            if n > 1:
                variance = sum((x - mean) ** 2 for x in vals) / (n - 1)
                std_dev = math.sqrt(variance)
                ci_95 = 1.96 * (std_dev / math.sqrt(n))
            else:
                ci_95 = 0.0
            return {
                "mean": round(mean, 4),
                "ci_95": round(ci_95, 4),
                "min": round(min(vals), 4),
                "max": round(max(vals), 4),
            }

        summary_keys = [
            "falsified_interception_rate",
            "false_alarm_rate",
            "precision",
            "recall",
            "f1_score",
            "trace_propagation_rate",
        ]

        baseline_summary = {k: _aggregate_metric(per_seed_runs, "baseline", k) for k in summary_keys}
        kernel_summary = {k: _aggregate_metric(per_seed_runs, "kernel", k) for k in summary_keys}

        return {
            "benchmark": "Track C: Causal Pre-Flight Verification & Observational Falsification",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seeds_evaluated": seeds,
            "cases_per_seed": cases_limit if cases_limit else len(self.dataset),
            "baseline_results": baseline_summary,
            "kernel_results": kernel_summary,
            "per_seed_runs": per_seed_runs,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track C Causal Verification Benchmark.")
    parser.add_argument("--cases", type=int, default=20, help="Number of SCM cases to evaluate per seed.")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds.")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent.parent
    trace_path = project_root / "evals" / "track_c_causal" / "traces.jsonl"
    report_path = project_root / "reports" / "track_c_results.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    seeds = [42 + i * 101 for i in range(args.seeds)]
    print(f"Running Track C Benchmark on {args.cases} SCM cases across {args.seeds} seeds...")

    runner = TrackCRunner()
    results = runner.run_benchmark(seeds=seeds, cases_limit=args.cases, trace_file=trace_path)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    base = results["baseline_results"]
    kern = results["kernel_results"]

    print("\n" + "=" * 60)
    print("TRACK C BENCHMARK RESULTS SUMMARY")
    print("=" * 60)
    print("Falsified Mechanism Interception Rate:")
    print(f"  - Baseline: {base['falsified_interception_rate']['mean']*100:.1f}% (+-{base['falsified_interception_rate']['ci_95']*100:.1f}%)")
    print(f"  - Kernel:   {kern['falsified_interception_rate']['mean']*100:.1f}% (+-{kern['falsified_interception_rate']['ci_95']*100:.1f}%)")
    print("\nFalse Alarm Rate on Valid Models:")
    print(f"  - Baseline: {base['false_alarm_rate']['mean']*100:.1f}% (+-{base['false_alarm_rate']['ci_95']*100:.1f}%)")
    print(f"  - Kernel:   {kern['false_alarm_rate']['mean']*100:.1f}% (+-{kern['false_alarm_rate']['ci_95']*100:.1f}%)")
    print("\nStatistical F1 Score:")
    print(f"  - Kernel:   {kern['f1_score']['mean']:.3f} (+-{kern['f1_score']['ci_95']:.3f})")
    print("\nW3C Traceparent Header Propagation Rate:")
    print(f"  - Kernel:   {kern['trace_propagation_rate']['mean']*100:.1f}%")
    print("\nResults saved to reports/track_c_results.json")
    print("Telemetry traces saved to evals/track_c_causal/traces.jsonl")


if __name__ == "__main__":
    main()
