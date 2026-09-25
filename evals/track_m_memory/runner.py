"""Benchmark Runner for Track M: Write-Path Memory Integrity & Admission Gate.

Evaluates un-gated baseline memory writes against the Tri-State Memory Admission Gate
across 200 longitudinal dialogue streams with planted temporal contradictions.
Measures contradiction admission rate, true-update acceptance, false-quarantine rate,
quarantine precision/recall, and downstream factual purity with 95% confidence intervals.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import random
from typing import Any

from domain.models import AdmissionStatus, CandidateFact, SemanticFact
from infrastructure.sqlite_fact_store import SQLiteFactStore
from core.memory_promoter import MemoryPromoter
from evals.track_m_memory.nli_evaluator import FrozenNLIClassifier


class TrackMRunner:
    """Orchestrates multi-seed Track M benchmark runs and calculates empirical metrics."""

    def __init__(
        self,
        dataset_path: str | Path | None = None,
        nli_classifier: FrozenNLIClassifier | None = None,
    ) -> None:
        if dataset_path is None:
            dataset_path = Path(__file__).parent / "dataset.json"
        self.dataset_path = Path(dataset_path)
        self.nli_classifier = nli_classifier or FrozenNLIClassifier()
        self.dataset: list[dict[str, Any]] = self._load_dataset()

    def _load_dataset(self) -> list[dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def run_single_seed(
        self,
        seed: int,
        cases_limit: int | None = None,
        trace_file: Path | None = None,
    ) -> dict[str, Any]:
        """Execute one complete evaluation pass across the dataset for a given seed."""
        random.seed(seed)
        cases = list(self.dataset)
        if cases_limit is not None and cases_limit > 0:
            cases = cases[:cases_limit]

        # Setup Baseline Store (Un-gated Direct Writes)
        baseline_store = SQLiteFactStore(":memory:")

        # Setup Kernel Store (Tri-State Admission Gate)
        kernel_store = SQLiteFactStore(":memory:")
        promoter = MemoryPromoter(
            fact_store=kernel_store,
            nli_provider=self.nli_classifier,
            min_confidence=0.75,
        )

        trace_records: list[dict[str, Any]] = []

        baseline_metrics = {
            "total_candidates": 0,
            "planted_contradictions": 0,
            "legitimate_updates": 0,
            "contradictions_admitted": 0,
            "contradictions_quarantined": 0,
            "legitimate_admitted": 0,
            "legitimate_quarantined": 0,
            "downstream_correct": 0,
        }

        kernel_metrics = {
            "total_candidates": 0,
            "planted_contradictions": 0,
            "legitimate_updates": 0,
            "contradictions_admitted": 0,
            "contradictions_quarantined": 0,
            "legitimate_admitted": 0,
            "legitimate_quarantined": 0,
            "downstream_correct": 0,
        }

        for case in cases:
            case_id = case["case_id"]
            domain = case["domain"]
            initial_facts = case["initial_facts"]
            candidate_updates = case["candidate_updates"]
            probe = case["downstream_probe"]

            # 1. Initialize Baseline and Kernel with confirmed base facts
            for fact in initial_facts:
                now = datetime.now(timezone.utc)
                base_fact_b = SemanticFact(
                    fact_id=f"base_b_{case_id}_{fact['subject']}_{fact['predicate']}",
                    candidate_id=f"cand_init_{case_id}",
                    source_episode_id=f"ep_init_{case_id}",
                    session_id=case_id,
                    tenant_id=domain,
                    subject=fact["subject"],
                    predicate=fact["predicate"],
                    object=fact["object"],
                    confidence=fact["confidence"],
                    valid_from=now,
                    is_active=True,
                    promoted_at=now,
                )
                base_fact_k = SemanticFact(
                    fact_id=f"base_k_{case_id}_{fact['subject']}_{fact['predicate']}",
                    candidate_id=f"cand_init_{case_id}",
                    source_episode_id=f"ep_init_{case_id}",
                    session_id=case_id,
                    tenant_id=domain,
                    subject=fact["subject"],
                    predicate=fact["predicate"],
                    object=fact["object"],
                    confidence=fact["confidence"],
                    valid_from=now,
                    is_active=True,
                    promoted_at=now,
                )
                baseline_store.insert_semantic_fact(base_fact_b)
                kernel_store.insert_semantic_fact(base_fact_k)

            # 2. Process incoming candidate updates
            for cand_data in candidate_updates:
                is_contradiction = cand_data["is_contradiction"]
                is_legit = cand_data["is_legitimate_update"]

                baseline_metrics["total_candidates"] += 1
                kernel_metrics["total_candidates"] += 1

                if is_contradiction:
                    baseline_metrics["planted_contradictions"] += 1
                    kernel_metrics["planted_contradictions"] += 1
                elif is_legit:
                    baseline_metrics["legitimate_updates"] += 1
                    kernel_metrics["legitimate_updates"] += 1

                # Execution A: Baseline System (Un-gated direct write)
                cand_id_b = f"cand_b_{case_id}_{len(trace_records)}"
                direct_fact = SemanticFact(
                    fact_id=f"fact_b_{case_id}_{len(trace_records)}",
                    candidate_id=cand_id_b,
                    source_episode_id=f"ep_{case_id}",
                    session_id=case_id,
                    tenant_id=domain,
                    subject=cand_data["subject"],
                    predicate=cand_data["predicate"],
                    object=cand_data["object"],
                    confidence=cand_data["confidence"],
                    valid_from=datetime.now(timezone.utc),
                    is_active=True,
                    promoted_at=datetime.now(timezone.utc),
                )
                baseline_store.insert_semantic_fact(direct_fact)

                # Baseline always admits directly into active memory
                if is_contradiction:
                    baseline_metrics["contradictions_admitted"] += 1
                elif is_legit:
                    baseline_metrics["legitimate_admitted"] += 1

                # Execution B: Kernel System (Tri-State Memory Gate)
                cand_obj = CandidateFact(
                    candidate_id=f"cand_k_{case_id}_{len(trace_records)}",
                    source_episode_id=f"ep_{case_id}",
                    session_id=case_id,
                    tenant_id=domain,
                    subject=cand_data["subject"],
                    predicate=cand_data["predicate"],
                    object=cand_data["object"],
                    confidence=cand_data["confidence"],
                )
                promoter.submit_candidate(cand_obj)
                promoted, decision_reason = promoter.evaluate_and_promote(cand_obj.candidate_id)

                if promoted:
                    if is_contradiction:
                        kernel_metrics["contradictions_admitted"] += 1
                    elif is_legit:
                        kernel_metrics["legitimate_admitted"] += 1
                else:
                    # Fact was rejected or quarantined
                    if is_contradiction:
                        kernel_metrics["contradictions_quarantined"] += 1
                    elif is_legit:
                        kernel_metrics["legitimate_quarantined"] += 1

                # Record JSONL telemetry trace
                trace_records.append({
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "seed": seed,
                    "case_id": case_id,
                    "domain": domain,
                    "candidate": {
                        "subject": cand_data["subject"],
                        "predicate": cand_data["predicate"],
                        "object": cand_data["object"],
                        "confidence": cand_data["confidence"],
                    },
                    "ground_truth": {
                        "is_contradiction": is_contradiction,
                        "is_legitimate_update": is_legit,
                        "expected_action": cand_data["expected_action"],
                    },
                    "kernel_decision": {
                        "promoted": promoted,
                        "reason": decision_reason,
                    },
                    "trace_id": f"trace_{seed}_{case_id}",
                })

            # 3. Downstream Factual Retrieval Probe
            # Baseline probe: query active facts for probe subject
            subj = initial_facts[0]["subject"]
            base_facts = baseline_store.get_active_facts_for_subject(subj, tenant_id=domain)
            kern_facts = kernel_store.get_active_facts_for_subject(subj, tenant_id=domain)

            # In baseline, if contradiction leaked, corrupted answer occurs
            has_leak_b = any(
                any(c["object"].lower() in bf.object.lower() for c in candidate_updates if c["is_contradiction"])
                for bf in base_facts
            )
            if not has_leak_b:
                baseline_metrics["downstream_correct"] += 1

            # In kernel, check if contradiction leaked
            has_leak_k = any(
                any(c["object"].lower() in kf.object.lower() for c in candidate_updates if c["is_contradiction"])
                for kf in kern_facts
            )
            if not has_leak_k:
                kernel_metrics["downstream_correct"] += 1

        # Optionally append to trace file
        if trace_file:
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            with open(trace_file, "a", encoding="utf-8") as f:
                for tr in trace_records:
                    f.write(json.dumps(tr) + "\n")

        # Compute rates for this seed
        return {
            "seed": seed,
            "cases_evaluated": len(cases),
            "baseline": self._compute_rates(baseline_metrics, len(cases)),
            "kernel": self._compute_rates(kernel_metrics, len(cases)),
        }

    @staticmethod
    def _compute_rates(m: dict[str, int], total_cases: int) -> dict[str, float]:
        contra_total = max(1, m["planted_contradictions"])
        legit_total = max(1, m["legitimate_updates"])

        contra_admitted_rate = m["contradictions_admitted"] / contra_total
        true_accept_rate = m["legitimate_admitted"] / legit_total
        false_quarantine_rate = m["legitimate_quarantined"] / legit_total

        # Precision & recall of quarantine
        tp = m["contradictions_quarantined"]
        fp = m["legitimate_quarantined"]
        fn = m["contradictions_admitted"]

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-9, prec + rec)
        downstream_factuality = m["downstream_correct"] / max(1, total_cases)

        return {
            "contradiction_admission_rate": contra_admitted_rate,
            "true_update_acceptance_rate": true_accept_rate,
            "false_quarantine_rate": false_quarantine_rate,
            "quarantine_precision": prec,
            "quarantine_recall": rec,
            "quarantine_f1": f1,
            "downstream_factuality": downstream_factuality,
            "raw_counts": m,
        }

    def run_benchmark(
        self,
        seeds: list[int] | None = None,
        cases_limit: int | None = None,
        trace_file: Path | None = None,
    ) -> dict[str, Any]:
        """Run benchmark across multiple seeds and compute mean +/- 95% confidence intervals."""
        if seeds is None:
            seeds = [42, 123, 999]

        # Reset trace file if present
        if trace_file and trace_file.exists():
            trace_file.unlink()

        seed_results = [
            self.run_single_seed(s, cases_limit=cases_limit, trace_file=trace_file)
            for s in seeds
        ]

        def aggregate_metrics(system_key: str) -> dict[str, Any]:
            keys = [
                "contradiction_admission_rate",
                "true_update_acceptance_rate",
                "false_quarantine_rate",
                "quarantine_precision",
                "quarantine_recall",
                "quarantine_f1",
                "downstream_factuality",
            ]
            agg = {}
            k_len = len(seeds)
            for key in keys:
                vals = [sr[system_key][key] for sr in seed_results]
                mean = sum(vals) / k_len
                variance = sum((v - mean) ** 2 for v in vals) / max(1, k_len - 1)
                std_err = math.sqrt(variance) / math.sqrt(k_len)
                ci_95 = 1.96 * std_err if k_len > 1 else 0.0

                agg[key] = {
                    "mean": round(mean, 4),
                    "ci_95": round(ci_95, 4),
                    "min": round(min(vals), 4),
                    "max": round(max(vals), 4),
                }
            return agg

        return {
            "benchmark": "Track M: Write-Path Memory Integrity",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seeds_evaluated": seeds,
            "cases_per_seed": seed_results[0]["cases_evaluated"],
            "baseline_results": aggregate_metrics("baseline"),
            "kernel_results": aggregate_metrics("kernel"),
            "seed_breakdown": seed_results,
        }


def main():
    parser = argparse.ArgumentParser(description="Run Track M Memory Integrity Benchmark")
    parser.add_argument("--cases", type=int, default=200, help="Number of cases to evaluate (default: 200)")
    parser.add_argument("--seeds", type=int, default=3, help="Number of random seeds (default: 3)")
    parser.add_argument("--output", type=str, default="reports/track_m_results.json", help="Path to save results JSON")
    parser.add_argument("--traces", type=str, default="evals/track_m_memory/traces.jsonl", help="Path to save JSONL traces")
    args = parser.parse_args()

    seeds = [42 + i * 101 for i in range(args.seeds)]
    runner = TrackMRunner()

    print(f"🚀 Running Track M Benchmark on {args.cases} cases across {len(seeds)} seeds...")
    results = runner.run_benchmark(
        seeds=seeds,
        cases_limit=args.cases,
        trace_file=Path(args.traces),
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("📊 TRACK M BENCHMARK RESULTS SUMMARY")
    print("=" * 60)

    b_res = results["baseline_results"]
    k_res = results["kernel_results"]

    print(f"Contradiction Admission Rate:")
    print(f"  - Baseline: {b_res['contradiction_admission_rate']['mean'] * 100:.1f}% (±{b_res['contradiction_admission_rate']['ci_95'] * 100:.1f}%)")
    print(f"  - Kernel:   {k_res['contradiction_admission_rate']['mean'] * 100:.1f}% (±{k_res['contradiction_admission_rate']['ci_95'] * 100:.1f}%)")

    print(f"\nDownstream Retrieval Factuality:")
    print(f"  - Baseline: {b_res['downstream_factuality']['mean'] * 100:.1f}% (±{b_res['downstream_factuality']['ci_95'] * 100:.1f}%)")
    print(f"  - Kernel:   {k_res['downstream_factuality']['mean'] * 100:.1f}% (±{k_res['downstream_factuality']['ci_95'] * 100:.1f}%)")

    print(f"\nQuarantine F1 Score:")
    print(f"  - Kernel:   {k_res['quarantine_f1']['mean']:.3f} (±{k_res['quarantine_f1']['ci_95']:.3f})")

    print(f"\n✅ Results saved to {out_path}")
    print(f"✅ Telemetry traces saved to {args.traces}")


if __name__ == "__main__":
    main()
