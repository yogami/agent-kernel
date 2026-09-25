# 🔬 Reproducibility Guide (`agent_kernel-bench`)

This document provides exact, deterministic commands to reproduce all benchmark results reported for the Autonomous Agent Kernel (`agent_kernel`).

> **Evaluation Architecture Notice**
> The evaluation framework contains two distinct tiers: (1) authentic live frontier LLM evaluations run against real clinical records using `ops/run_abc_clinical_eval_v2.py` with results in `reports/ABC_CLINICAL_EVALUATION_V2.md`, and (2) deterministic algorithmic validation suites (Tracks M, T, and C) that test memory quarantine, sandbox containment, and causal DAG verification across multi-seed configurations.

---

## 1. System Requirements & Environment Setup

The evaluation suite runs on Python 3.10+ (tested on Python 3.12). It requires zero proprietary GPUs and runs standard CPU workloads in under 60 seconds.

```bash
# Clone the repository
git clone https://github.com/yogami/agent-kernel.git
cd agent-kernel

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Running Individual Evaluation Tracks

### Track M: Write-Path Memory Integrity & Admission Gate
Evaluates 200 longitudinal multi-turn dialogue streams with planted temporal contradictions across Clinical, Enterprise IT, Fintech, and Support domains.

```bash
# Run multi-seed evaluation (Seeds: 42, 143, 244)
python -m evals.track_m_memory.runner --cases 200 --seeds 3

# Expected Output Artifacts:
# - reports/track_m_results.json
# - evals/track_m_memory/traces.jsonl
```

Expected output:
- Baseline Contradiction Admission Rate: 100.0%
- Kernel Contradiction Admission Rate: 0.0%
- True Update Acceptance Rate: 100.0%
- Quarantine F1 Score: 0.940

---

### Track T: Tool Policy & Confinement Suite
Evaluates 50 carrier tasks (25 benign, 25 adversarial exploits covering 8 attack vectors including socket egress, credential theft, cycles, path traversal, and command injection).

```bash
# Run multi-seed evaluation (Seeds: 42, 143, 244)
python -m evals.track_t_confinement.runner --cases 50 --seeds 3

# Expected Output Artifacts:
# - reports/track_t_results.json
# - evals/track_t_confinement/traces.jsonl
```

Expected output:
- Baseline Exploit Block Rate: 0.0%
- Kernel Exploit Block Rate: 100.0%
- False Block Rate on Benign Tools: 0.0%
- Secrets Leaked: 0 in Kernel vs 3 in Baseline

---

### Track C: Causal Pre-Flight Verification & Observational Falsification
Evaluates 20 Structural Causal Models in GraphML format paired with observational tabular records ($N = 500$). Exactly 10 models contain planted hidden confounders breaking testable conditional independencies.

```bash
# Run multi-seed evaluation (Seeds: 42, 143, 244)
python -m evals.track_c_causal.runner --cases 20 --seeds 3

# Expected Output Artifacts:
# - reports/track_c_results.json
# - evals/track_c_causal/traces.jsonl
```

Expected output:
- Baseline Falsified Interception Rate: 0.0%
- Kernel Falsified Interception Rate: 100.0%
- False Alarm Rate on Valid Models: 0.0%
- Multiple Testing Correction: Benjamini-Hochberg FDR applied

---

## 3. Unified Factorial Matrix & McNemar Hypothesis Testing

To reproduce the complete factorial grid across model tiers and run the McNemar paired exact hypothesis test:

```bash
# Run all 270 cases across 3 seeds
python -m ops.run_council_bench --seeds 3 --track-m-cases 200 --track-t-cases 50 --track-c-cases 20

# Expected Output Artifacts:
# - reports/council_bench_results.json
# - reports/BENCHMARK_VERIFIED.md
# - evals/traces_unified.jsonl
```

### Fast Smoke Test Option
For rapid continuous integration or quick verification:

```bash
python -m ops.run_council_bench --seeds 1 --track-m-cases 20 --track-t-cases 10 --track-c-cases 6
```

---

## 4. Full Pytest Test Suite Verification

Run all unit, integration, and security tests across the test suite (165 passing tests):

```bash
pytest tests/ -v
```

All tests must pass with zero failures and zero skipped checks.
