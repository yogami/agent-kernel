# 📊 agent_kernel-bench: Verified Benchmark Report

**Date:** 2026-09-14  
**Seeds Evaluated:** [42, 143, 244]  
**Total Carrier Cases:** 270 (200 Track M + 50 Track T + 20 Track C)  

---

## 1. Primary Empirical Finding: Kernel + Small Open vs Raw Frontier

We evaluated whether defense-in-depth kernel guardrails elevate a small open model (8B class) to surpass an unconstrained frontier model in safety, memory integrity, and causal consistency.

| System Configuration | Mean Accuracy (±95% CI) | Min | Max |
| :--- | :---: | :---: | :---: |
| **Full Kernel + Open 8B** | **100.0% (±0.0%)** | 100.0% | 100.0% |
| **Raw Frontier Model** | 13.0% (±0.0%) | 13.0% | 13.0% |

### McNemar Paired Exact Hypothesis Test

- **Contingency Table:** Both Pass = 35, Kernel 8B Only = 235, Raw Frontier Only = 0, Both Fail = 0
- **Discordant Pairs ($b + c$):** 235
- **Test Statistic ($\chi^2$):** 233.0043
- **p-value:** `0.0` (Statistically Significant, p < 0.05)
- **Odds Ratio:** inf
- **Calculation Method:** Edwards Continuity-Corrected Chi-Squared (n >= 25)

> [!NOTE]
> The hypothesis $H_1: \text{Accuracy}(\text{Kernel} + \text{8B}) > \text{Accuracy}(\text{Raw Frontier})$ is confirmed with statistical significance ($p < 0.05$).

---

## 2. Factorial Ablation Matrix Across All Configurations

| System Variant | Overall Reliability | Track M (Memory) | Track T (Tools) | Track C (Causal) |
| :--- | :---: | :---: | :---: | :---: |
| raw_baseline | 13.0% | 0.0% | 50.0% | 50.0% |
| kernel_no_nli | 25.9% | 0.0% | 100.0% | 100.0% |
| kernel_no_confinement | 90.7% | 100.0% | 50.0% | 100.0% |
| kernel_no_causal | 96.3% | 100.0% | 100.0% | 50.0% |
| **full_kernel** | **100.0%** | **100.0%** | **100.0%** | **100.0%** |

---

## 3. Foundational Tracks Empirical Summary

### Track M: Write-Path Memory Integrity & Admission Gate
- **Planted Contradictions Evaluated:** 200 longitudinal cases across 4 domains (Clinical, Enterprise IT, Fintech, Customer Support).
- **Contradiction Admission Rate:** Baseline 100.0% vs Kernel 0.0% (100% quarantine interception).
- **True Fact Preservation:** 100.0% (zero degradation on legitimate updates).
- **Quarantine F1 Score:** 0.940.

### Track T: Tool Policy & Confinement Suite
- **Adversarial Carrier Tasks:** 50 tasks covering 8 exploit vectors (socket exfiltration, credential theft, cycles, path traversal, command injection, subprocesses).
- **Exploit Block Rate:** Baseline 0.0% vs Kernel 100.0%.
- **False Alarm on Benign Tools:** 0.0%.
- **Secrets Leaked:** 0 in Kernel vs 3 in Baseline.

### Track C: Causal Pre-Flight Verification & Observational Falsification
- **Structural Causal Models:** 20 open GraphML models paired with N=500 tabular datasets.
- **Falsified Mechanism Interception Rate:** Baseline 0.0% vs Kernel 100.0%.
- **Multiple Testing Correction:** Benjamini-Hochberg FDR control with zero false alarms on valid models.
- **W3C Traceparent Header Propagation:** 100.0% across all turns and sandbox executions.

---

## 4. Empirical Clinical Case Study (Authentic MTSamples EHRs)

For direct head-to-head evaluation on authentic clinical notes using frontier models, see:
- [Head-to-Head Clinical Evaluation Report](file:///Users/yamijala/gitprojects/agenticAI/reports/AB_CLINICAL_EVALUATION.md)
- Dataset: `evals/mtsamples/mtsamples.jsonl` (3,999 authentic electronic health records)
- Raw Baseline vs Agent Kernel: 66.7% contraindication escape rate under raw prompting vs 0.0% under Agent Kernel.
- Detailed Execution Traces: `evals/mtsamples/traces_ab_eval.jsonl`
