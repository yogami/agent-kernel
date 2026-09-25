# Architectural Simulation Report: Longitudinal Multi-Encounter Stress Test

**Evaluation Type:** Deterministic Pipeline Simulation (Architectural Stress Test)  
**Dataset:** 100 Multi-Encounter Longitudinal Patient Records (200 Encounters)  
**Corpus Source:** Synthetic Synthea Clinical Cohort with Planted Temporal Contradictions  
**Measurement Method:** Deterministic pipeline modeling (state transition overhead, rule evaluation, token accounting)  
**Live LLM Benchmark Reference:** For live frontier model evaluations with measured API latencies, see [reports/ABC_CLINICAL_EVALUATION_V2.md](reports/ABC_CLINICAL_EVALUATION_V2.md) and [ops/run_abc_clinical_eval_v2.py](ops/run_abc_clinical_eval_v2.py).  

---

## Summary Comparison Matrix

| Evaluation Metric | System A (Standard RAG) | System A+ (Schema-Checked) | System B (Gated Agent Kernel) | Metric Delta (A -> B) |
| :--- | :---: | :---: | :---: | :---: |
| **Fact Contradiction Rate** | 12.5% | 12.5% | **0.0%** | **-12.5%** |
| **Schema Breakage Rate** | 12.5% | 0.0% | **0.0%** | **-12.5%** |
| **PII / PHI Leakage Rate** | 100.0% | 87.5% | **0.0%** | **-100.0%** |
| **Runaway Loop Incidents** | 12.5% | 0.0% | **0.0%** | **-12.5%** |
| **p95 Turn Latency (ms)** | 15.0 ms | 10.0 ms | **7.7 ms** | **7.3 ms faster** |
| **Cost per Patient Record** | $0.0032 | $0.0025 | **$0.0014** | **-56.2% cost** |

---

## Key Architectural Insights

1. **Tri-State Admission Gate Prevents Memory Contradictions:**
   In System A and System A+, un-gated flat memory blindly accepts conflicting assertions (12.5% contradiction rate). System B traps conflicting assertions in quarantine (25 items quarantined), preserving 0.0% contradiction in active semantic memory.

2. **Automated PII Scrubbing:**
   Passing note text through DeidentifyTextTool and OutputGuardrails eliminates patient names and clinic identifiers, dropping PII leakage from 100.0% to 0.0%.

3. **Step Budget and FSM Termination:**
   Hard step limits prevent runaway execution loops on conflicting inputs, reducing loop incidents from 12.5% to 0.0%.
