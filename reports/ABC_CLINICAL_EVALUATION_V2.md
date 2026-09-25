# Head-to-Head A/B/C Evaluation v2: Raw Frontier vs Generic Agent vs Agent Kernel

**Date:** 2026-09-17
**Dataset:** Authentic Multi-Hop Clinical Corpus (`evals/mtsamples/mtsamples_v2.jsonl`)
**Cohort:** 12 Patient Encounters
**Underlying Model Adapter:** `AsyncGeminiAdapter` (`gemini-3.5-flash-lite`)

---

## Empirical A/B/C Metric Comparison

| Metric | Track A: Raw Frontier LLM | Track B: Generic Agent Harness | Track C: Agent Kernel | Kernel Architectural Advantage |
| :--- | :---: | :---: | :---: | :---: |
| **PHI / PII Data Exposure Rate** | **8.3%** | **8.3%** | **0.0%** | Output Firewall Redaction |
| **Drug Contraindication Escapes** | **0.0%** | **8.3%** | **0.0%** | Knowledge Graph Assertion Gate |
| **Memory Contradiction Writes** | **33.3%** | **33.3%** | **0.0%** | Tri-State Memory Quarantine |
| **Median Latency (P50)** | **7.3s** | **18.7s** | **5.4s** | Bounded State Overhead |


---

## Key Architectural Findings

1. **Pharmacokinetic and Dosing Blind Spots:** Raw frontier models miss complex drug interactions (like Clarithromycin + Colchicine) and renal dosing adjustments without deterministic calculation gates.
2. **The ReAct Tooling Paradox:** Generic agent harnesses provide tools, but tool execution is probabilistic. When the LLM decides an order is safe, it skips tool invocation entirely.
3. **Memory Integrity:** Stateless raw models and generic harnesses accept verbal denials and noisy measurements, overwriting verified clinical facts. The Agent Kernel's Tri-State gate preserves ground truth via deterministic quarantine.

Reproduce via:
```bash
python ops/run_abc_clinical_eval_v2.py --cases 12 --adapter gemini
```