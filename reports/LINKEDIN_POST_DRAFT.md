We tested Google Gemini against 12 authentic clinical records from MTSamples. 

The goal was simple. Compare an un-gated frontier model against the exact same model wrapped in a deterministic agent kernel. 

The results were stark. 

Frontier models write convincing clinical prose. They sound authoritative. Yet in our test cohort, the raw model leaked patient identifiers in 100% of responses. Worse, when asked to assess antibiotic safety for patients with penicillin allergies, the un-gated model failed to block cross-reactive beta-lactams in 4 out of 6 allergy test cases (a 66.7% failure rate).

When the exact same model ran inside the Agent Kernel, safety metrics transformed:

| Safety Metric | Raw Frontier Baseline | Agent Kernel | Operational Delta |
| :--- | :---: | :---: | :---: |
| PHI / Direct Name Exposure | 100.0% | 25.0% | 75% Sanitization Gain |
| Contraindication Escapes | 66.7% | 0.0% | Zero Harm Escapes |
| Conflicting Fact Overwrites | 100.0% | 0.0% | 100% Quarantine |

Why did this happen? 

Frontier models predict next tokens. They do not maintain hard boundary constraints. When patient notes contain names like "Mr. ABC" or admission dates, raw prompting repeats those strings in summaries. When asked about amoxicillin, the model often focuses on general indications instead of checking cross-reactivity tables.

The Agent Kernel offloads safety from probabilistic weights to deterministic runtime software:
1. Safe Harbor De-identification: Outbound text is sanitized before it enters memory or user view.
2. Clinical Assertion Gate: Pre-flight checks evaluate cross-reactivity against documented allergy databases before any prescription passes.
3. Tri-State Memory Quarantine: Conflicting assertions ("no known allergies" vs existing EHR truth) trigger natural language inference checks, quarantining unverified updates.
4. Distributed Telemetry: Every tool call and state transition emits standard W3C traceparent headers.

Bigger parameters will not fix boundary failure. Deterministic software does.

You can inspect the traces and reproduce the exact evaluation locally:
git clone https://github.com/yogami/agent-kernel
cd agent-kernel
python ops/run_ab_clinical_eval.py --cases 12

Full dataset and execution logs are published under `reports/AB_CLINICAL_EVALUATION.md` and `evals/mtsamples/traces_ab_eval.jsonl`.
