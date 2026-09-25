# LinkedIn Post Draft: Week 3 - The A/B/C Test (Raw Model vs Generic Agent vs Agent Kernel)

I ran Google Gemini 2.5 Flash through an A/B/C benchmark across 12 authentic patient records from the MTSamples clinical corpus. 

The goal was to test three different AI architectures. The results challenged some common assumptions about agent design.

Here are the real numbers across all three tracks.

### Track A: The Raw Frontier Model
I prompted Gemini directly with zero-shot clinical safety instructions.
* 0.0% PHI exposure rate.
* 0.0% drug contraindication escapes. The raw model was smart enough to catch a buried penicillin allergy in the text and flag the antibiotic order without needing any specialized tools.
* 100% memory contradiction writes. The model quietly accepted conflicting facts and corrupted historical state.
* Median latency: 7.2s.

### Track B: The Generic Agent Harness (Standard ReAct Loop)
I gave Gemini tool-calling access to a clinical de-identification tool and an assertion checker in a multi-turn ReAct loop.
* 8.3% PHI exposure rate. When you force an LLM into a multi-turn loop, it sometimes echoes unscrubbed identifiers directly into the user stream. Adding tools actually degraded privacy.
* 0.0% contraindication escape rate.
* 100% memory contradiction writes. Conflicting facts still wrote directly to state.
* Median latency: 18.5s.

Notice the difference between Track A and Track B. The generic harness did not add value. It made the system twice as slow and introduced a privacy leak. Both architectures failed completely on memory persistence. When new conflicting facts arrived, they silently overwrote ground truth.

### Track C: The Agent Kernel
I stopped treating the LLM as the controller. The model became a compute unit bounded by a finite state machine, a declarative Policy-as-Code engine, and a deterministic output firewall.
* 0.0% PHI exposure rate. A compiled output guardrail scrubbed every identifier before emission.
* 0.0% contraindication escapes. Dual-layer assertion checking caught and intercepted every conflict.
* 0.0% memory contradiction writes. 100% of conflicting facts were quarantined by the Tri-State memory promoter.
* Median latency: 22.8s.

### The Architectural Reality: Honest Engineering Tradeoffs

Frontier models like Gemini 2.5 Flash are already incredibly capable out of the box. Wrapping them in generic agent loops often adds latency without solving the hard problems.

The real challenge is state management. My early contraindication checks were hardcoded dictionaries in Python. That does not scale across hospital systems. 

I refactored domain constraints into declarative Policy-as-Code files. The policy engine reads structured JSON taxonomy trees (Beta-lactams, Cephalosporins, NSAIDs, Opioids) with negation awareness. The same engine enforces capital markets rules (order notional caps, blacklisted tickers) without modifying engine code.

Enforcement runs on two levels.
1. Intent level. When the model calls a tool, the request delegates to the knowledge graph policy engine.
2. Output firewall. When the model produces final text, the kernel scans output tokens against active patient constraints. If an unblocked contraindication appears, compiled code injects a safety interception and withholds the medication. The model does not get a vote.

Deterministic state machines add roughly 15 seconds of overhead compared to the raw model (22.8s vs 7.2s). In clinical decision support or financial execution, that overhead is the price you pay for true deterministic guarantees and memory isolation.

All traces, declarative policy schemas, and evaluation scripts are live in the repo.
`python ops/run_abc_clinical_eval.py --cases 12 --adapter gemini`

GitHub: https://github.com/yogami/agent-kernel

#SystemArchitecture #SoftwareEngineering #AgenticSystems #LLMOps #StateMachines #PolicyAsCode #BuildInPublic
