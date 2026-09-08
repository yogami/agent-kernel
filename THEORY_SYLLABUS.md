# Autonomous Agent Kernel: Theoretical Mastery Syllabus

**Category:** Core Systems Theory & Architecture Deep Dive  
**Context:** Grounded in Thread `e5c68d37-bfd2-456e-8ad5-2db2c0046a8d`  
**Target Repository:** `yogami/agent-kernel` (`/Users/yamijala/gitprojects/agenticAI`)  
**Status:** Active Study Track  

---

## 🎯 Syllabus Scope & Objectives

This syllabus unpacks the architectural patterns, math, failure modes, and control systems designed and implemented in the Autonomous Agent Kernel. Each module covers the theory, structural mechanics, concrete implementation details, and trade-offs.

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        AUTONOMOUS AGENT KERNEL THEORY TRACK                            │
├───────────────────────────────┬───────────────────────────────┬────────────────────────┤
│ Module 1: Architecture Core   │ Module 2: Execution Runtimes  │ Module 3: Memory Gate  │
│ Zero-Framework & Hexagonal    │ FSMs, Budgets & Loop Breaks   │ Tri-State Isolation    │
├───────────────────────────────┼───────────────────────────────┼────────────────────────┤
│ Module 4: Context RAM         │ Module 5: Typed Tool Pipeline │ Module 6: Self-Healing │
│ Memory Budgets & Untrusted Delim │ Schemas, De-ID & Guardrails │ Monotonic Gates & Eval │
├───────────────────────────────┴───────────────────────────────┴────────────────────────┤
│ Module 7: Empirical Benchmarking & Offline Deterministic Replay                        │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 📚 Module Breakdown

### Module 1: Agent Systems Architecture & The Zero-Framework Paradigm
* The fragility of off-the-shelf frameworks (LangChain, AutoGen, LlamaIndex): hidden state, opaque recursion, and tight coupling.
* Hexagonal Architecture (Ports and Adapters) for LLM systems.
* Domain isolation: keeping business models and FSM transitions free of vendor SDKs and database drivers.
* Typed domain modeling using Pydantic v2 entities (`Turn`, `Episode`, `CandidateFact`, `SemanticFact`).

### Module 2: Finite State Machine (FSM) Execution Loops & Determinism
* Moving beyond unbounded while-loops to formal state transition graphs.
* The 8-state execution lifecycle: `COMPOSE_CONTEXT` -> `MODEL_CALL` -> `VALIDATE_TOOL_CALL` -> `EXECUTE_TOOL` -> `VALIDATE_OUTPUT` -> `PERSIST_EPISODE` -> `COMPLETED` / `FAILED`.
* Hard resource budgeting: maximum step caps (5 steps), token spending ceilings, and latency limits.
* Cycle-breaking algorithms: hashing tool signatures (`tool_name` + canonicalized arguments) to stop repetitive tool calls.

### Module 3: Multi-Tier Memory Hierarchy & The Tri-State Write Gate
* Cognitive memory taxonomy: Working Memory (Context RAM), Episodic Memory (append-only logs), Procedural Memory (prompts and tool schemas), and Semantic Memory (distilled knowledge).
* The memory poisoning problem: contradictory assertions and temporal drift across longitudinal sessions.
* The Tri-State Write Gate:
  1. Extraction lands strictly in `fact_quarantine` with `PENDING` status.
  2. Admission Control Gate: closed schema validation, confidence thresholding ($\ge 0.85$), and neighborhood contradiction queries.
  3. Verified promotion to `semantic_facts` with temporal validity intervals (`valid_from`, `valid_until`) and source lineage.
* SQLite WAL-mode append-only guarantees versus mutable table writes.

### Module 4: Context RAM Engineering & Untrusted Memory Injection
* Treating the model context window as CPU cache with explicit token budgets.
* Indirect prompt injection vectors emerging from long-term memory retrieval.
* Cognitive perimeter defense: wrapping retrieved semantic facts inside `<untrusted_retrieved_memory>` XML tags with explicit boundary instructions.

### Module 5: Typed Tool Contracts & Domain Pipelines
* Pydantic-driven input and output validation contracts.
* Fail-closed error feedback: converting tool schema rejections into self-correcting prompt turns.
* Clinical data pipeline case study: deterministic de-identification (PII scrubbing), ontology harmonization (SNOMED-CT / ICD-10), FHIR R4 schema enforcement, and biological guardrails.

### Module 6: LLM-Ops, Monotonic Non-Regression & Self-Healing
* Granular telemetry: tracking token yield, step latency, dollar costs, and write-gate admission yield.
* Immutable configuration packs (`config/packs/vN/`) with SHA-256 integrity validation.
* Three-tiered evaluation matrices: Golden (frozen non-regression), Dev (diagnostic), and Holdout (sealed generalization).
* Monotonic self-healing algorithms: root-cause diagnosis, candidate pack generation, and rollback rules.

### Module 7: Empirical Benchmarking & Deterministic Replayability
* Designing comparative agent benchmarks: Standard RAG vs Schema-Checked RAG vs Gated Agent Kernel.
* Core evaluation metrics: contradiction rates, schema breakage, PII leakage, runaway loops, and p95 latency.
* Record-and-replay engines: testing agent control logic offline with zero LLM API cost.

---

## 🔗 Cross-References
* Architecture Specification: [[01_Projects/Internal_Startups/Agent_Kernel/Agent_Kernel_Spec.md]]
* Microservices Catalog: [[03_Resources/Microservices_Catalog.md]]
* Master Career Syllabus: [[02_Areas/Professional_Development/Agentic_Systems_Engineering_Syllabus.md]]
