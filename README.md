# Autonomous Agent Kernel (`agent_kernel`)

A production-grade, framework-free AI agent execution kernel engineered with pure Python 3.12, typed Pydantic v2 domain contracts, a step-budgeted Finite State Machine, an append-only episodic log, a tri-state memory quarantine write gate, an Anthropic Model Context Protocol (MCP) server, Causal LLM Reasoning (Pearl's Ladder: SCM DAGs, do(X) interventional simulation, and counterfactual attribution), Multimodal Kinetics & Computer Vision tools, Hybrid RAG (BM25 + Dense + RRF + Reranking), Prometheus telemetry, and a monotonic non-regression evaluation self-healer.

```
                                  [ User Request / Clinical / Video Landmark Input ]
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. CONTEXT RAM, HYBRID RAG & CAUSAL MODELS (Working Memory, Literature & World Models)                  │
│    ├── System Prompt Pack (Versioned immutable config from `config/packs/vN/`)                          │
│    ├── Procedural Memory (Strict Pydantic tool schemas and operational contracts)                       │
│    ├── Structural Causal Models (`core/causal_graph.py`): Directed Acyclic Graphs with causal mechanisms │
│    ├── Hybrid RAG Engine (`infrastructure/hybrid_rag.py`): BM25 + Dense Cosine + Reciprocal Rank Fusion │
│    ├── Semantic Memory (Reads ONLY from promoted `semantic_facts`, filtered by TTL and tenant)          │
│    │   └── Untrusted Data Delimiter: Memory facts injected as reference data, never as system rules    │
│    ├── Episodic Context (Recent N turns retrieved from immutable `episodes` table)                      │
│    └── User Input & Runtime State                                                                       │
└────────────────────────────────────────────────┬────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. INNER EXECUTION ENGINE & INTERVENTIONAL GATE (Finite State Machine + Pearl's do(X))                  │
│    State Transitions:                                                                                   │
│    [compose_context] ──> [model_call] ──> [validate_tool_call] ──> [interventional_gate]                   │
│                                                     │                        │                          │
│                                                     ▼                        ▼                          │
│                                             [fail_closed]            [execute_tool] ──> [validate_output]│
│                                                                                              │          │
│                                                                                              ▼          │
│                                                                                      [persist_episode]  │
│                                                                                                         │
│    Hard Limits: max_steps (5), max_tokens, max_usd_cost, max_identical_tool_repeats (Loop Detection)   │
│    Causal Gate: Pearl's do-operator on mutilated graphs G_{/X} prevents adverse causal cascades         │
└────────────────────────────────────────────────┬────────────────────────────────────────────────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        ▼                                                 ▼
             [ WebSocket / SSE Streaming ]               [ Append-Only SQLite `episodes` Table ]
             [ Prometheus `/metrics` Stats ]                              │
                                                                          ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. MIDDLE LOOP: TRI-STATE MEMORY ADMISSION PIPELINE (Gated Promotion)                                   │
│    Step 1: Extractor generates Candidate Facts from episodic trace                                      │
│    Step 2: Commit candidate facts ONLY to `fact_quarantine` table (Retrieval NEVER reads quarantine)    │
│    Step 3: Gated Promotion Engine evaluates candidates against promotion criteria:                      │
│            ├── Criteria A: Closed Pydantic schema validation (subject, predicate, object, confidence)   │
│            ├── Criteria B: Semantic neighborhood query (Detects and rejects contradictory claims)       │
│            ├── Criteria C: High confidence threshold (Score >= 0.85)                                    │
│            ├── Criteria D: Causal Consistency Check (SCM validates causal plausibility of assertion)    │
│            └── Action: IF PASS -> Insert into `semantic_facts` with full provenance metadata            │
│                        IF FAIL -> Mark quarantine status as `REJECTED` with rejection reason            │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                                 │
                                                 ▼
┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. OUTER LOOP: LLM-OPS & COUNTERFACTUAL DIAGNOSTICS (Versioned Experimentation & Pearl's Level 3)       │
│    ├── Split Evaluation Matrices:                                                                       │
│    │    ├── `evals/golden/`  -> Frozen deterministic assertions (Zero regression allowed)               │
│    │    ├── `evals/dev/`     -> Visible to diagnostic healer for root-cause analysis                     │
│    │    └── `evals/holdout/` -> Sealed test set for generalization scoring                              │
│    ├── Counterfactual Root-Cause Attributor:                                                            │
│    │    3-step Pearlian algorithm (Abduction -> Action -> Prediction) isolates necessary causes        │
│    └── Self-Healing Healer Loop:                                                                        │
│         1. Evaluates failed dev traces and diagnoses root-cause error category                          │
│         2. Proposes a new versioned config pack (`config/packs/vN+1/`)                                  │
│         3. Runs full evaluation on candidate pack                                                       │
│         4. GATING RULE: if golden_passed and holdout_passed: commit else rollback                       │
└─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🏛️ Core Architecture Pillars

### 1. Causal LLM Reasoning (Pearl's Ladder of Causation)
Standard agents operate purely on Level 1 (Association / Correlation). `agent_kernel` implements Judea Pearl's full hierarchy:
- **Level 1 (Association):** Observational evidence across patient state and context.
- **Level 2 (Intervention - $do(X)$):** Before tool execution, simulates the action on a mutilated causal graph $G_{\bar{X}}$ where incoming arrows are severed. Catches severe drug contraindications or organ stress before physical execution.
- **Level 3 (Counterfactuals):** Computes *"What would have happened if action $A'$ had been taken instead of $A$?"* using 3-step Pearlian mechanics (Abduction &rarr; Action &rarr; Prediction) to attribute failure root causes.

### 2. Tri-State Memory Write Gate with Causal Consistency
Direct writes to vector databases cause memory poisoning. `agent_kernel` solves this with a three-table architecture:
1. **`episodes`**: Immutable, append-only SQLite log recording raw turns and tool executions.
2. **`fact_quarantine`**: New extractions are committed exclusively here with `PENDING` status. The active agent context never queries quarantine.
3. **`semantic_facts`**: Promotes candidate facts only after schema validation, confidence thresholding ($score \ge 0.85$), neighborhood contradiction checks, and **Causal Consistency Checks** against the SCM.

### 3. Step-Budgeted Finite State Machine
Open-ended `while True` loops lead to runaway costs and infinite tool recursion. `agent_kernel` enforces explicit state transitions with hard budget guards:
- **Max Step Limit**: Default 5 steps per turn before raising `BudgetExceededError`.
- **Loop Breaker**: Tracks tool call signatures and raises `LoopDetectedError` if an identical tool invocation repeats.
- **Dollar & Token Caps**: Terminates early if cost exceeds safety thresholds.

### 4. Multimodal Kinetics & Biomechanical Safety Circuit Breakers
- **3D Pose Landmark Parser:** Validates 33 body joint coordinates (MediaPipe/OpenCV) and calculates temporal occlusion rates.
- **Biomechanical Angle Calculator:** Computes 3D joint flexion/extension angles, angular velocities ($\Delta \theta / \Delta t$), and medial valgus displacement.
- **Movement Safety Circuit Breaker:** Deterministic safety rules intercepting knee valgus collapse ($> 15^\circ$), lumbar flexion rounding under load ($> 30^\circ$), and ballistic velocity spikes ($> 450^\circ/s$).
- **VLM Exercise Evaluator:** Produces structured physical therapy evaluation report cards from movement kinematics.

### 5. Hybrid RAG with Reciprocal Rank Fusion & Citations
Standard vector search misses exact alphanumeric codes (ICD-10 `I10` or drug dosages). `agent_kernel` combines:
- **BM25 Sparse Lexical Search:** Exact token matching across guideline literature.
- **Dense Cosine Vector Index:** Semantic similarity retrieval.
- **Reciprocal Rank Fusion (RRF):** Merges dense and sparse ranks with constant $k=60$.
- **Strict Citation Provenance:** Formats every retrieved chunk with document, section, paragraph, and table references.

### 6. Anthropic Model Context Protocol (MCP) Server
Implements the open JSON-RPC 2.0 standard over stdio and HTTP, allowing external clients (Cursor, Claude Desktop, external agent swarms) to execute kernel tools seamlessly.

### 7. Prometheus Telemetry & Low-Latency WebSocket Streaming
- **WebSocket Streaming (`/ws/chat`):** Streams tokens and broadcasts live FSM transitions (`COMPOSE_CONTEXT`, `MODEL_CALL`, `EXECUTE_TOOL`).
- **Prometheus Scrape Endpoint (`/metrics`):** Exposes request counts, P50/P90/P99 latency histograms, and error distributions.

---

## 📊 Evaluation & Verification

We evaluate `agent_kernel` through two complementary frameworks: live frontier model testing and deterministic high-volume stress simulation.

### 1. Empirical Live Frontier Evaluation (Head-to-Head A/B/C v2)

Benchmarked against raw frontier models and generic agent harnesses using authentic clinical records (`evals/mtsamples/mtsamples_v2.jsonl`) with live API execution via `AsyncGeminiAdapter` (`gemini-3.5-flash-lite`):

| Metric | Track A: Raw Frontier LLM | Track B: Generic Agent Harness | Track C: Agent Kernel | Kernel Architectural Advantage |
| :--- | :---: | :---: | :---: | :---: |
| **Memory Contradiction Writes** | 33.3% | 33.3% | **0.0%** | Tri-State Memory Quarantine blocks stale/conflicting writes |
| **Drug Contraindication Escapes** | 0.0% | 8.3% | **0.0%** | Deterministic Knowledge Graph Assertion Gate |
| **PHI / PII Data Exposure Rate** | 8.3% | 8.3% | **0.0%** | Output Firewall Redaction |
| **Median Latency (P50)** | 7.3s | 18.7s | **5.4s** | Bounded Finite State Machine overhead |

Complete report: [reports/ABC_CLINICAL_EVALUATION_V2.md](reports/ABC_CLINICAL_EVALUATION_V2.md).  

> [!NOTE]
> **PHI / PII Gate Enforcement & Metric Reconciliation:**
> - In Live Frontier Evaluation (Section 1), Track C achieves a 0.0% PHI exposure rate via the `OutputGuardrails` deterministic privacy gate (`PAT-*`, `MRN:*`), preventing unscrubbed model emissions. Raw models (Track A) and generic harnesses (Track B) exhibited an 8.3% exposure rate.
> - In Pipeline Simulation (Section 2), the 0.0% leakage rate reflects automated de-identification across 100 multi-encounter Synthea records.
> - Pipeline simulation metrics represent algorithmic state machine latency (p95 overhead 7.7 ms, $0.0014 per record), whereas Section 1 measures end-to-end network API latency against live frontier models.

Reproduce with live models:
```bash
python ops/run_abc_clinical_eval_v2.py --cases 12 --adapter gemini
```


### 2. High-Volume Pipeline Simulation (100 Longitudinal Records)

*(Deterministic simulation: not measured on live patient data).*

To profile memory growth, loop breaking, and admission gates at scale without incurring API spend, `ops/benchmark.py` provides deterministic algorithmic stress testing across 100 multi-encounter Synthea records (200 encounters):

| Metric | System A (Flat RAG) | System A+ (Schema-Checked) | System B (Agent Kernel) | Impact |
| :--- | :---: | :---: | :---: | :---: |
| **Fact Contradiction Rate** (Simulation) | 12.5% | 12.5% | **0.0%** | Contradictions isolated to quarantine backlog |
| **Schema Breakage Rate** (Simulation) | 12.5% | 0.0% | **0.0%** | Strict Pydantic v2 contract enforcement |
| **PII / PHI Leakage Rate** (Simulation) | 100.0% | 87.5% | **0.0%** | Automated de-identification pipeline |
| **Runaway Loop Incidents** (Simulation) | 12.5% | 0.0% | **0.0%** | Hard step budgets and signature cycle breakers |
| **p95 Turn Overhead (ms)** (Simulation) | 15.0 ms | 10.0 ms | **7.7 ms** | Local algorithmic execution latency |

*(Deterministic simulation: not measured on live patient data. Reconciled against ops/benchmark.py timing: p95 turn overhead 7.7 ms, $0.0014 cost per record).*



Complete report: [BENCHMARK_REPORT.md](BENCHMARK_REPORT.md).  
Run simulation:
```bash
python ops/benchmark.py
```

## 🛠️ Project Structure (Hexagonal Architecture)

```
agent_kernel/
├── domain/                      # Pure business logic and typed interfaces
│   ├── models.py                # Pydantic v2 entities (Turn, Episode, CandidateFact, SemanticFact)
│   ├── ports.py                 # Protocol interfaces (LLMProvider, Stores, Tools)
│   └── state_machine.py         # FSM states and typed runtime exceptions
├── infrastructure/              # Adapters implementing domain ports
│   ├── sqlite_episode_store.py  # Immutable append-only episode store
│   ├── sqlite_fact_store.py     # Tri-state fact tables (quarantine, semantic facts)
│   ├── hybrid_rag.py            # BM25 + Dense vector search + RRF + Citation formatting
│   ├── mcp_server.py            # Anthropic Model Context Protocol (JSON-RPC 2.0) server
│   ├── vector_index.py          # Vector accelerator with metadata filtering
│   └── llm_adapter.py           # Multi-provider client with local deterministic mock
├── core/                        # Application orchestration services
│   ├── causal_graph.py          # Structural Causal Models (SCMs), DAGs, do(X), counterfactuals
│   ├── causal_reasoner.py       # Causal Reasoner, interventional gates, and root-cause attribution
│   ├── context_ram.py           # Context RAM assembler with untrusted memory delimiters
│   ├── execution_loop.py        # Step-budgeted FSM execution engine
│   ├── memory_promoter.py       # Admission control gate (Quarantine -> Validate -> Promote)
│   └── guardrails.py            # Output firewall and safety filters
├── tools/                       # Standard typed tool registry
│   ├── registry.py              # Registry mapping schemas to execution handlers
│   ├── causal_tools.py          # do(X) simulation, counterfactual attribution, graph query
│   ├── clinical_tools.py        # De-ID scrubber, SNOMED/ICD-10 mapper, FHIR validator
│   ├── multimodal_tools.py      # Pose parser, 3D joint angles, safety circuit breaker, VLM eval
│   └── system_tools.py          # Math calculator and date validator
├── ops/                         # LLM-Ops, Evaluation, and Self-Healing
│   ├── prometheus_exporter.py   # Prometheus metrics collector (/metrics endpoint)
│   ├── telemetry.py             # Span collection (tokens, cost, latency, yield)
│   ├── config_manager.py        # Versioned config packs (config/packs/vN/)
│   ├── eval_runner.py           # Golden, dev, and holdout suite execution
│   ├── self_healer.py           # Root-cause diagnosis and monotonic gating
│   └── benchmark.py             # 3-way longitudinal benchmark runner
├── evals/                       # Frozen test datasets and synthetic generators
│   ├── datasets/                # Synthea patient generator and ground truth
│   ├── golden/                  # Frozen non-regression test cases
│   ├── dev/                     # Diagnostic test cases
│   └── holdout/                 # Sealed generalization test cases
├── api/                         # FastAPI application and UI cockpit
│   ├── app.py                   # REST endpoints and Prometheus scrape router
│   ├── websocket_stream.py      # Real-time WebSocket token and state event stream
│   └── static/index.html        # Real-time FSM, Memory, and Benchmark cockpit
├── tests/                       # Comprehensive pytest suite (165 tests passing in ~5s)
│   ├── test_causal_reasoning.py # SCM DAGs, do(X) intervention, counterfactuals, memory gate
│   ├── test_state_machine.py    # FSM transitions, budget caps, loop break triggers
│   ├── test_memory_gate.py      # Quarantine isolation, contradiction rejection, promotion
│   ├── test_tool_sandbox.py     # Subprocess containment, path traversal and network guards
│   ├── test_policy_engine.py    # Declarative policy gates and monotonic checks
│   ├── test_llm_adapters_multi_provider.py # Multi-provider client cascade and fallback
│   ├── test_multimodal.py      # Pose landmarks, 3D angle math, safety breakers
│   ├── test_hybrid_rag.py       # BM25, dense vectors, RRF score bounds, citations
│   ├── test_mcp.py              # JSON-RPC 2.0 tool discovery, execution, error handling
│   ├── test_streaming.py        # WebSocket streaming and Prometheus metrics format
│   ├── test_eval_gating.py      # Monotonic non-regression and automatic rollback
│   ├── test_replay.py           # Deterministic episode replay from recorded traces
│   └── test_clinical_pipeline.py# De-ID, ontology extraction, FHIR validation
├── Makefile                     # make test, make eval, make benchmark, make run
├── requirements.txt             # Minimal dependencies (FastAPI, Pydantic, pytest, numpy)
└── Dockerfile                   # Multi-stage production container
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Test Suite
```bash
make test
# or pytest tests/ -v (165 passing tests in ~5s)
```

### 3. Run MCP Server (Stdio Mode)
```bash
python3 infrastructure/mcp_server.py
```

### 4. Launch Cockpit & Scrape Prometheus Metrics
```bash
make run
# App UI: http://localhost:8000
# Prometheus Scrape: http://localhost:8000/metrics
# WebSocket Stream: ws://localhost:8000/ws/chat
# Causal Simulation API: POST http://localhost:8000/api/causal/simulate
```
