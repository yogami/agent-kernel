# Extantia Platform Architecture & Technical Walkthrough Guide

**Author:** Yami Gopal  
**Repository:** [yogami/agent-kernel](https://github.com/yogami/agent-kernel) (`/Users/yamijala/gitprojects/agenticAI`)  
**Audience:** Carlota (Partner at Extantia Capital) & Technical Review Team  
**Date:** September 2026  

---

## 1. Executive Summary

Extantia Capital requires a dedicated software platform to transform fund operations across four core areas:
1. Institutional Memory & Versioning
2. Agentic Deal Triage
3. Agentic Due Diligence
4. Portfolio Tools & Multi-Tenant Infrastructure

Most venture funds stitch together brittle scripts wrapped around commercial LLM frameworks. Those setups fail when models hallucinate, drop context, or confuse older pitch decks with revised side letters. 

The [`agent-kernel`](file:///Users/yamijala/gitprojects/agenticAI) codebase provides a framework-free execution kernel written in pure Python. It uses hexagonal architecture (ports and adapters), strict Pydantic v2 schemas, a deterministic finite state machine, a tri-state memory quarantine, and Pearl's causality ladder. 

This guide maps each component in the repository directly to Extantia's four platform pillars. It also provides a 10-minute code walkthrough script for technical discussions with investment partners and engineering evaluators.

---

## 2. The High-Stakes Proving Ground: Why Healthcare Informs Venture Diligence

Reviewers often ask why this codebase contains clinical records, Synthea cohorts, and pharmacotherapy schemas.

The answer is structural equivalence.

Longitudinal healthcare and venture capital diligence share identical failure modes. Both operate in high-consequence environments where chronological updates constantly supersede prior baselines. 

Consider what happens when retrieval fails in each domain:
* In healthcare: an agent confuses an amended drug dosage with a superseded baseline prescription. The patient suffers toxic drug interaction.
* In venture capital: an agent confuses a 2023 Series Seed term sheet with a 2024 Series A side letter. The investment committee miscalculates liquidation preferences, anti-dilution rights, or board composition.

In both fields, standard vector retrieval using cosine similarity fails. Cosine similarity only measures topical overlap. It has no intrinsic awareness of time or authority. When an updated contract uses 95% of the same words as the original draft, naive search regularly serves obsolete clauses to the language model.

We solved this in [`core/memory_promoter.py`](file:///Users/yamijala/gitprojects/agenticAI/core/memory_promoter.py) and [`infrastructure/sqlite_fact_store.py`](file:///Users/yamijala/gitprojects/agenticAI/infrastructure/sqlite_fact_store.py) through a deterministic admission controller:
* Every candidate assertion enters a quarantine table first.
* It must pass schema checks, confidence thresholds, and Natural Language Inference contradiction screening.
* When a newer assertion supersedes an older fact, the engine marks the previous fact retired (`valid_until = now()`, `is_active = FALSE`).
* Active context generation only pulls verified, active records.

By proving this architecture on strict longitudinal clinical datasets, we established that the kernel prevents document supersession bugs before touching financial data rooms.

---

## 3. Direct Mapping to Extantia's Platform Pillars

```
+---------------------------------------------------------------------------------------+
|                               EXTANTIA PLATFORM PILLARS                               |
+---------------------------+---------------------------+-------------------------------+
|  1. Institutional Memory  |  2. Agentic Deal Triage   |  3. Agentic Due Diligence     |
|     & Versioning          |                           |     & Counter-Case            |
+---------------------------+---------------------------+-------------------------------+
|  - SQLite / Postgres      |  - Deterministic FSM      |  - Judea Pearl Causal Ladder  |
|    Fact Stores            |  - Policy Engine (JSON)   |  - Observational Verifier     |
|  - Memory Promoter Gate   |  - Step & Cost Budgets    |  - Counterfactual Analysis    |
|  - Temporal Supersession  |  - Ambiguity & Conviction |  - Safe AST Mathematical      |
|  - Untrusted Context RAM  |    Queues                 |    Formula Evaluation         |
+---------------------------+---------------------------+-------------------------------+
|                               4. Portfolio Tools & Multi-Tenancy                     |
|  - Tenant-ID Database Isolation    - In-Process Sandbox Network Firewall              |
|  - Anthropic Model Context Protocol (MCP) Server for Partner Desktop Tools             |
+---------------------------------------------------------------------------------------+
```

### Pillar 1: Institutional Memory & Document Versioning

Extantia needs a shared memory store that captures firm knowledge across hundreds of meetings, decks, and technical memos without returning stale data.

#### Implemented Modules
* [`infrastructure/sqlite_fact_store.py`](file:///Users/yamijala/gitprojects/agenticAI/infrastructure/sqlite_fact_store.py) and [`infrastructure/postgres_store.py`](file:///Users/yamijala/gitprojects/agenticAI/infrastructure/postgres_store.py): Implements dual tables (`fact_quarantine` and `semantic_facts`) equipped with temporal boundaries (`valid_from`, `valid_until`, `is_active`, `retired_at`, `provenance_json`).
* [`core/memory_promoter.py`](file:///Users/yamijala/gitprojects/agenticAI/core/memory_promoter.py): The gatekeeper. Candidate facts are evaluated against existing knowledge using Cross-Encoder Natural Language Inference (NLI) to flag contradictions.
* [`core/context_ram.py`](file:///Users/yamijala/gitprojects/agenticAI/core/context_ram.py): Assembles working memory. Retrieved facts are isolated within `<untrusted_retrieved_memory>` XML blocks. This boundary prevents prompt injection attacks embedded inside founder pitch decks from rewriting core instructions.
* [`infrastructure/hybrid_rag.py`](file:///Users/yamijala/gitprojects/agenticAI/infrastructure/hybrid_rag.py): Hybrid search engine combining BM25 keyword matching with dense cosine embeddings and lexical score boosting.

#### How It Solves Extantia's Problems
When a founder submits an updated cap table or financial model, the engine does not overwrite historical records. It evaluates the delta. If the update contradicts prior verified metrics, it enters quarantine for partner review. If it represents an authorized amendment, the system sets `retired_at` on the old data and activates the new record. Analysts always see the current governing state alongside complete revision history.

---

### Pillar 2: Agentic Deal Triage

Deal flow in deep decarbonization is broad and messy. Inbound pitch decks, conference leads, and scout submissions flood partner inboxes daily.

#### Implemented Modules
* [`core/state_machine.py`](file:///Users/yamijala/gitprojects/agenticAI/core/state_machine.py): Finite State Machine enforcing state transitions (`BOOTSTRAP -> INGESTION -> REASONING -> SYNTHESIS -> TERMINATED`). Prevents agents from entering unbounded execution loops.
* [`core/policy_engine.py`](file:///Users/yamijala/gitprojects/agenticAI/core/policy_engine.py): Evaluates candidate tool executions against declarative JSON security policies before calling any external APIs.
* [`core/guardrails.py`](file:///Users/yamijala/gitprojects/agenticAI/core/guardrails.py): Enforces input sanitization, token ceilings, and PII masking.
* [`core/sandbox/runner.py`](file:///Users/yamijala/gitprojects/agenticAI/core/sandbox/runner.py): Confinements parser scripts inside temporary sandboxes, blocking arbitrary file system traversal and unauthorized socket connections.

#### How It Solves Extantia's Problems
Instead of letting an LLM run freely, deal triage runs within strict execution budgets. The kernel parses pitch decks, verifies patent claims, and scores founder backgrounds using deterministic rules. 

Outputs split into two distinct queues:
1. **The High-Conviction Queue:** Deals that strictly meet Extantia's stage, geography, and carbon threshold.
2. **The Ambiguity Queue:** Deals with missing technical data or contradictory unit economics, flagged specifically for human partner review.

This structure eliminates manual triage work without introducing false-negative filtering.

---

### Pillar 3: Agentic Due Diligence & Counter-Thesis Red-Teaming

Extantia avoids feel-good ESG narratives. The fund targets hardware and software solutions that stand on solid physics and unit economics. Standard language models exhibit sycophancy: when handed an optimistic pitch deck, they echo the founder's claims.

#### Implemented Modules
* [`core/causal_reasoner.py`](file:///Users/yamijala/gitprojects/agenticAI/core/causal_reasoner.py): Implements Judea Pearl's Causality Ladder:
  * Level 1 (Association): Observational correlations.
  * Level 2 (Intervention): Simulating policy actions via Pearl's `do`-calculus.
  * Level 3 (Counterfactuals): Executing the three-step counterfactual cycle (Abduction, Action, Prediction).
* [`core/causal_verifier.py`](file:///Users/yamijala/gitprojects/agenticAI/core/causal_verifier.py): Tests structural causal graphs against empirical tabular data using partial correlations and d-separation tests.
* [`core/causal_graph.py`](file:///Users/yamijala/gitprojects/agenticAI/core/causal_graph.py): Structural Causal Model engine using safe Abstract Syntax Tree (AST) parsing to evaluate mathematical formulas without running unsafe `eval()` calls.
* [`evals/track_c_causal/`](file:///Users/yamijala/gitprojects/agenticAI/evals/track_c_causal/): Test suite demonstrating causal graph verification, including financial credit default graphs.

#### How It Solves Extantia's Problems
The diligence agent acts as an adversarial prosecutor. 

Instead of asking the model: *"Is this startup a good investment?"*, the system builds a Structural Causal Model of the startup's revenue and cost drivers. It runs counterfactual interventions:
* *"What happens to gross margin if grid electricity costs rise 30%?"*
* *"What is the breakeven factory capacity if capital expenditure increases by 15 million euros?"*

The agent cross-examines pitch deck assertions against physical laws and supplier price sheets. The final delivery to the Investment Committee is not a summary. It is an adversarial punch-list of the five hardest questions partners must ask the founder.

---

### Pillar 4: Portfolio Tools & Multi-Tenant Infrastructure

Extantia wants to build software tools that serve both the investment team and portfolio founders. Providing services across multiple independent startups requires hard boundaries.

#### Implemented Modules
* [`infrastructure/postgres_store.py`](file:///Users/yamijala/gitprojects/agenticAI/infrastructure/postgres_store.py): Every database query mandates a `tenant_id` constraint. Data from Portfolio Company A can never leak into queries executed for Portfolio Company B.
* [`infrastructure/mcp_server.py`](file:///Users/yamijala/gitprojects/agenticAI/infrastructure/mcp_server.py): Implements Anthropic's Model Context Protocol. Exposes internal diligence and search tools directly to tools like Claude Desktop, Cursor, and internal dashboards.
* [`infrastructure/telemetry_adapter.py`](file:///Users/yamijala/gitprojects/agenticAI/infrastructure/telemetry_adapter.py) and [`ops/prometheus_exporter.py`](file:///Users/yamijala/gitprojects/agenticAI/ops/prometheus_exporter.py): Live distributed tracing with W3C `traceparent` propagation and real-time Prometheus metric exports.

#### How It Solves Extantia's Problems
Partners can run custom due diligence workflows directly inside their day-to-day writing tools via MCP. Simultaneously, portfolio founders gain access to Extantia's internal intelligence tools with mathematical isolation between tenants.

---

## 4. The 10-Minute Technical Walkthrough Script

Use this script during live reviews with Carlota or external VC technical assessors.

```
+------------------------------------------------------------------------------------+
|                          10-MINUTE WALKTHROUGH TIMELINE                            |
+------------+-----------------------------------------------------------------------+
|  00 - 02m  |  Foundational Philosophy: Framework-free, pure Python, hexagonal      |
|  02 - 04m  |  Pillar 1: Tri-State Memory Quarantine & Document Supersession Demo   |
|  04 - 06m  |  Pillar 2: Deterministic State Machine & Token Budget Enforcement    |
|  06 - 08m  |  Pillar 3: Pearl's Causal Ladder & Red-Teaming Founder Projections   |
|  08 - 10m  |  Pillar 4 & Roadmap: Multi-Tenancy, MCP Server, 3-Sprint Deployment   |
+------------+-----------------------------------------------------------------------+
```

### Minute 0 to 2: Foundational Philosophy
**Goal:** Establish technical authority by demonstrating why common wrappers fail.

**Speaking Points:**
> "Most agent architectures in venture funds rely on LangChain, LlamaIndex, or AutoGen. In production, those frameworks hide failure modes behind layers of abstractions. When an LLM enters a retry loop or confuses a document version, debugging them is painful.
> 
> We built this repository from scratch in pure Python using hexagonal architecture. Every external dependency (vector search, relational storage, model providers) sits behind strict abstract interfaces. The core reasoning loop knows nothing about the network or specific database engines.
> 
> The entire test suite runs 166 passing tests in 5.2 seconds with zero network access required."

**Code to Open:**
* [`core/domain/ports.py`](file:///Users/yamijala/gitprojects/agenticAI/domain/ports.py): Show the clean protocol interfaces (`LLMPort`, `FactStorePort`, `EpisodeStorePort`, `ToolPort`).
* Run `.venv/bin/pytest tests/` in the terminal to show instant, clean test execution.

---

### Minute 2 to 4: Institutional Memory & Document Supersession (Pillar 1)
**Goal:** Address Extantia's core data infrastructure pain.

**Speaking Points:**
> "The number one issue in venture data layers is document supersession. When an investment team uploads draft four of a term sheet, standard vector search retrieves pieces of draft one and draft two because they share 90% vocabulary.
> 
> We resolved this through a Tri-State Memory Quarantine. Candidate facts never write directly to active memory. They land in a quarantine table. 
> 
> Our promoter evaluates the candidate against existing facts. If it detects a chronological conflict, it explicitly marks the prior fact retired with timestamps and promotes the new fact. When the agent builds context for an analyst, it only retrieves active facts, wrapping them inside XML delimiters to prevent prompt injections."

**Code to Open:**
* [`core/memory_promoter.py`](file:///Users/yamijala/gitprojects/agenticAI/core/memory_promoter.py): Highlight the admission workflow and contradiction checks.
* [`infrastructure/sqlite_fact_store.py`](file:///Users/yamijala/gitprojects/agenticAI/infrastructure/sqlite_fact_store.py): Show the dual schemas (`fact_quarantine` and `semantic_facts`) and temporal status columns.

---

### Minute 4 to 6: Deal Triage & Deterministic Budgeting (Pillar 2)
**Goal:** Show how the platform automates deal ingestion safely without runaway costs.

**Speaking Points:**
> "Autonomous agents frequently get stuck in circular reasoning loops when processing noisy input documents. If left unconstrained, API bills spiral.
> 
> Here, the execution engine is a strict Finite State Machine with a step counter and token ceiling. If an agent fails to extract clean parameters within its allocated step budget, it does not loop indefinitely. The FSM forces a transition to a degraded synthesis state and routes the pitch deck to the Ambiguity Queue for human analyst inspection.
> 
> Next, all tool calls pass through an in-memory policy engine that validates permissions and arguments against declarative JSON policies before execution."

**Code to Open:**
* [`core/state_machine.py`](file:///Users/yamijala/gitprojects/agenticAI/core/state_machine.py): Show the state transition table and hard budget checks.
* [`core/policy_engine.py`](file:///Users/yamijala/gitprojects/agenticAI/core/policy_engine.py): Show how tool policies intercept unauthorized calls.

---

### Minute 6 to 8: Counter-Thesis Due Diligence (Pillar 3)
**Goal:** Show how the platform avoids sycophancy and stress-tests deep tech claims.

**Speaking Points:**
> "Extantia invests in hard technology: energy security, industrial decarbonization, and hardware scaling. A typical LLM acts sycophantically, agreeing with whatever claims the founder wrote in their deck.
> 
> To provide real diligence value, we implemented Judea Pearl's Causality Ladder. 
> 
> We model the startup's economic and thermodynamic claims as a Structural Causal Model. Using safe AST formula evaluation, we run counterfactual simulations. We can ask: 'Given that the pilot plant consumed 50 megawatt-hours, what would the cost per ton have been had raw feedstock prices doubled?'
> 
> In parallel, our observational verifier runs partial correlation and d-separation tests on tabular accounting and operations data to verify that the founder's claimed causal links actually exist."

**Code to Open:**
* [`core/causal_reasoner.py`](file:///Users/yamijala/gitprojects/agenticAI/core/causal_reasoner.py): Walk through the counterfactual cycle (`abduction -> action -> prediction`).
* [`core/causal_graph.py`](file:///Users/yamijala/gitprojects/agenticAI/core/causal_graph.py): Show the safe AST expression parser (`_evaluate_node`) that blocks code injection.
* [`evals/track_c_causal/models/credit_default_risk.graphml`](file:///Users/yamijala/gitprojects/agenticAI/evals/track_c_causal/models/credit_default_risk.graphml): Show that financial risk models are already supported.

---

### Minute 8 to 10: Multi-Tenancy, MCP, and 3-Sprint Roadmap (Pillar 4)
**Goal:** Prove readiness to deploy inside Extantia in weeks, not months.

**Speaking Points:**
> "To serve both Extantia's partners and portfolio companies, every table enforces strict tenant partitioning. Cross-tenant leakage is impossible at the SQL query layer.
> 
> We also exposed our diligence tools via Anthropic's Model Context Protocol. Partners do not need to learn a new user interface. They can invoke our deal verification tools directly from Claude Desktop.
> 
> Because this foundational architecture is already built and tested, we do not need months of exploratory prototyping. We can roll this out in three focused sprints."

**Code to Open:**
* [`infrastructure/postgres_store.py`](file:///Users/yamijala/gitprojects/agenticAI/infrastructure/postgres_store.py): Highlight the `tenant_id` parameters across all queries.
* [`infrastructure/mcp_server.py`](file:///Users/yamijala/gitprojects/agenticAI/infrastructure/mcp_server.py): Show how kernel tools map to standard MCP tool schemas.

---

## 5. Three-Sprint Production Deployment Plan

Deploying this platform into Extantia follows an incremental, value-first roadmap:

```
+---------------------------------------------------------------------------------------+
|                            THREE-SPRINT DEPLOYMENT ROADMAP                            |
+--------------------+----------------------------------+-------------------------------+
| Sprint             | Focus Area                       | Shipped Capability            |
+--------------------+----------------------------------+-------------------------------+
| Sprint 1 (Wks 1-2) | Sourcing Connectors & Ingestion  | Automated deal extraction &   |
|                    |                                  | European registry lookups     |
+--------------------+----------------------------------+-------------------------------+
| Sprint 2 (Wks 3-4) | Data Room Versioning & Memory    | Document supersession fix for |
|                    |                                  | cap tables and side letters   |
+--------------------+----------------------------------+-------------------------------+
| Sprint 3 (Wks 5-6) | Adversarial Diligence & MCP Tools| Red-team counter-thesis memo  |
|                    |                                  | inside Claude Desktop         |
+--------------------+----------------------------------+-------------------------------+
```

### Sprint 1: Sourcing Connectors & Entity Ingestion (Weeks 1 to 2)
* **Objective:** Stabilize the inbound deal triage pipeline.
* **Engineering Work:**
  * Connect European public registries (German Handelsregister, UK Companies House) via existing `ToolPort` interfaces.
  * Ingest patent data streams from the European Patent Office (EPO) to cross-reference technical claims.
  * Connect webhooks for inbound deck attachments from email and Airtable.
* **Deliverable:** Automated extraction of inbound deck metrics into the `fact_quarantine` table with zero human data entry.

### Sprint 2: Data Room Ingestion & Versioning Engine (Weeks 3 to 4)
* **Objective:** Fix Extantia's document supersession and memory confusion.
* **Engineering Work:**
  * Build PDF and Excel ingestion adapters for data room files (financial models, cap tables, customer contracts).
  * Configure `MemoryPromoter` to track version tags across successive revisions of founder documents.
  * Integrate our hybrid BM25 and vector search engine to serve current, un-superseded deal context to analysts.
* **Deliverable:** Analysts query deals without fear of receiving outdated terms from previous drafts.

### Sprint 3: Adversarial Diligence & MCP Integration (Weeks 5 to 6)
* **Objective:** Deliver the counter-case generator directly into partner workflows.
* **Engineering Work:**
  * Deploy the `MCPServer` on Extantia's private internal cloud.
  * Configure Claude Desktop and Cursor endpoints for partner laptops.
  * Build the dual-agent red-teaming prompt harness to produce five-question IC punch-lists.
* **Deliverable:** Partners highlight a deal in Claude Desktop and click "Generate Counter-Case" to receive an adversarial analysis backed by verified causal checks.
