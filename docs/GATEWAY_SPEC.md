# 🛡️ Enterprise Agent Safety & Memory Verification Gateway Specification

**Version:** 1.0.0  
**Base URL:** `http://localhost:8000`  
**Target Runtimes:** LangGraph, CrewAI, AutoGen, Claude Code, Semantic Kernel, and custom autonomous agent frameworks.

---

## 1. Overview & Architecture

Modern agent frameworks execute tools and commit facts into vector databases without safety verification. When autonomous agents operate in regulated domains (healthcare, banking, defense, enterprise operations), un-gated memory writes and unconstrained tool execution lead to severe risks:
1. **Contradiction Ingestion:** Un-gated writes accept false or contradictory claims into long-term memory.
2. **Confounded Actions:** Agents trigger interventions without checking if assumed causal links hold in empirical data.
3. **Sandbox Escapes:** Exploitative prompts inject shell metacharacters, probe root directories, and exfiltrate secrets.

The **Agent Kernel Enterprise Gateway** operates as a framework-agnostic sidecar proxy. External agents invoke the gateway prior to committing memory, executing interventions, or running tools.

```
+-----------------------------------------------------------------------------------+
|               External Agent Runtimes (LangGraph / CrewAI / Claude Code)          |
+-----------------------------------------------------------------------------------+
        | (POST /v1/memory/admit)      | (POST /v1/causal/verify)   | (POST /v1/tools/execute)
        v                              v                            v
+-----------------------------------------------------------------------------------+
|                        Agent Kernel Enterprise Safety Gateway                     |
|                                                                                   |
|   +-----------------------+  +-----------------------+  +---------------------+   |
|   |  Tri-State NLI Gate   |  |   Causal Pre-Flight   |  |  Tool Confinement   |   |
|   | (Contradiction Check) |  |   (FDR Verification)  |  |  (Path/Cmd Guard)   |   |
|   +-----------------------+  +-----------------------+  +---------------------+   |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |                Multi-Tenant Security Policy & Rate Limiter                |   |
|   +---------------------------------------------------------------------------+   |
|                                                                                   |
|   +---------------------------------------------------------------------------+   |
|   |           CEF / JSON Audit Event Streaming (Splunk / Datadog SIEM)        |   |
|   +---------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------+
```

---

## 2. API Endpoints Reference

### 2.1 Memory Admission Gate (`POST /v1/memory/admit`)

Evaluates whether a candidate fact can be safely admitted into persistent memory without contradicting established facts.

#### Request Body
```json
{
  "tenant_id": "corp_finance",
  "session_id": "session_881",
  "source_episode_id": "turn_12",
  "subject": "acme_inc",
  "predicate": "credit_rating",
  "object": "BBB+",
  "confidence": 0.94
}
```

#### Response (Success: Fact Promoted)
```json
{
  "admitted": true,
  "status": "PROMOTED",
  "candidate_id": "cand_9f81a2b",
  "detail": "Admitted: High confidence, verified novel fact",
  "quarantined": false
}
```

#### Response (Contradiction Detected: Quarantined)
```json
{
  "admitted": false,
  "status": "QUARANTINED",
  "candidate_id": "cand_4a71c8e",
  "detail": "Quarantined: Contradicts active fact (existing: 'AAA' vs proposed: 'BBB+')",
  "quarantined": true
}
```

---

### 2.2 Causal Pre-Flight Verification (`POST /v1/causal/verify`)

Verifies that a proposed intervention and target outcome adhere to Pearl's d-separation properties and observational data before taking action.

#### Request Body
```json
{
  "tenant_id": "ops_cluster",
  "scm_id": "cluster_autoscaling",
  "proposed_intervention": "replica_count",
  "target_outcome": "p99_latency"
}
```

#### Response (Verified Safe)
```json
{
  "verified": true,
  "status": "VERIFIED",
  "violations": [],
  "testable_independencies_count": 0
}
```

#### Response (Falsified Mechanism Denied)
```json
{
  "verified": false,
  "status": "FALSIFIED",
  "violations": [
    "Independence routing_weight _||_ rtt_latency | ['link_congestion'] violated (p=0.00012, FDR-adjusted threshold=0.003)"
  ],
  "testable_independencies_count": 1
}
```

---

### 2.3 Sandboxed Tool Execution (`POST /v1/tools/execute`)

Executes a tool within a sandboxed runtime, enforcing path boundaries, shell injection prevention, and execution timeouts.

#### Request Body
```json
{
  "tenant_id": "dev_team",
  "tool_name": "calculator",
  "arguments": {
    "expression": "42 * 10"
  },
  "timeout_seconds": 3.0
}
```

#### Response (Success)
```json
{
  "success": true,
  "output": 420.0,
  "error": null,
  "security_blocked": false
}
```

#### Response (Security Violation Intercepted)
```json
{
  "success": false,
  "output": null,
  "error": "Security Violation: Command injection metacharacters detected in arguments.",
  "security_blocked": true
}
```

---

### 2.4 Multi-Tenant Policy Management (`GET / PUT /v1/policies/{tenant_id}`)

#### GET `/v1/policies/{tenant_id}`
Returns active security configuration for the tenant.

#### PUT `/v1/policies/{tenant_id}`
Updates egress allowlists, timeout ceilings, or guardrail flags.

```json
{
  "allowed_egress_domains": ["api.github.com", "fhir.hospital.org"],
  "quarantine_retention_days": 60,
  "command_injection_guard_enabled": true,
  "path_confinement_enabled": true,
  "causal_verification_strictness": "STRICT",
  "rate_limit_rpm": 300
}
```

---

### 2.5 Compliance & SIEM Audit Export (`GET /v1/audit/logs`)

Streams security events in JSON or standard ArcSight Common Event Format (CEF).

#### Query Parameters
- `format`: `json` (default) or `cef`
- `tenant_id`: optional tenant filter
- `severity`: `INFO`, `WARNING`, `CRITICAL`
- `limit`: integer (default 100)

#### CEF Output Example
```text
CEF:0|AgentKernel|SecurityGateway|1.0|tool.security_violation|tool.security_violation|9|src=external_agent act=execute_tool res=bash_shell outcome=DENY cs1=tenant_a cs1Label=tenant_id msg=Command injection metacharacters detected
CEF:0|AgentKernel|SecurityGateway|1.0|memory.contradiction_quarantined|memory.contradiction_quarantined|6|src=memory_promoter act=admit_fact res=patient:diagnosis outcome=QUARANTINE cs1=tenant_a cs1Label=tenant_id msg=Contradicts active fact
```

---

## 3. Client Integration Examples

### 3.1 LangGraph Drop-in Integration (Python)

```python
import httpx

GATEWAY_URL = "http://localhost:8000"

def write_memory_with_gate(tenant_id: str, subject: str, predicate: str, obj: str) -> bool:
    """Pre-write hook for LangGraph agents before committing to long-term memory."""
    payload = {
        "tenant_id": tenant_id,
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "confidence": 0.95,
    }
    resp = httpx.post(f"{GATEWAY_URL}/v1/memory/admit", json=payload)
    data = resp.json()
    if not data.get("admitted", False):
        print(f"Write blocked by Kernel Gate: {data.get('detail')}")
        return False
    return True

def execute_tool_with_sandbox(tenant_id: str, tool_name: str, args: dict) -> dict:
    """Pre-execution hook for LangGraph tool nodes."""
    payload = {
        "tenant_id": tenant_id,
        "tool_name": tool_name,
        "arguments": args,
    }
    resp = httpx.post(f"{GATEWAY_URL}/v1/tools/execute", json=payload)
    return resp.json()
```

### 3.2 CrewAI Tool Guardrail Integration (Python)

```python
from crewai.tools import BaseTool
import httpx

class VerifiedGatewayTool(BaseTool):
    name: str = "VerifiedGatewayTool"
    description: str = "Routes execution through Agent Kernel Confinement Layer"

    def _run(self, command: str) -> str:
        resp = httpx.post(
            "http://localhost:8000/v1/tools/execute",
            json={
                "tenant_id": "crew_workspace",
                "tool_name": "calculator",
                "arguments": {"expression": command},
            },
        )
        res = resp.json()
        if res.get("security_blocked"):
            return f"Error: Command blocked by safety policy: {res.get('error')}"
        return str(res.get("output"))
```

### 3.3 Command Line cURL Examples

```bash
# 1. Check memory admission
curl -X POST http://localhost:8000/v1/memory/admit \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "test_corp", "subject": "server", "predicate": "status", "object": "healthy"}'

# 2. Verify causal intervention
curl -X POST http://localhost:8000/v1/causal/verify \
  -H "Content-Type: application/json" \
  -d '{"tenant_id": "test_corp", "scm_id": "cluster_autoscaling", "proposed_intervention": "replica_count", "target_outcome": "p99_latency"}'

# 3. Stream CEF audit logs for Splunk
curl "http://localhost:8000/v1/audit/logs?format=cef&limit=20"
```
