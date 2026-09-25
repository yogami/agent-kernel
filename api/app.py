"""FastAPI REST and WebSocket server for Agent Kernel with inspection cockpit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from fastapi.security import APIKeyHeader
from fastapi import Depends

from core.causal_gate import CausalPreFlightGate, CausalFalsificationViolationError
from core.causal_graph import StructuralCausalModel
from core.context_ram import ContextRAM
from core.execution_loop import ExecutionEngine
from core.guardrails import OutputGuardrails
from core.memory_promoter import MemoryPromoter
from core.policy_manager import TenantPolicyStore
from core.quarantine_cleaner import QuarantineCleaner
from core.sandbox.command_guard import CommandSecurityViolationError, inspect_arguments_for_command_injection
from core.sandbox.path_guard import PathSecurityViolationError, inspect_arguments_for_path_violations
from core.sandbox.runner import InProcessSandboxRunner
from domain.models import (
    AdmissionStatus,
    AuditEvent,
    AuditEventSeverity,
    CandidateFact,
    QuarantineRetentionPolicy,
    TenantSecurityPolicy,
    ToolCapability,
)
from infrastructure.audit_logger import audit_logger
from infrastructure.embedding_adapter import MockEmbeddingAdapter
from infrastructure.nli_adapter import MockNLIAdapter
from infrastructure.telemetry_adapter import InMemoryTracer
from infrastructure.vector_index import InMemoryVectorIndex
from infrastructure.llm_adapter import MockLLMAdapter
from infrastructure.sqlite_episode_store import SQLiteEpisodeStore
from infrastructure.sqlite_fact_store import SQLiteFactStore
from ops.benchmark import BenchmarkRunner
from ops.config_manager import ConfigManager
from ops.eval_runner import EvalRunner
from ops.prometheus_exporter import metrics_collector
from ops.self_healer import SelfHealer
from api.websocket_stream import ws_router
from tools.clinical_tools import (
    ClinicalAssertionCheckerTool,
    DeidentifyTextTool,
    FHIRValidatorTool,
    MedicalOntologyMapperTool,
)
from tools.multimodal_tools import (
    BiomechanicalAngleCalculatorTool,
    MovementSafetyCircuitBreakerTool,
    PoseLandmarkParserTool,
    VLMExerciseEvaluatorTool,
)
from tools.causal_tools import (
    CausalGraphQueryTool,
    ExplainCounterfactualAttributionTool,
    SimulateCausalInterventionTool,
)
from core.causal_reasoner import CausalReasoner
from tools.registry import ToolRegistry
from tools.system_tools import CalculatorTool, DateValidatorTool



import os
from domain.serialization import to_dict
from fastapi import Request, WebSocket
from api.ablation_router import ablation_router

API_KEY_NAME = "X-API-Key"
PUBLIC_PATHS = {"/", "/metrics", "/docs", "/openapi.json", "/favicon.ico", "/v1/ablation/cases", "/v1/ablation/ladder"}


def _is_public_request(request: Request) -> bool:
    path = request.url.path
    if path in PUBLIC_PATHS or path.startswith("/static"):
        return True
    return False


def _extract_request_key(request: Request, websocket: WebSocket) -> Optional[str]:
    if request:
        return request.headers.get(API_KEY_NAME)
    if websocket:
        return websocket.headers.get(API_KEY_NAME)
    return None


def _validate_provided_key(api_key: Optional[str], expected_key: str) -> str:
    if not api_key or api_key != expected_key:
        raise HTTPException(status_code=403, detail="Could not validate credentials")
    return api_key


def _assert_valid_key(api_key: Optional[str], expected_key: Optional[str]) -> str:
    if not expected_key:
        raise HTTPException(
            status_code=500,
            detail="Server security configuration error: AGENT_KERNEL_API_KEY is not configured.",
        )
    return _validate_provided_key(api_key, expected_key)



def verify_api_key(request: Request = None, websocket: WebSocket = None):
    if request and _is_public_request(request):
        return None
    api_key = _extract_request_key(request, websocket)
    expected_key = os.getenv("AGENT_KERNEL_API_KEY")
    return _assert_valid_key(api_key, expected_key)


from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Graceful shutdown logic
    if hasattr(episode_store, "close"):
        await episode_store.close()
    if hasattr(fact_store, "close"):
        await fact_store.close()

app = FastAPI(
    title="Autonomous Agent Kernel API",
    version="1.0.0",
    description="Battle-hardened Agent Kernel with Tri-State Memory Write Gates, Deterministic FSM, and Monotonic Non-Regression LLM-Ops.",
    dependencies=[Depends(verify_api_key)],
    lifespan=lifespan,
)

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ablation_router)


tracer = InMemoryTracer()


@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    carrier = dict(request.headers)
    parent_ctx = tracer.extract_traceparent(carrier)
    span_name = f"http {request.method} {request.url.path}"
    with tracer.start_span(
        span_name,
        parent_context=parent_ctx,
        attributes={
            "http.method": request.method,
            "http.url": str(request.url),
            "http.route": request.url.path,
        },
    ) as span:
        try:
            response = await call_next(request)
            span.set_attribute("http.status_code", response.status_code)
            if response.status_code >= 500:
                span.set_status("ERROR", f"HTTP {response.status_code}")
            else:
                span.set_status("OK")
            response.headers["traceparent"] = span.context.to_traceparent()
            return response
        except Exception as exc:
            span.record_exception(exc)
            raise

ENVIRONMENT = os.getenv("AGENT_KERNEL_ENV", "local")

if ENVIRONMENT == "production":
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise ValueError("DATABASE_URL must be set in production")
    from infrastructure.postgres_store import PostgresEpisodeStore, PostgresFactStore
    episode_store = PostgresEpisodeStore(db_url)
    fact_store = PostgresFactStore(db_url)
else:
    DATA_DIR = Path(os.getenv("AGENT_KERNEL_DATA_DIR", "data"))
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_path = str(DATA_DIR / "agent_kernel.sqlite")
    episode_store = SQLiteEpisodeStore(db_path)
    fact_store = SQLiteFactStore(db_path)

config_manager = ConfigManager("config/packs")
context_ram = ContextRAM(fact_store)
causal_reasoner = CausalReasoner()
vector_index = InMemoryVectorIndex()

if ENVIRONMENT == "production":
    from infrastructure.embedding_adapter import OpenAICompatibleEmbeddingAdapter
    from infrastructure.nli_adapter import LLMBasedNLIAdapter
    from infrastructure.llm_adapter import AsyncOpenRouterAdapter
    from core.sandbox.runner import SubprocessSandboxRunner

    embedding_adapter = OpenAICompatibleEmbeddingAdapter()
    llm_adapter = AsyncOpenRouterAdapter()
    nli_adapter = LLMBasedNLIAdapter(llm_provider=llm_adapter)
    sandbox_runner = SubprocessSandboxRunner(
        default_timeout_s=float(os.getenv("SANDBOX_TIMEOUT", "5.0")),
        max_memory_mb=int(os.getenv("SANDBOX_MEMORY_MB", "256"))
    )
else:
    embedding_adapter = MockEmbeddingAdapter()
    nli_adapter = MockNLIAdapter()
    llm_adapter = MockLLMAdapter()
    sandbox_runner = InProcessSandboxRunner(default_timeout_s=5.0)

    # Configure realistic scripted behaviors for mock adapter
    llm_adapter.set_scripted_response(
        "deidentify",
        {
            "content": "Running de-identification scrubber on clinical text.",
            "tool_calls": [{
                "id": "call_deid_1",
                "type": "function",
                "function": {
                    "name": "deidentify_clinical_text",
                    "arguments": json.dumps({"text": "Patient: Herr Schmidt (PAT-1001), Charité Berlin. Date: 12.04.1978"}),
                },
            }],
            "tokens_in": 45,
            "tokens_out": 20,
            "cost_usd": 0.0001,
        },
    )
    llm_adapter.set_scripted_response(
        "patient_001",
        {
            "content": "I am checking the memory trace.",
            "tokens_in": 10,
            "tokens_out": 5,
            "cost_usd": 0.00005,
        },
    )

quarantine_cleaner = QuarantineCleaner(fact_store)
promoter = MemoryPromoter(
    fact_store,
    vector_index=vector_index,
    embedding_provider=embedding_adapter,
    nli_provider=nli_adapter,
    causal_reasoner=causal_reasoner,
)
eval_runner = EvalRunner("evals")
self_healer = SelfHealer(config_manager, eval_runner)
benchmark_runner = BenchmarkRunner()

tool_registry = ToolRegistry(sandbox_runner=sandbox_runner)
tool_registry.register(DeidentifyTextTool())
tool_registry.register(MedicalOntologyMapperTool())
tool_registry.register(FHIRValidatorTool())
tool_registry.register(ClinicalAssertionCheckerTool())
tool_registry.register(CalculatorTool())
tool_registry.register(DateValidatorTool())
tool_registry.register(PoseLandmarkParserTool())
tool_registry.register(BiomechanicalAngleCalculatorTool())
tool_registry.register(MovementSafetyCircuitBreakerTool())
tool_registry.register(VLMExerciseEvaluatorTool())
tool_registry.register(SimulateCausalInterventionTool(causal_reasoner))
tool_registry.register(ExplainCounterfactualAttributionTool(causal_reasoner))
tool_registry.register(CausalGraphQueryTool(causal_reasoner))

policy_store = TenantPolicyStore()
causal_pre_flight_gate = CausalPreFlightGate(alpha=0.05, apply_fdr=True)

from ops.prometheus_exporter import metrics_collector
execution_engine = ExecutionEngine(
    llm=llm_adapter,
    episode_store=episode_store,
    tool_registry=tool_registry,
    context_ram=context_ram,
    guardrails=OutputGuardrails(),
    tracer=tracer,
    metrics_collector=metrics_collector,
)


class ChatRequest(BaseModel):
    user_input: str
    session_id: str = Field(default="default_session")
    tenant_id: str = Field(default="default_tenant")
    max_steps: int = 5


class CandidateFactRequest(BaseModel):
    session_id: str
    source_episode_id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.95
    tenant_id: str = Field(default="default_tenant")


class PurgeQuarantineRequest(BaseModel):
    max_age_days: int = 30
    tenant_id: str = "default_tenant"
    statuses: list[str] = Field(default_factory=lambda: ["REJECTED"])


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    active_pack = config_manager.get_active_pack()
    try:
        # Ping the database
        episode_store.get_recent_turns("health_ping", limit=1)
        db_status = "OK"
    except Exception as exc:
        db_status = f"ERROR: {exc}"
        raise HTTPException(status_code=503, detail="Database unhealthy")
        
    return {
        "status": "HEALTHY",
        "database": db_status,
        "active_pack_version": active_pack.version,
        "active_sha256": active_pack.sha256_hash,
        "registered_tools": [t.name for t in tool_registry.list_tools()],
    }


@app.post("/api/chat")
def chat_turn(req: ChatRequest) -> dict[str, Any]:
    active_pack = config_manager.get_active_pack()
    try:
        turn = execution_engine.run_turn(
            user_input=req.user_input,
            session_id=req.session_id,
            active_config=active_pack,
            max_steps=req.max_steps,
            tenant_id=req.tenant_id,
        )
        return {
            "status": "SUCCESS",
            "turn": to_dict(turn),
        }
    except Exception as exc:
        import logging
        logging.error(f"Internal server error in chat_turn: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/api/chat/stream")
async def chat_turn_stream(req: ChatRequest):
    """Server-Sent Events (SSE) streaming endpoint for agent execution."""
    active_pack = config_manager.get_active_pack()

    async def event_generator():
        try:
            async for event in execution_engine.run_turn_stream(
                user_input=req.user_input,
                session_id=req.session_id,
                active_config=active_pack,
                max_steps=req.max_steps,
                tenant_id=req.tenant_id,
            ):
                payload_str = json.dumps({
                    "event": event.event.value,
                    "payload": event.payload,
                    "timestamp": event.timestamp.isoformat(),
                })
                yield f"data: {payload_str}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            import logging
            logging.error(f"Internal server error in chat_turn_stream: {exc}", exc_info=True)
            err_str = json.dumps({"event": "error", "error": "Internal server error"})
            yield f"data: {err_str}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/traces")
def get_traces(limit: int = 50, trace_id: str | None = None) -> list[dict[str, Any]]:
    """Retrieve collected distributed tracing spans."""
    spans = tracer.get_spans(trace_id=trace_id, limit=limit)
    return [to_dict(s) for s in spans]


@app.get("/api/episodes")
def list_episodes(limit: int = 50, tenant_id: str = "default_tenant") -> list[dict[str, Any]]:
    episodes = episode_store.list_episodes(limit=limit, tenant_id=tenant_id)
    return [to_dict(ep) for ep in episodes]


@app.get("/api/turns/{session_id}")
def get_turns(session_id: str, limit: int = 20, tenant_id: str = "default_tenant") -> list[dict[str, Any]]:
    turns = episode_store.get_recent_turns(session_id, limit=limit, tenant_id=tenant_id)
    return [to_dict(t) for t in turns]


@app.get("/api/quarantine")
def get_quarantine(limit: int = 100, tenant_id: str = "default_tenant") -> list[dict[str, Any]]:
    records = fact_store.get_quarantine_records(limit=limit, tenant_id=tenant_id)
    return [to_dict(r) for r in records]


@app.post("/api/quarantine/submit")
def submit_quarantine_fact(req: CandidateFactRequest) -> dict[str, Any]:
    cand = CandidateFact(
        source_episode_id=req.source_episode_id,
        session_id=req.session_id,
        subject=req.subject,
        predicate=req.predicate,
        object=req.object,
        confidence=req.confidence,
        tenant_id=req.tenant_id,
    )
    cand_id = promoter.submit_candidate(cand)
    promoted, detail = promoter.evaluate_and_promote(cand_id)
    return {
        "candidate_id": cand_id,
        "promoted": promoted,
        "detail": detail,
    }


@app.post("/api/quarantine/purge")
def purge_quarantine(req: PurgeQuarantineRequest) -> dict[str, Any]:
    statuses = [AdmissionStatus(s) for s in req.statuses if s in [st.value for st in AdmissionStatus]]
    policy = QuarantineRetentionPolicy(max_age_days=req.max_age_days, statuses_to_purge=statuses)
    return quarantine_cleaner.run_purge_cycle(policy=policy, tenant_id=req.tenant_id)


@app.get("/api/facts")
def get_facts(session_id: str | None = None, tenant_id: str = "default_tenant") -> list[dict[str, Any]]:
    facts = fact_store.query_all_semantic_facts(session_id=session_id, tenant_id=tenant_id)
    return [to_dict(f) for f in facts]


@app.post("/api/eval/{suite_name}")
def run_eval_suite(suite_name: str) -> dict[str, Any]:
    active_pack = config_manager.get_active_pack()
    return eval_runner.run_suite(suite_name, active_pack)


@app.post("/api/healer/run")
def trigger_self_healing() -> dict[str, Any]:
    return self_healer.run_repair_cycle()


@app.post("/api/benchmark/run")
def run_benchmark() -> dict[str, Any]:
    return benchmark_runner.run_benchmark()


class CausalSimulateRequest(BaseModel):
    intervention: dict[str, Any] = Field(default_factory=dict)
    patient_context: dict[str, Any] = Field(default_factory=dict)

class CounterfactualRequest(BaseModel):
    factual_evidence: dict[str, Any] = Field(default_factory=dict)
    hypothetical_action: dict[str, Any] = Field(default_factory=dict)
    adverse_outcome_target: str = ""

@app.post("/api/causal/simulate")
def simulate_causal_intervention(req: CausalSimulateRequest) -> dict[str, Any]:
    """Pearl's Level 2: Simulate do(Action) and evaluate safety on a mutilated SCM."""
    return causal_reasoner.simulate_intervention_safety(
        proposed_action=req.intervention,
        patient_evidence=req.patient_context,
    )


@app.post("/api/causal/counterfactual")
def evaluate_counterfactual(req: CounterfactualRequest) -> dict[str, Any]:
    """Pearl's Level 3: Evaluate counterfactual attribution for an outcome."""
    return causal_reasoner.explain_counterfactual_attribution(
        actual_evidence=req.factual_evidence,
        hypothetical_alternative=req.hypothetical_action,
        observed_bad_outcome=req.adverse_outcome_target,
    )


app.include_router(ws_router)


# -------------------------------------------------------------
# Enterprise Gateway Proxy Endpoints (v1)
# -------------------------------------------------------------

class MemoryAdmitRequest(BaseModel):
    tenant_id: str = "default_tenant"
    session_id: str = "default_session"
    source_episode_id: str = "external_agent"
    subject: str
    predicate: str
    object: str
    confidence: float = 0.95


class CausalVerifyRequest(BaseModel):
    tenant_id: str = "default_tenant"
    scm_id: str
    proposed_intervention: str
    target_outcome: str
    observational_data: list[dict[str, float]] | None = None
    scm_graphml: str | None = None


class ToolExecuteRequest(BaseModel):
    tenant_id: str = "default_tenant"
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    capability: ToolCapability | None = None
    timeout_seconds: float = 5.0


class TenantPolicyUpdateRequest(BaseModel):
    allowed_egress_domains: list[str] | None = None
    quarantine_retention_days: int | None = None
    command_injection_guard_enabled: bool | None = None
    path_confinement_enabled: bool | None = None
    causal_verification_strictness: str | None = None
    rate_limit_rpm: int | None = None


@app.post("/v1/memory/admit")
def gateway_memory_admit(req: MemoryAdmitRequest) -> dict[str, Any]:
    """Write-Path Memory Admission Gate proxy for external agents.

    Evaluates candidate facts against tenant memory using the NLI contradiction
    classifier and causal graph consistency before committing to permanent storage.
    """
    cand = CandidateFact(
        source_episode_id=req.source_episode_id,
        session_id=req.session_id,
        subject=req.subject,
        predicate=req.predicate,
        object=req.object,
        confidence=req.confidence,
        tenant_id=req.tenant_id,
    )
    cand_id = promoter.submit_candidate(cand)
    promoted, detail = promoter.evaluate_and_promote(cand_id)

    if promoted:
        audit_logger.log_event(
            AuditEvent(
                tenant_id=req.tenant_id,
                event_type="memory.admitted",
                severity=AuditEventSeverity.INFO,
                actor="external_agent",
                resource=f"{req.subject}:{req.predicate}",
                action="admit_fact",
                outcome="ALLOW",
                details={"object": req.object, "detail": detail},
            )
        )
        return {
            "admitted": True,
            "status": "PROMOTED",
            "candidate_id": cand_id,
            "detail": detail,
            "quarantined": False,
        }
    else:
        audit_logger.log_memory_quarantine(
            tenant_id=req.tenant_id,
            subject=req.subject,
            predicate=req.predicate,
            obj=req.object,
            reason=detail,
        )
        return {
            "admitted": False,
            "status": "QUARANTINED",
            "candidate_id": cand_id,
            "detail": detail,
            "quarantined": True,
        }


@app.post("/v1/causal/verify")
def gateway_causal_verify(req: CausalVerifyRequest) -> dict[str, Any]:
    """Causal Pre-Flight Verification proxy for external agents.

    Verifies that a proposed intervention and target outcome adhere to
    Pearl's d-separation constraints and observational data before tool execution.
    """
    scm: StructuralCausalModel | None = None
    if req.scm_graphml:
        try:
            scm = StructuralCausalModel.from_graphml(req.scm_graphml)
        except Exception as exc:
            import logging
            logging.error(f"GraphML parsing error: {exc}", exc_info=True)
            raise HTTPException(status_code=400, detail="Invalid GraphML schema provided.")
    else:
        model_file = Path("evals/track_c_causal/models") / f"{req.scm_id}.graphml"
        if model_file.exists():
            scm = StructuralCausalModel.from_graphml(model_file.read_text(encoding="utf-8"))

    if scm is None:
        scm = StructuralCausalModel(req.scm_id)
        scm.add_node(req.proposed_intervention)
        scm.add_node(req.target_outcome)
        scm.add_edge(req.proposed_intervention, req.target_outcome)

    obs_data = req.observational_data
    if obs_data is None:
        data_file = Path("evals/track_c_causal/data") / f"{req.scm_id}_data.json"
        if not data_file.exists():
            data_file = Path("evals/track_c_causal/data") / f"{req.scm_id}.json"
        if data_file.exists():
            with open(data_file, "r", encoding="utf-8") as f:
                obs_data = json.load(f)

    if not obs_data:
        raise HTTPException(
            status_code=400,
            detail="Observational tabular data is required for causal falsification verification.",
        )

    report = causal_pre_flight_gate.evaluate_scm_consistency(
        scm=scm,
        observational_data=obs_data,
        intervention_var=req.proposed_intervention,
        outcome_var=req.target_outcome,
    )

    if not report.is_valid:
        violations = [v.violation_detail for v in report.violations if v.violation_detail]
        audit_logger.log_causal_denial(
            tenant_id=req.tenant_id,
            scm_id=req.scm_id,
            intervention=req.proposed_intervention,
            violations=violations,
        )
        return {
            "verified": False,
            "status": "FALSIFIED",
            "violations": violations,
            "testable_independencies_count": len(violations),
        }

    audit_logger.log_event(
        AuditEvent(
            tenant_id=req.tenant_id,
            event_type="causal.verified",
            severity=AuditEventSeverity.INFO,
            actor="external_agent",
            resource=req.scm_id,
            action="verify_intervention",
            outcome="ALLOW",
            details={
                "intervention": req.proposed_intervention,
                "target_outcome": req.target_outcome,
            },
        )
    )
    return {
        "verified": True,
        "status": "VERIFIED",
        "violations": [],
        "testable_independencies_count": 0,
    }


@app.post("/v1/tools/execute")
def gateway_tool_execute(req: ToolExecuteRequest) -> dict[str, Any]:
    """Sandboxed Tool Execution proxy for external agents.

    Applies tenant policy confinement, command injection filters, path boundary
    checks, and hard execution deadlines.
    """
    policy = policy_store.get_policy(req.tenant_id)

    if policy.command_injection_guard_enabled:
        try:
            inspect_arguments_for_command_injection(req.arguments)
        except CommandSecurityViolationError as exc:
            audit_logger.log_tool_violation(
                tenant_id=req.tenant_id,
                tool_name=req.tool_name,
                violation_type="command_injection",
                detail=str(exc),
            )
            return {
                "success": False,
                "output": None,
                "error": str(exc),
                "security_blocked": True,
            }

    if policy.path_confinement_enabled:
        try:
            inspect_arguments_for_path_violations(req.arguments)
        except PathSecurityViolationError as exc:
            audit_logger.log_tool_violation(
                tenant_id=req.tenant_id,
                tool_name=req.tool_name,
                violation_type="path_traversal",
                detail=str(exc),
            )
            return {
                "success": False,
                "output": None,
                "error": str(exc),
                "security_blocked": True,
            }

    registered = tool_registry.get(req.tool_name)
    if registered is None:
        raise HTTPException(status_code=404, detail=f"Tool '{req.tool_name}' is not registered.")

    tool_res = tool_registry.execute(req.tool_name, req.arguments)
    is_success = not tool_res.is_error
    outcome = "ALLOW" if is_success else "ERROR"
    audit_logger.log_event(
        AuditEvent(
            tenant_id=req.tenant_id,
            event_type="tool.executed",
            severity=AuditEventSeverity.INFO if is_success else AuditEventSeverity.WARNING,
            actor="external_agent",
            resource=req.tool_name,
            action="execute_tool",
            outcome=outcome,
            details={"output": str(tool_res.output or "")[:200], "error": tool_res.error_message or ""},
        )
    )
    return {
        "success": is_success,
        "output": tool_res.output,
        "error": tool_res.error_message,
        "security_blocked": False,
    }


@app.get("/v1/policies/{tenant_id}")
def get_tenant_policy(tenant_id: str) -> dict[str, Any]:
    """Retrieve active security policy configuration for a tenant."""
    policy = policy_store.get_policy(tenant_id)
    return to_dict(policy)


@app.put("/v1/policies/{tenant_id}")
def update_tenant_policy(tenant_id: str, req: TenantPolicyUpdateRequest) -> dict[str, Any]:
    """Update security policy rules for a tenant."""
    current = policy_store.get_policy(tenant_id)
    updated_dict = to_dict(current)
    req_dict = req.model_dump(exclude_unset=True)
    for k, v in req_dict.items():
        if v is not None:
            updated_dict[k] = v

    new_policy = TenantSecurityPolicy(**updated_dict)
    policy_store.set_policy(new_policy)
    audit_logger.log_event(
        AuditEvent(
            tenant_id=tenant_id,
            event_type="policy.updated",
            severity=AuditEventSeverity.INFO,
            actor="admin",
            resource=tenant_id,
            action="update_policy",
            outcome="ALLOW",
            details=req_dict,
        )
    )
    return to_dict(new_policy)


@app.get("/v1/audit/logs")
def get_audit_logs(
    format: str = "json",
    tenant_id: str | None = None,
    severity: str | None = None,
    limit: int = 100,
):
    """Export compliance audit logs in JSON or ArcSight Common Event Format (CEF)."""
    if format.lower() == "cef":
        cef_lines = audit_logger.export_cef(tenant_id=tenant_id, limit=limit)
        return PlainTextResponse("\n".join(cef_lines), media_type="text/plain")

    events = audit_logger.get_events(tenant_id=tenant_id, severity=severity, limit=limit)
    return [to_dict(e) for e in events]


@app.get("/metrics", response_class=PlainTextResponse)
def prometheus_metrics() -> str:
    """Expose application metrics in standard Prometheus exposition format."""
    return metrics_collector.generate_prometheus_format()


# Static cockpit interface
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def serve_dashboard() -> FileResponse:
    index_file = static_dir / "index.html"
    if not index_file.exists():
        return FileResponse(static_dir / "index.html")
    return FileResponse(index_file)
