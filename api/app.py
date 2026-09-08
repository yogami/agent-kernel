"""FastAPI REST and WebSocket server for Agent Kernel with inspection cockpit."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.context_ram import ContextRAM
from core.execution_loop import ExecutionEngine
from core.guardrails import OutputGuardrails
from core.memory_promoter import MemoryPromoter
from domain.models import CandidateFact
from infrastructure.llm_adapter import MockLLMAdapter
from infrastructure.sqlite_episode_store import SQLiteEpisodeStore
from infrastructure.sqlite_fact_store import SQLiteFactStore
from ops.benchmark import BenchmarkRunner
from ops.config_manager import ConfigManager
from ops.eval_runner import EvalRunner
from ops.prometheus_exporter import metrics_collector
from ops.self_healer import SelfHealer
from api.websocket_stream import ws_router
from fastapi.responses import PlainTextResponse
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



app = FastAPI(
    title="Autonomous Agent Kernel API",
    version="1.0.0",
    description="Battle-hardened Agent Kernel with Tri-State Memory Write Gates, Deterministic FSM, and Monotonic Non-Regression LLM-Ops.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize singletons
DATA_DIR = Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)

db_path = str(DATA_DIR / "agent_kernel.sqlite")
episode_store = SQLiteEpisodeStore(db_path)
fact_store = SQLiteFactStore(db_path)
config_manager = ConfigManager("config/packs")
context_ram = ContextRAM(fact_store)
causal_reasoner = CausalReasoner()
promoter = MemoryPromoter(fact_store, causal_reasoner=causal_reasoner)
eval_runner = EvalRunner("evals")
self_healer = SelfHealer(config_manager, eval_runner)
benchmark_runner = BenchmarkRunner()

tool_registry = ToolRegistry()
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



llm_adapter = MockLLMAdapter()

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

execution_engine = ExecutionEngine(
    llm=llm_adapter,
    episode_store=episode_store,
    tool_registry=tool_registry,
    context_ram=context_ram,
    guardrails=OutputGuardrails(),
)


class ChatRequest(BaseModel):
    user_input: str
    session_id: str = Field(default="default_session")
    max_steps: int = 5


class CandidateFactRequest(BaseModel):
    session_id: str
    source_episode_id: str
    subject: str
    predicate: str
    object: str
    confidence: float = 0.95


@app.get("/api/health")
def health_check() -> dict[str, Any]:
    active_pack = config_manager.get_active_pack()
    return {
        "status": "HEALTHY",
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
        )
        return {
            "status": "SUCCESS",
            "turn": turn.model_dump(mode="json"),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/api/episodes")
def list_episodes(limit: int = 50) -> list[dict[str, Any]]:
    episodes = episode_store.list_episodes(limit=limit)
    return [ep.model_dump(mode="json") for ep in episodes]


@app.get("/api/turns/{session_id}")
def get_turns(session_id: str, limit: int = 20) -> list[dict[str, Any]]:
    turns = episode_store.get_recent_turns(session_id, limit=limit)
    return [t.model_dump(mode="json") for t in turns]


@app.get("/api/quarantine")
def get_quarantine(limit: int = 100) -> list[dict[str, Any]]:
    records = fact_store.get_quarantine_records(limit=limit)
    return [r.model_dump(mode="json") for r in records]


@app.post("/api/quarantine/submit")
def submit_quarantine_fact(req: CandidateFactRequest) -> dict[str, Any]:
    cand = CandidateFact(
        source_episode_id=req.source_episode_id,
        session_id=req.session_id,
        subject=req.subject,
        predicate=req.predicate,
        object=req.object,
        confidence=req.confidence,
    )
    cand_id = promoter.submit_candidate(cand)
    promoted, detail = promoter.evaluate_and_promote(cand_id)
    return {
        "candidate_id": cand_id,
        "promoted": promoted,
        "detail": detail,
    }


@app.get("/api/facts")
def get_facts(session_id: str | None = None) -> list[dict[str, Any]]:
    facts = fact_store.query_all_semantic_facts(session_id=session_id)
    return [f.model_dump(mode="json") for f in facts]


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


@app.post("/api/causal/simulate")
def simulate_causal_intervention(payload: dict[str, Any]) -> dict[str, Any]:
    """Pearl's Level 2: Simulate do(Action) and evaluate safety on a mutilated SCM."""
    intervention = payload.get("intervention", {})
    context = payload.get("patient_context", {})
    return causal_reasoner.simulate_intervention_safety(
        proposed_action=intervention,
        patient_evidence=context,
    )


@app.post("/api/causal/counterfactual")
def evaluate_counterfactual(payload: dict[str, Any]) -> dict[str, Any]:
    """Pearl's Level 3: Evaluate counterfactual attribution for an outcome."""
    factual = payload.get("factual_evidence", {})
    hypothetical = payload.get("hypothetical_action", {})
    target = payload.get("adverse_outcome_target", "")
    return causal_reasoner.explain_counterfactual_attribution(
        actual_evidence=factual,
        hypothetical_alternative=hypothetical,
        observed_bad_outcome=target,
    )


app.include_router(ws_router)



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
