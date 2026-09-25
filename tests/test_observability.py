"""Verification suite for Phase 5 Distributed Observability and Real-Time Telemetry."""

from __future__ import annotations

import json
import pytest
from fastapi.testclient import TestClient

from api.app import app, tracer
from domain.models import ConfigPack, TraceContext
from domain.state_machine import KernelState
from core.context_ram import ContextRAM
from core.execution_loop import ExecutionEngine
from core.guardrails import OutputGuardrails
from infrastructure.llm_adapter import MockLLMAdapter
from infrastructure.sqlite_episode_store import SQLiteEpisodeStore
from infrastructure.sqlite_fact_store import SQLiteFactStore
from infrastructure.telemetry_adapter import (
    InMemoryTracer,
    OpenTelemetryAdapter,
    generate_span_id,
    generate_trace_id,
)
from ops.prometheus_exporter import PrometheusMetricsCollector
from tools.registry import ToolRegistry
from tools.system_tools import CalculatorTool


def test_w3c_traceparent_format_and_parsing():
    """Verify W3C traceparent formatting, parsing, and boundary validations."""
    raw_header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    ctx = TraceContext.from_traceparent(raw_header)
    assert ctx is not None
    assert ctx.trace_id == "4bf92f3577b34da6a3ce929d0e0e4736"
    assert ctx.span_id == "00f067aa0ba902b7"
    assert ctx.trace_flags == "01"
    assert ctx.to_traceparent() == raw_header

    # Invalid cases
    assert TraceContext.from_traceparent("") is None
    assert TraceContext.from_traceparent("invalid-header") is None
    # All zeros is invalid in W3C spec
    assert TraceContext.from_traceparent("00-00000000000000000000000000000000-00f067aa0ba902b7-01") is None
    assert TraceContext.from_traceparent("00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01") is None
    # Invalid hex
    assert TraceContext.from_traceparent("00-4bf92f3577b34da6a3ce929d0e0e473z-00f067aa0ba902b7-01") is None


def test_in_memory_tracer_span_lifecycle_and_hierarchy():
    """Verify tracer starts spans, nests children under active context, and records events."""
    custom_tracer = InMemoryTracer()
    assert custom_tracer.current_span() is None

    with custom_tracer.start_span("root.operation", attributes={"env": "test"}) as root:
        assert custom_tracer.current_span() is root
        assert len(root.context.trace_id) == 32
        assert len(root.context.span_id) == 16
        assert root.parent_span_id is None

        with custom_tracer.start_span("child.task", attributes={"worker": 1}) as child:
            assert custom_tracer.current_span() is child
            assert child.context.trace_id == root.context.trace_id
            assert child.parent_span_id == root.context.span_id
            child.add_event("data_fetched", {"rows": 42})
            child.set_status("OK")

        # After child exits, root is restored
        assert custom_tracer.current_span() is root
        root.set_attribute("completed", True)

    assert custom_tracer.current_span() is None

    # Verify captured spans
    records = custom_tracer.get_spans(trace_id=root.context.trace_id)
    assert len(records) == 2
    names = [r.name for r in records]
    assert "root.operation" in names
    assert "child.task" in names

    child_rec = next(r for r in records if r.name == "child.task")
    assert child_rec.duration_ms is not None
    assert child_rec.duration_ms >= 0.0
    assert len(child_rec.events) == 1
    assert child_rec.events[0].name == "data_fetched"
    assert child_rec.events[0].attributes["rows"] == 42


def test_opentelemetry_adapter_delegation():
    """Verify OpenTelemetryAdapter falls back cleanly to InMemoryTracer when SDK is absent."""
    adapter = OpenTelemetryAdapter()
    with adapter.start_span("otel.turn", attributes={"tenant": "t1"}) as span:
        span.set_attribute("key", "val")
        carrier = adapter.inject_traceparent({})
        assert "traceparent" in carrier
        extracted = adapter.extract_traceparent(carrier)
        assert extracted is not None
        assert extracted.trace_id == span.context.trace_id

    spans = adapter.get_spans()
    assert any(s.name == "otel.turn" for s in spans)
    adapter.clear()
    assert len(adapter.get_spans()) == 0


def test_prometheus_metrics_collector_percentiles_and_counters():
    """Verify live calculation of P50, P90, P99 percentiles and operational counters."""
    collector = PrometheusMetricsCollector()

    # Populate synthetic latencies
    latencies = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    for lat in latencies:
        collector.record_turn(
            model_id="gpt-test-model",
            latency_ms=lat,
            ttft_ms=lat * 0.5,
            tokens_in=100,
            tokens_out=50,
            cost_usd=0.0005,
            status="success",
        )

    percentiles = collector.compute_percentiles("gpt-test-model")
    assert percentiles["p50"] == pytest.approx(50.0, abs=5.0)
    assert percentiles["p90"] == pytest.approx(90.0, abs=5.0)
    assert percentiles["p99"] == pytest.approx(100.0, abs=5.0)

    # Operational events
    collector.record_tool_execution("calculate", "success")
    collector.record_fsm_transition("MODEL_CALL", "EXECUTE_TOOL")
    collector.record_memory_admission("ADMITTED")
    collector.record_span("agent.turn", "OK")

    text = collector.generate_prometheus_format()
    assert 'agent_turn_latency_p50_ms{model="gpt-test-model"}' in text
    assert 'agent_turn_latency_p90_ms{model="gpt-test-model"}' in text
    assert 'agent_turn_latency_p99_ms{model="gpt-test-model"}' in text
    assert 'agent_tool_executions_total{tool="calculate",status="success"} 1' in text
    assert 'agent_fsm_transitions_total{from="MODEL_CALL",to="EXECUTE_TOOL"} 1' in text
    assert 'agent_memory_admissions_total{decision="ADMITTED"} 1' in text
    assert 'agent_spans_total{name="agent.turn",status="OK"} 1' in text


def test_execution_engine_distributed_tracing(tmp_path):
    """Verify execution engine creates root turn span and child spans for each lifecycle step."""
    db_path = str(tmp_path / "obs_test.sqlite")
    ep_store = SQLiteEpisodeStore(db_path)
    fact_store = SQLiteFactStore(db_path)
    context_ram = ContextRAM(fact_store)
    tool_reg = ToolRegistry()
    tool_reg.register(CalculatorTool())

    mock_llm = MockLLMAdapter()
    mock_llm.set_scripted_response(
        "calc",
        {
            "content": None,
            "tool_calls": [{
                "id": "c1",
                "type": "function",
                "function": {
                    "name": "calculate",
                    "arguments": json.dumps({"expression": "25 * 4"}),
                },
            }],
            "tokens_in": 12,
            "tokens_out": 8,
            "cost_usd": 0.0001,
        },
    )

    custom_tracer = InMemoryTracer()
    engine = ExecutionEngine(
        llm=mock_llm,
        episode_store=ep_store,
        tool_registry=tool_reg,
        context_ram=context_ram,
        guardrails=OutputGuardrails(),
        tracer=custom_tracer,
    )

    pack = ConfigPack(version="v1", system_prompt="You calculate.", sha256_hash="h1")
    turn = engine.run_turn(
        user_input="calc please",
        session_id="obs_sess_01",
        active_config=pack,
        tenant_id="tenant_alpha",
    )

    assert turn is not None
    spans = custom_tracer.get_spans()
    span_names = [s.name for s in spans]

    assert "agent.turn" in span_names
    assert "agent.compose_context" in span_names
    assert "agent.llm_call" in span_names
    assert "agent.execute_tool" in span_names
    assert "agent.validate_output" in span_names
    assert "agent.persist_episode" in span_names

    # Check root span properties
    root_span = next(s for s in spans if s.name == "agent.turn")
    assert root_span.status == "OK"
    assert root_span.attributes["tenant_id"] == "tenant_alpha"
    assert root_span.attributes["session_id"] == "obs_sess_01"

    # Verify tool span recorded tool name
    tool_span = next(s for s in spans if s.name == "agent.execute_tool")
    assert tool_span.attributes["tool_name"] == "calculate"


def test_http_middleware_traceparent_propagation():
    """Verify HTTP requests parse incoming traceparent and inject traceparent in responses."""
    client = TestClient(app, headers={"X-API-Key": "dev-secret-key"})

    # 1. Health check generates a new traceparent header
    resp = client.get("/api/health")
    assert resp.status_code == 200
    traceparent_header = resp.headers.get("traceparent")
    assert traceparent_header is not None
    parsed = TraceContext.from_traceparent(traceparent_header)
    assert parsed is not None

    # 2. Chat request with incoming traceparent propagates trace ID
    in_trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    in_span_id = "00f067aa0ba902b7"
    incoming_header = f"00-{in_trace_id}-{in_span_id}-01"

    chat_resp = client.post(
        "/api/chat",
        headers={"traceparent": incoming_header},
        json={"user_input": "ping", "session_id": "test_obs_http"},
    )
    assert chat_resp.status_code == 200
    out_header = chat_resp.headers.get("traceparent")
    assert out_header is not None
    out_parsed = TraceContext.from_traceparent(out_header)
    assert out_parsed is not None
    assert out_parsed.trace_id == in_trace_id

    # 3. Query /api/traces endpoint
    traces_resp = client.get(f"/api/traces?trace_id={in_trace_id}")
    assert traces_resp.status_code == 200
    trace_list = traces_resp.json()
    assert len(trace_list) > 0
    assert all(t["context"]["trace_id"] == in_trace_id for t in trace_list)


def test_websocket_streaming_trace_correlation():
    """Verify WebSocket streaming frames carry correlated trace_id and span_id."""
    client = TestClient(app, headers={"X-API-Key": "dev-secret-key"})
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text(json.dumps({
            "user_input": "Hello observability agent",
            "session_id": "test_ws_obs_sess",
            "tenant_id": "obs_tenant",
        }))

        events = []
        while True:
            msg_text = websocket.receive_text()
            data = json.loads(msg_text)
            events.append(data)
            if data.get("event") == "turn_completed":
                break

        # Check trace correlation
        completion = next(e for e in events if e["event"] == "turn_completed")
        assert "trace_id" in completion
        assert completion["trace_id"] is not None
        assert len(completion["trace_id"]) == 32

        # State events must carry matching trace_id
        fsm_events = [e for e in events if e["event"] == "fsm_state"]
        assert len(fsm_events) > 0
        for fsm in fsm_events:
            assert fsm["trace_id"] == completion["trace_id"]
