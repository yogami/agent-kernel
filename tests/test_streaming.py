"""Tests for Prometheus Metrics Exporter and WebSocket Streaming."""

import json
import pytest
from fastapi.testclient import TestClient
from api.app import app
from ops.prometheus_exporter import PrometheusMetricsCollector


def test_prometheus_metrics_collector():
    """Verify metrics collector records latency histograms, tokens, and generates Prometheus text."""
    collector = PrometheusMetricsCollector()

    collector.record_turn(
        model_id="anthropic-claude-3-5",
        latency_ms=180.0,
        ttft_ms=75.0,
        tokens_in=150,
        tokens_out=45,
        cost_usd=0.0012,
        status="success",
    )

    collector.record_turn(
        model_id="anthropic-claude-3-5",
        latency_ms=420.0,
        ttft_ms=190.0,
        tokens_in=200,
        tokens_out=80,
        cost_usd=0.0020,
        status="error",
        error_class="ToolExecutionError",
    )

    metrics_text = collector.generate_prometheus_format()

    assert "agent_requests_total" in metrics_text
    assert "agent_tokens_input_total" in metrics_text
    assert "agent_tokens_output_total" in metrics_text
    assert "agent_turn_latency_ms_bucket" in metrics_text
    assert "agent_ttft_ms_bucket" in metrics_text
    assert 'agent_errors_total{class="ToolExecutionError",model="anthropic-claude-3-5"} 1' in metrics_text


def test_metrics_endpoint():
    """Verify /metrics HTTP endpoint responds with 200 and Prometheus payload."""
    client = TestClient(app)
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "agent_requests_total" in response.text


def test_websocket_chat_streaming():
    """Verify /ws/chat streaming protocol emits tokens, FSM states, and turn metrics."""
    client = TestClient(app)
    with client.websocket_connect("/ws/chat") as websocket:
        websocket.send_text(json.dumps({
            "user_input": "Hello streaming agent",
            "session_id": "test_ws_sess",
        }))

        events = []
        while True:
            msg_text = websocket.receive_text()
            data = json.loads(msg_text)
            events.append(data)
            if data.get("event") == "turn_completed":
                break

        event_types = [e["event"] for e in events]
        assert "fsm_state" in event_types
        assert "token" in event_types
        assert "turn_completed" in event_types

        completion = next(e for e in events if e["event"] == "turn_completed")
        assert "ttft_ms" in completion
        assert "total_latency_ms" in completion
