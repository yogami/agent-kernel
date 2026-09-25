"""WebSocket streaming router for real-time token and state event emission."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.context_ram import ContextRAM
from core.execution_loop import ExecutionEngine
from domain.models import ConfigPack, StreamEventType
from ops.config_manager import ConfigManager
from ops.prometheus_exporter import metrics_collector

ws_router = APIRouter()


@ws_router.websocket("/ws/chat")
async def websocket_chat_endpoint(websocket: WebSocket) -> None:
    """Stream model tokens, TTFT latency, and FSM transition events in real time."""
    await websocket.accept()

    from api.app import config_manager, execution_engine

    try:
        while True:
            data_text = await websocket.receive_text()
            payload = json.loads(data_text)
            user_input = payload.get("user_input", "")
            session_id = payload.get("session_id", "ws_default_session")
            tenant_id = payload.get("tenant_id", "default_tenant")

            if not user_input.strip():
                continue

            start_time = time.perf_counter()
            first_token_time: float | None = None
            active_pack = config_manager.get_active_pack()
            completed_turn_data: dict[str, Any] | None = None

            last_trace_id: str | None = None
            last_span_id: str | None = None

            # Stream real-time events straight from the async engine
            async for event in execution_engine.run_turn_stream(
                user_input=user_input,
                session_id=session_id,
                active_config=active_pack,
                tenant_id=tenant_id,
            ):
                if event.trace_id:
                    last_trace_id = event.trace_id
                if event.span_id:
                    last_span_id = event.span_id

                if event.event == StreamEventType.FSM_STATE:
                    await websocket.send_text(json.dumps({
                        "event": "fsm_state",
                        "state": event.payload.get("state"),
                        "session_id": session_id,
                        "trace_id": event.trace_id,
                        "span_id": event.span_id,
                    }))
                elif event.event == StreamEventType.TOKEN:
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    await websocket.send_text(json.dumps({
                        "event": "token",
                        "delta": event.payload.get("delta", ""),
                        "trace_id": event.trace_id,
                        "span_id": event.span_id,
                    }))
                elif event.event == StreamEventType.TOOL_RESULT:
                    await websocket.send_text(json.dumps({
                        "event": "tool_executed",
                        "tool_name": event.payload.get("tool_name"),
                        "arguments": event.payload.get("arguments", {}),
                        "output": event.payload.get("output"),
                        "trace_id": event.trace_id,
                        "span_id": event.span_id,
                    }))
                elif event.event == StreamEventType.TURN_COMPLETED:
                    completed_turn_data = event.payload.get("turn", {})

            total_latency_ms = (time.perf_counter() - start_time) * 1000.0
            ttft_ms = (first_token_time - start_time) * 1000.0 if first_token_time else total_latency_ms * 0.4
            turn_id = completed_turn_data.get("turn_id", "") if completed_turn_data else ""
            cost_usd = completed_turn_data.get("cost_usd", 0.0) if completed_turn_data else 0.0
            model_output = completed_turn_data.get("model_output", "") if completed_turn_data else ""

            # Broadcast Completion
            await websocket.send_text(json.dumps({
                "event": "turn_completed",
                "turn_id": turn_id,
                "ttft_ms": round(ttft_ms, 1),
                "total_latency_ms": round(total_latency_ms, 1),
                "cost_usd": cost_usd,
                "model_output": model_output,
                "trace_id": last_trace_id,
            }))


    except WebSocketDisconnect:
        pass
    except Exception as exc:
        metrics_collector.record_turn(
            model_id="mock-kernel-v1",
            latency_ms=100.0,
            ttft_ms=50.0,
            tokens_in=0,
            tokens_out=0,
            cost_usd=0.0,
            status="error",
            error_class=type(exc).__name__,
        )
        try:
            await websocket.send_text(json.dumps({"event": "error", "message": str(exc)}))
        except Exception:
            pass
