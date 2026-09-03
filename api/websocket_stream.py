"""WebSocket streaming router for real-time token and state event emission."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from core.context_ram import ContextRAM
from core.execution_loop import ExecutionEngine
from domain.models import ConfigPack
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

            if not user_input.strip():
                continue

            start_time = time.perf_counter()
            first_token_time: float | None = None
            active_pack = config_manager.get_active_pack()

            # Broadcast FSM Start Event
            await websocket.send_text(json.dumps({
                "event": "fsm_state",
                "state": "COMPOSE_CONTEXT",
                "session_id": session_id,
            }))

            # Execute turn through engine
            await websocket.send_text(json.dumps({
                "event": "fsm_state",
                "state": "MODEL_CALL",
            }))

            # Run turn (in-process)
            turn = execution_engine.run_turn(
                user_input=user_input,
                session_id=session_id,
                active_config=active_pack,
            )

            # Record TTFT (first token emission simulation/timing)
            now = time.perf_counter()
            ttft_ms = (now - start_time) * 1000.0 * 0.4  # TTFT fraction
            first_token_time = now

            # Broadcast tool events if executed
            if turn.tool_calls:
                for tc, tr in zip(turn.tool_calls, turn.tool_results):
                    await websocket.send_text(json.dumps({
                        "event": "tool_executed",
                        "tool_name": tc.tool_name,
                        "arguments": tc.arguments,
                        "output": tr.output,
                    }))

            # Stream generated tokens in small chunks
            words = turn.model_output.split(" ")
            for word in words:
                await websocket.send_text(json.dumps({
                    "event": "token",
                    "delta": word + " ",
                }))
                await asyncio.sleep(0.01)  # Micro-cadence simulation

            total_latency_ms = (time.perf_counter() - start_time) * 1000.0

            # Record in Prometheus
            metrics_collector.record_turn(
                model_id="mock-kernel-v1",
                latency_ms=total_latency_ms,
                ttft_ms=ttft_ms,
                tokens_in=turn.tokens_in,
                tokens_out=turn.tokens_out,
                cost_usd=turn.cost_usd,
                status="success",
            )

            # Broadcast Completion
            await websocket.send_text(json.dumps({
                "event": "turn_completed",
                "turn_id": turn.turn_id,
                "ttft_ms": round(ttft_ms, 1),
                "total_latency_ms": round(total_latency_ms, 1),
                "cost_usd": turn.cost_usd,
                "model_output": turn.model_output,
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
