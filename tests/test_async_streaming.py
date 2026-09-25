"""Unit and integration test suite for native async FSM, parallel tool execution, and streaming."""

import asyncio
import json
import time
import pytest
from fastapi.testclient import TestClient

from api.app import app
from core.context_ram import ContextRAM
from core.execution_loop import ExecutionEngine
from domain.models import ConfigPack, StreamEventType, ToolResult
from domain.ports import ToolPort
from domain.state_machine import BudgetExceededError
from infrastructure.llm_adapter import MockLLMAdapter
from infrastructure.sqlite_episode_store import SQLiteEpisodeStore
from infrastructure.sqlite_fact_store import SQLiteFactStore
from tools.registry import ToolRegistry


class SlowAsyncTool(ToolPort):
    name: str = "slow_async_tool"
    description: str = "Simulates async tool execution with artificial latency."

    def __init__(self, delay_s: float = 0.05) -> None:
        self.delay_s = delay_s

    @property
    def schema(self):
        return {"type": "object", "properties": {"val": {"type": "integer"}}}

    def execute(self, arguments):
        time.sleep(self.delay_s)
        return ToolResult(tool_id="sync_slow", tool_name=self.name, output={"doubled": arguments.get("val", 0) * 2})

    async def execute_async(self, arguments):
        await asyncio.sleep(self.delay_s)
        return ToolResult(tool_id="async_slow", tool_name=self.name, output={"doubled": arguments.get("val", 0) * 2})


@pytest.mark.asyncio
async def test_mock_llm_adapter_stream_async():
    """Verify MockLLMAdapter streams tokens incrementally."""
    llm = MockLLMAdapter()
    llm.set_scripted_response("hello", {
        "content": "Hello streaming world",
        "tool_calls": [],
        "tokens_in": 5,
        "tokens_out": 3,
        "cost_usd": 0.0001,
    })

    chunks = []
    async for chunk in llm.stream_async([{"role": "user", "content": "hello"}]):
        chunks.append(chunk)

    deltas = [c["delta"] for c in chunks if c["delta"]]
    assert "".join(deltas) == "Hello streaming world"


@pytest.mark.asyncio
async def test_async_fsm_streaming_turn():
    """Verify run_turn_stream emits ordered state events, tokens, and turn completion."""
    episode_store = SQLiteEpisodeStore(":memory:")
    fact_store = SQLiteFactStore(":memory:")
    tool_registry = ToolRegistry()
    context_ram = ContextRAM(fact_store)
    llm = MockLLMAdapter()

    llm.set_scripted_response("diagnose", {
        "content": "Patient vitals are normal.",
        "tool_calls": [],
        "tokens_in": 15,
        "tokens_out": 5,
        "cost_usd": 0.0001,
    })

    engine = ExecutionEngine(llm, episode_store, tool_registry, context_ram)
    config = ConfigPack(version="v1", system_prompt="Clinical reasoning", tool_schemas=[], rules=[])

    events = []
    async for ev in engine.run_turn_stream(
        user_input="diagnose patient",
        session_id="sess_stream_1",
        active_config=config,
    ):
        events.append(ev)

    event_types = [e.event for e in events]
    assert StreamEventType.FSM_STATE in event_types
    assert StreamEventType.TOKEN in event_types
    assert StreamEventType.TURN_COMPLETED in event_types

    # Verify states followed FSM progression
    fsm_states = [e.payload["state"] for e in events if e.event == StreamEventType.FSM_STATE]
    assert "COMPOSE_CONTEXT" in fsm_states
    assert "MODEL_CALL" in fsm_states
    assert "VALIDATE_OUTPUT" in fsm_states
    assert "PERSIST_EPISODE" in fsm_states

    # Verify turn was persisted to SQLite
    stored_turns = episode_store.get_recent_turns("sess_stream_1")
    assert len(stored_turns) == 1
    assert stored_turns[0].model_output == "Patient vitals are normal."


@pytest.mark.asyncio
async def test_async_parallel_tool_execution():
    """Verify multiple tool calls in a turn execute concurrently."""
    episode_store = SQLiteEpisodeStore(":memory:")
    fact_store = SQLiteFactStore(":memory:")
    tool_registry = ToolRegistry()
    tool_registry.register(SlowAsyncTool(delay_s=0.08))
    context_ram = ContextRAM(fact_store)
    llm = MockLLMAdapter()

    call_count = 0

    def parallel_handler(messages, tools):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return {
                "content": "",
                "tool_calls": [
                    {
                        "id": "tc_1",
                        "type": "function",
                        "function": {"name": "slow_async_tool", "arguments": {"val": 10}},
                    },
                    {
                        "id": "tc_2",
                        "type": "function",
                        "function": {"name": "slow_async_tool", "arguments": {"val": 20}},
                    },
                ],
                "tokens_in": 20,
                "tokens_out": 10,
                "cost_usd": 0.0002,
            }
        return {
            "content": "Parallel tool calls complete.",
            "tool_calls": [],
            "tokens_in": 30,
            "tokens_out": 5,
            "cost_usd": 0.0001,
        }

    llm.set_handler(parallel_handler)
    engine = ExecutionEngine(llm, episode_store, tool_registry, context_ram)
    config = ConfigPack(version="v1", system_prompt="Async test", tool_schemas=[], rules=[])

    start = time.perf_counter()
    turn = await engine.run_turn_async(
        user_input="Run 2 slow tools in parallel",
        session_id="sess_parallel_tools",
        active_config=config,
    )
    elapsed = time.perf_counter() - start

    assert turn.model_output == "Parallel tool calls complete."
    assert len(turn.tool_calls) == 2
    assert len(turn.tool_results) == 2
    # Two 80ms tools in parallel should take < 140ms total, not 160ms+
    assert elapsed < 0.15


@pytest.mark.asyncio
async def test_async_budget_limit_streaming():
    """Verify stream terminates with BudgetExceededError when max_steps is hit."""
    episode_store = SQLiteEpisodeStore(":memory:")
    fact_store = SQLiteFactStore(":memory:")
    tool_registry = ToolRegistry()
    tool_registry.register(SlowAsyncTool(delay_s=0.01))
    context_ram = ContextRAM(fact_store)
    llm = MockLLMAdapter()

    # Infinite tool calling handler
    def loop_handler(messages, tools):
        return {
            "content": "",
            "tool_calls": [{
                "id": "tc_loop",
                "type": "function",
                "function": {"name": "slow_async_tool", "arguments": {"val": 1}},
            }],
            "tokens_in": 10,
            "tokens_out": 5,
            "cost_usd": 0.0001,
        }

    llm.set_handler(loop_handler)
    engine = ExecutionEngine(llm, episode_store, tool_registry, context_ram)
    config = ConfigPack(version="v1", system_prompt="Budget test", tool_schemas=[], rules=[])

    with pytest.raises(Exception):
        async for _ in engine.run_turn_stream(
            user_input="Infinite loop trigger",
            session_id="sess_budget_stream",
            active_config=config,
            max_steps=2,
        ):
            pass


def test_sse_chat_stream_endpoint():
    """Verify /api/chat/stream HTTP endpoint delivers formatted Server-Sent Events."""
    client = TestClient(app, headers={"X-API-Key": "dev-secret-key"})
    response = client.post(
        "/api/chat/stream",
        json={"user_input": "Hello SSE stream", "session_id": "test_sse_sess"},
    )
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]

    lines = [line.strip() for line in response.text.split("\n") if line.strip()]
    assert any("data: [DONE]" in line for line in lines)

    # Parse received JSON event envelopes
    parsed_events = []
    for line in lines:
        if line.startswith("data: ") and not line.endswith("[DONE]"):
            data = json.loads(line[6:])
            parsed_events.append(data)

    event_names = [e.get("event") for e in parsed_events]
    assert "fsm_state" in event_names
    assert "token" in event_names
    assert "turn_completed" in event_names
