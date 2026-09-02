"""Tests for the Finite State Machine, budget limits, and loop detection."""

import pytest
from core.context_ram import ContextRAM
from core.execution_loop import ExecutionEngine
from core.guardrails import OutputGuardrails
from domain.models import ConfigPack, ToolResult
from domain.ports import ToolPort
from domain.state_machine import BudgetExceededError, LoopDetectedError
from infrastructure.llm_adapter import MockLLMAdapter
from infrastructure.sqlite_episode_store import SQLiteEpisodeStore
from infrastructure.sqlite_fact_store import SQLiteFactStore
from tools.registry import ToolRegistry


class DummyEchoTool(ToolPort):
    name = "dummy_echo"
    description = "Echoes back input argument."

    @property
    def schema(self):
        return {"type": "object", "properties": {"msg": {"type": "string"}}}

    def execute(self, arguments):
        return ToolResult(tool_id="d1", tool_name=self.name, output={"echo": arguments.get("msg")})


def test_fsm_completes_direct_turn():
    """Verify standard single-turn completion with FSM transition."""
    episode_store = SQLiteEpisodeStore(":memory:")
    fact_store = SQLiteFactStore(":memory:")
    tool_registry = ToolRegistry()
    context_ram = ContextRAM(fact_store)
    llm = MockLLMAdapter()

    llm.set_scripted_response("hello", {
        "content": "Hello! How can I assist you with clinical data today?",
        "tool_calls": [],
        "tokens_in": 10,
        "tokens_out": 15,
        "cost_usd": 0.0001,
    })

    engine = ExecutionEngine(llm, episode_store, tool_registry, context_ram)
    config = ConfigPack(version="v1", system_prompt="You are a clinical agent.", tool_schemas=[], rules=[])

    turn = engine.run_turn(
        user_input="Hello",
        session_id="sess_fsm_1",
        active_config=config,
    )

    assert turn.model_output == "Hello! How can I assist you with clinical data today?"
    assert turn.tokens_in == 10
    assert turn.tokens_out == 15
    assert len(episode_store.get_recent_turns("sess_fsm_1")) == 1


def test_fsm_tool_execution_loop():
    """Verify model triggers a tool, receives feedback, and produces final answer."""
    episode_store = SQLiteEpisodeStore(":memory:")
    fact_store = SQLiteFactStore(":memory:")
    tool_registry = ToolRegistry()
    tool_registry.register(DummyEchoTool())
    context_ram = ContextRAM(fact_store)
    llm = MockLLMAdapter()

    step_counter = 0

    def custom_handler(messages, tools):
        nonlocal step_counter
        step_counter += 1
        if step_counter == 1:
            return {
                "content": "",
                "tool_calls": [{
                    "id": "tc_1",
                    "type": "function",
                    "function": {"name": "dummy_echo", "arguments": {"msg": "test_echo"}},
                }],
                "tokens_in": 20,
                "tokens_out": 10,
                "cost_usd": 0.0001,
            }
        else:
            return {
                "content": "Tool finished echoing message.",
                "tool_calls": [],
                "tokens_in": 30,
                "tokens_out": 10,
                "cost_usd": 0.0001,
            }

    llm.set_handler(custom_handler)
    engine = ExecutionEngine(llm, episode_store, tool_registry, context_ram)
    config = ConfigPack(version="v1", system_prompt="Agent prompt", tool_schemas=[], rules=[])

    turn = engine.run_turn(
        user_input="Run echo test",
        session_id="sess_fsm_2",
        active_config=config,
    )

    assert turn.model_output == "Tool finished echoing message."
    assert len(turn.tool_calls) == 1
    assert turn.tool_calls[0].tool_name == "dummy_echo"
    assert turn.tool_results[0].output == {"echo": "test_echo"}


def test_fsm_step_budget_limit():
    """Verify that infinite loop or excessive turns raise BudgetExceededError."""
    episode_store = SQLiteEpisodeStore(":memory:")
    fact_store = SQLiteFactStore(":memory:")
    tool_registry = ToolRegistry()
    tool_registry.register(DummyEchoTool())
    context_ram = ContextRAM(fact_store)
    llm = MockLLMAdapter()

    counter = 0

    def endless_tool_loop(messages, tools):
        nonlocal counter
        counter += 1
        return {
            "content": "",
            "tool_calls": [{
                "id": f"tc_{counter}",
                "type": "function",
                "function": {"name": "dummy_echo", "arguments": {"msg": f"unique_step_{counter}"}},
            }],
            "tokens_in": 10,
            "tokens_out": 10,
            "cost_usd": 0.0001,
        }

    llm.set_handler(endless_tool_loop)
    engine = ExecutionEngine(llm, episode_store, tool_registry, context_ram)
    config = ConfigPack(version="v1", system_prompt="Agent prompt", tool_schemas=[], rules=[])

    with pytest.raises(BudgetExceededError) as exc_info:
        engine.run_turn(
            user_input="Trigger loop",
            session_id="sess_fsm_3",
            active_config=config,
            max_steps=3,
        )

    assert "exceeded hard step budget" in str(exc_info.value)


def test_fsm_loop_detection():
    """Verify that calling the exact same tool signature twice triggers LoopDetectedError."""
    episode_store = SQLiteEpisodeStore(":memory:")
    fact_store = SQLiteFactStore(":memory:")
    tool_registry = ToolRegistry()
    tool_registry.register(DummyEchoTool())
    context_ram = ContextRAM(fact_store)
    llm = MockLLMAdapter()

    def repetitive_tool_call(messages, tools):
        return {
            "content": "",
            "tool_calls": [{
                "id": "tc_rep",
                "type": "function",
                "function": {"name": "dummy_echo", "arguments": {"msg": "identical_call"}},
            }],
            "tokens_in": 10,
            "tokens_out": 10,
            "cost_usd": 0.0001,
        }

    llm.set_handler(repetitive_tool_call)
    engine = ExecutionEngine(llm, episode_store, tool_registry, context_ram)
    config = ConfigPack(version="v1", system_prompt="Agent prompt", tool_schemas=[], rules=[])

    with pytest.raises(LoopDetectedError) as exc_info:
        engine.run_turn(
            user_input="Repeat tool",
            session_id="sess_fsm_4",
            active_config=config,
            max_steps=5,
        )

    assert "Repeated execution loop detected" in str(exc_info.value)
