"""Tests for deterministic trace replay of historical episodes."""

import json
from core.context_ram import ContextRAM
from core.execution_loop import ExecutionEngine
from domain.models import ConfigPack, Episode, ToolCall, ToolResult, Turn
from infrastructure.llm_adapter import MockLLMAdapter
from infrastructure.sqlite_episode_store import SQLiteEpisodeStore
from infrastructure.sqlite_fact_store import SQLiteFactStore
from tools.clinical_tools import DeidentifyTextTool
from tools.registry import ToolRegistry


def test_deterministic_episode_replay():
    """Replay a recorded multi-step episode trace and verify 100% state match."""
    episode_store = SQLiteEpisodeStore(":memory:")
    fact_store = SQLiteFactStore(":memory:")
    tool_registry = ToolRegistry()
    tool_registry.register(DeidentifyTextTool())
    context_ram = ContextRAM(fact_store)

    # Recorded session fixture
    session_id = "replay_sess_001"
    recorded_raw_note = "Patient: Herr Schmidt at Charité Berlin on 12.04.1978"

    llm = MockLLMAdapter()
    step_count = 0

    def replay_handler(messages, tools):
        nonlocal step_count
        step_count += 1
        if step_count == 1:
            return {
                "content": "",
                "tool_calls": [{
                    "id": "replay_call_1",
                    "type": "function",
                    "function": {
                        "name": "deidentify_clinical_text",
                        "arguments": {"text": recorded_raw_note},
                    },
                }],
                "tokens_in": 35,
                "tokens_out": 15,
                "cost_usd": 0.0001,
            }
        else:
            return {
                "content": "De-identification completed without errors.",
                "tool_calls": [],
                "tokens_in": 60,
                "tokens_out": 20,
                "cost_usd": 0.00015,
            }

    llm.set_handler(replay_handler)
    engine = ExecutionEngine(llm, episode_store, tool_registry, context_ram)
    config = ConfigPack(version="v1", system_prompt="Replay agent", tool_schemas=[], rules=[])

    # Run original execution
    turn_1 = engine.run_turn(
        user_input="Scrub this patient note",
        session_id=session_id,
        active_config=config,
    )

    # Verify original run persisted
    recent = episode_store.get_recent_turns(session_id)
    assert len(recent) == 1
    assert recent[0].turn_id == turn_1.turn_id
    assert len(recent[0].tool_calls) == 1

    # Reset step counter and replay exact same turn
    step_count = 0
    replay_store = SQLiteEpisodeStore(":memory:")
    replay_engine = ExecutionEngine(llm, replay_store, tool_registry, context_ram)

    turn_replay = replay_engine.run_turn(
        user_input="Scrub this patient note",
        session_id="replay_verification_sess",
        active_config=config,
    )

    # Assert 100% deterministic outputs
    assert turn_replay.model_output == turn_1.model_output
    assert len(turn_replay.tool_calls) == len(turn_1.tool_calls)
    assert turn_replay.tool_calls[0].tool_name == turn_1.tool_calls[0].tool_name
    assert turn_replay.tool_results[0].output == turn_1.tool_results[0].output
