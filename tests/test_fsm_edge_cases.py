import pytest
import json
from core.execution_loop import ExecutionEngine
from domain.models import ConfigPack
from infrastructure.llm_adapter import MockLLMAdapter
from infrastructure.sqlite_episode_store import SQLiteEpisodeStore
from core.context_ram import ContextRAM
from infrastructure.sqlite_fact_store import SQLiteFactStore
from tools.registry import ToolRegistry
from domain.models import ToolResult

class DummyOutputGuardrails:
    def validate_text_output(self, text: str) -> str:
        return text

class FakeTool:
    name = "fake"
    description = "fake"
    schema = {}
    def get_schema(self): return {"name": "fake", "description": "fake", "parameters": {}}
    def execute(self, kwargs):
        if kwargs.get("fail"):
            return ToolResult(tool_id="f", tool_name="fake", output={}, is_error=True, error_message="failed")
        return ToolResult(tool_id="t", tool_name="fake", output=kwargs, is_error=False, error_message=None)

@pytest.fixture
def engine():
    llm = MockLLMAdapter()
    store = SQLiteEpisodeStore(":memory:")
    tr = ToolRegistry()
    tr.register(FakeTool())
    e = ExecutionEngine(llm, store, tr, ContextRAM(SQLiteFactStore(":memory:")))
    e.guardrails = DummyOutputGuardrails()
    return e, llm

def test_sync_dollar_budget(engine):
    e, llm = engine
    llm.set_scripted_response("hi", {"content": "hi", "tokens_in": 10, "tokens_out": 10, "cost_usd": 1.0})
    with pytest.raises(Exception) as exc:
        e.run_turn("hi", "sess1", ConfigPack(version="v1", system_prompt=""), cost_budget_usd=0.0001)
    assert "dollar budget" in str(exc.value)

@pytest.mark.asyncio
async def test_async_dollar_budget(engine):
    e, llm = engine
    llm.set_scripted_response("hi", {"content": "hi", "tokens_in": 10, "tokens_out": 10, "cost_usd": 1.0})
    with pytest.raises(Exception) as exc:
        async for _ in e.run_turn_stream("hi", "sess1", ConfigPack(version="v1", system_prompt=""), cost_budget_usd=0.0001):
            pass
    assert "dollar budget" in str(exc.value)

def test_json_decode_fallback(engine):
    e, llm = engine
    llm.set_scripted_response("tool", {
        "content": "",
        "tool_calls": [{"id": "1", "function": {"name": "fake", "arguments": "invalid json {["}}]
    })
    e.run_turn("tool", "sess2", ConfigPack(version="v1", system_prompt=""))

@pytest.mark.asyncio
async def test_stream_json_decode_fallback(engine):
    e, llm = engine
    llm.set_scripted_response("tool", {
        "content": "",
        "tool_calls": [{"id": "1", "function": {"name": "fake", "arguments": "invalid json {["}}]
    })
    async for _ in e.run_turn_stream("tool", "sess2", ConfigPack(version="v1", system_prompt="")):
        pass

def test_tool_error_span(engine):
    e, llm = engine
    llm.set_scripted_response("fail_tool", {
        "content": "",
        "tool_calls": [{"id": "1", "function": {"name": "fake", "arguments": json.dumps({"fail": True})}}]
    })
    e.run_turn("fail_tool", "sess3", ConfigPack(version="v1", system_prompt=""))

@pytest.mark.asyncio
async def test_stream_tool_error_span(engine):
    e, llm = engine
    llm.set_scripted_response("fail_tool", {
        "content": "",
        "tool_calls": [{"id": "1", "function": {"name": "fake", "arguments": json.dumps({"fail": True})}}]
    })
    async for _ in e.run_turn_stream("fail_tool", "sess3", ConfigPack(version="v1", system_prompt="")):
        pass

@pytest.mark.asyncio
async def test_stream_loop_detected(engine):
    e, llm = engine
    def loop_handler(messages, tools):
        return {"content": "", "tool_calls": [{"id": "1", "type": "function", "function": {"name": "fake", "arguments": "{}"}}]}
    llm.set_handler(loop_handler)
    
    with pytest.raises(Exception) as exc:
        async for _ in e.run_turn_stream("loop", "sess4", ConfigPack(version="v1", system_prompt=""), max_steps=10):
            pass
    assert "Repeated execution loop detected" in str(exc.value)

@pytest.mark.asyncio
async def test_async_turn_missing_completion(engine):
    e, llm = engine
    llm.set_scripted_response("hi", {"content": "hi"})
    
    async def bad_stream(*args, **kwargs):
        from domain.models import StreamEvent, StreamEventType
        yield StreamEvent(event=StreamEventType.FSM_STATE, session_id="", trace_id="", span_id="", payload={})
        
    e.run_turn_stream = bad_stream
    with pytest.raises(Exception) as exc:
        await e.run_turn_async("hi", "sess5", ConfigPack(version="v1", system_prompt=""))
    assert "did not produce a TurnCompleted event" in str(exc.value)
