"""Deterministic Finite State Machine Execution Engine."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import time
from dataclasses import dataclass, field
from domain.serialization import to_dict
from typing import Any, AsyncIterator

from domain.ports import ContextRAMPort, OutputGuardrailsPort

from domain.models import (
    ConfigPack,
    ExecutionContext,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolResult,
    Turn,
)
from domain.ports import EpisodeStorePort, LLMProviderPort, TracerPort, ToolRegistryPort, MetricsPort
from domain.state_machine import (
    BudgetExceededError,
    KernelState,
    LoopDetectedError,
    can_transition,
)


class DummyContext:
    trace_id = "0"
    span_id = "0"

class DummySpan:
    def __init__(self, ctx=None):
        self.context = ctx or DummyContext()
    def set_attribute(self, k, v): pass
    def add_event(self, n, a=None): pass
    def set_status(self, s, d=None): pass
    def record_exception(self, e): pass
    def end(self): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass

class DummyTracer:
    def start_span(self, n, parent_context=None, attributes=None):
        return DummySpan()
    def current_span(self): return None
    def extract_traceparent(self, c): return None
    def inject_traceparent(self, c, ctx=None): return c

class DummyMetrics:
    def record_turn(self, *args, **kwargs): pass
    def record_tool_execution(self, *args, **kwargs): pass
    def record_fsm_transition(self, *args, **kwargs): pass
    def record_span(self, *args, **kwargs): pass


@dataclass
class TurnState:
    current_state: KernelState = KernelState.COMPOSE_CONTEXT
    executed_tool_calls: list[ToolCall] = field(default_factory=list)
    executed_tool_results: list[ToolResult] = field(default_factory=list)
    accumulated_tokens_in: int = 0
    accumulated_tokens_out: int = 0
    final_model_output: str = ""
    messages: list[dict[str, Any]] = field(default_factory=list)
    pending_calls: list[ToolCall] = field(default_factory=list)
    raw_tool_calls: list[dict[str, Any]] = field(default_factory=list)
    first_token_time: float | None = None
    recent_turns: list[Turn] = field(default_factory=list)


class ExecutionEngine:
    """Step-budgeted Finite State Machine executing agent interactions with hard limits."""

    def __init__(
        self,
        llm: LLMProviderPort,
        episode_store: EpisodeStorePort,
        tool_registry: ToolRegistryPort,
        context_ram: ContextRAMPort,
        guardrails: OutputGuardrailsPort | None = None,
        tracer: TracerPort | None = None,
        metrics_collector: MetricsPort | None = None,
    ) -> None:
        self.llm = llm
        self.episode_store = episode_store
        self.tool_registry = tool_registry
        self.context_ram = context_ram
        self.guardrails = guardrails or DummyGuardrails()
        self.tracer = tracer or DummyTracer()
        self.metrics_collector = metrics_collector or DummyMetrics()
        
    def _check_budgets(self, ctx: ExecutionContext) -> None:
        if ctx.step_index >= ctx.max_steps:
            raise BudgetExceededError(
                f"Execution exceeded hard step budget of {ctx.max_steps} steps.",
                {"step_index": ctx.step_index, "max_steps": ctx.max_steps},
            )

        if ctx.cost_usd >= ctx.cost_budget_usd:
            raise BudgetExceededError(
                f"Execution exceeded dollar budget of ${ctx.cost_budget_usd:.4f}.",
                {"cost_usd": ctx.cost_usd, "budget": ctx.cost_budget_usd},
            )
            
    def _handle_compose_context(
        self, state: TurnState, user_input: str, active_config: ConfigPack, session_id: str, tenant_id: str, turn_span_context: Any
    ) -> None:
        with self.tracer.start_span("agent.compose_context", parent_context=turn_span_context):
            state.messages = self.context_ram.compose_context(
                user_input=user_input,
                recent_turns=state.recent_turns,
                active_config=active_config,
                session_id=session_id,
                tenant_id=tenant_id,
            )
        self.metrics_collector.record_fsm_transition(
            KernelState.COMPOSE_CONTEXT.value, KernelState.MODEL_CALL.value
        )
        state.current_state = KernelState.MODEL_CALL

    def _parse_tool_calls(self, raw_tool_calls: list[Any], ctx: ExecutionContext) -> list[ToolCall]:
        pending_calls: list[ToolCall] = []
        for raw_tc in raw_tool_calls:
            if isinstance(raw_tc, dict):
                fn_info = raw_tc.get("function", {})
                fn_name = fn_info.get("name", "")
                args_val = fn_info.get("arguments", {})
                if isinstance(args_val, str):
                    try:
                        args_dict = json.loads(args_val)
                    except json.JSONDecodeError:
                        args_dict = {"raw": args_val}
                else:
                    args_dict = args_val or {}
                pending_calls.append(
                    ToolCall(
                        tool_id=raw_tc.get("id", str(ctx.step_index)),
                        tool_name=fn_name,
                        arguments=args_dict,
                    )
                )
        return pending_calls

    def _check_loop_detection(self, pending_calls: list[ToolCall], ctx: ExecutionContext) -> None:
        for tc in pending_calls:
            sig = f"{tc.tool_name}:{json.dumps(tc.arguments, sort_keys=True)}"
            if ctx.executed_tool_signatures.count(sig) >= 2:
                raise LoopDetectedError(
                    f"Repeated execution loop detected on tool signature '{tc.tool_name}'.",
                    {"signature": sig, "repeat_count": ctx.executed_tool_signatures.count(sig)},
                )
            ctx.executed_tool_signatures.append(sig)

    def _handle_model_call(self, state: TurnState, ctx: ExecutionContext, turn_span_context: Any) -> None:
        ctx.step_index += 1
        tool_schemas = self.tool_registry.get_schemas()
        with self.tracer.start_span(
            "agent.llm_call",
            parent_context=turn_span_context,
            attributes={"step_index": ctx.step_index},
        ) as call_span:
            llm_resp = self.llm.generate(
                messages=state.messages,
                tools=tool_schemas if tool_schemas else None,
            )
            tokens_in = llm_resp.get("tokens_in", 0)
            tokens_out = llm_resp.get("tokens_out", 0)
            step_cost = llm_resp.get("cost_usd", 0.0)
            call_span.set_attribute("tokens_in", tokens_in)
            call_span.set_attribute("tokens_out", tokens_out)
            call_span.set_attribute("cost_usd", step_cost)

        state.accumulated_tokens_in += tokens_in
        state.accumulated_tokens_out += tokens_out
        ctx.cost_usd += step_cost
        ctx.accumulated_tokens += (tokens_in + tokens_out)

        raw_tool_calls = llm_resp.get("tool_calls", [])
        content = llm_resp.get("content", "")

        if raw_tool_calls:
            state.raw_tool_calls = raw_tool_calls
            state.pending_calls = self._parse_tool_calls(raw_tool_calls, ctx)
            self._check_loop_detection(state.pending_calls, ctx)
            self.metrics_collector.record_fsm_transition(
                KernelState.MODEL_CALL.value, KernelState.VALIDATE_TOOL_CALL.value
            )
            state.current_state = KernelState.VALIDATE_TOOL_CALL
        else:
            state.final_model_output = content
            self.metrics_collector.record_fsm_transition(
                KernelState.MODEL_CALL.value, KernelState.VALIDATE_OUTPUT.value
            )
            state.current_state = KernelState.VALIDATE_OUTPUT

    def _handle_validate_tool_call(self, state: TurnState) -> None:
        self.metrics_collector.record_fsm_transition(
            KernelState.VALIDATE_TOOL_CALL.value, KernelState.EXECUTE_TOOL.value
        )
        state.current_state = KernelState.EXECUTE_TOOL
        
    def _handle_execute_tool(self, state: TurnState, turn_span_context: Any) -> None:
        executed_results: list[ToolResult] = []
        for tc in state.pending_calls:
            state.executed_tool_calls.append(tc)
            with self.tracer.start_span(
                "agent.execute_tool",
                parent_context=turn_span_context,
                attributes={"tool_name": tc.tool_name},
            ) as tool_s:
                res = self.tool_registry.execute(tc.tool_name, tc.arguments)
                res.tool_id = tc.tool_id
                state.executed_tool_results.append(res)
                executed_results.append(res)
                status_str = "error" if res.is_error else "success"
                self.metrics_collector.record_tool_execution(tc.tool_name, status_str)
                if res.is_error:
                    tool_s.set_status("ERROR", res.error_message)
                else:
                    tool_s.set_status("OK")

        # Group all tool calls from this turn into a single assistant message
        state.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": state.raw_tool_calls if state.raw_tool_calls else [
                {
                    "id": tc.tool_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in state.pending_calls
            ],
        })
        for tc, res in zip(state.pending_calls, executed_results):
            state.messages.append({
                "role": "tool",
                "tool_call_id": tc.tool_id,
                "content": json.dumps(res.output) if not res.is_error else f"Error: {res.error_message}",
            })

        self.metrics_collector.record_fsm_transition(
            KernelState.EXECUTE_TOOL.value, KernelState.MODEL_CALL.value
        )
        state.current_state = KernelState.MODEL_CALL
        
    def _handle_validate_output(self, state: TurnState, turn_span_context: Any) -> None:
        with self.tracer.start_span("agent.validate_output", parent_context=turn_span_context):
            sanitized_output = self.guardrails.validate_text_output(state.final_model_output)
            state.final_model_output = sanitized_output
        self.metrics_collector.record_fsm_transition(
            KernelState.VALIDATE_OUTPUT.value, KernelState.PERSIST_EPISODE.value
        )
        state.current_state = KernelState.PERSIST_EPISODE

    def _handle_persist_episode(
        self, state: TurnState, ctx: ExecutionContext, start_time: float,
        user_input: str, session_id: str, active_config: ConfigPack,
        turn_index: int, tenant_id: str, turn_span_context: Any
    ) -> Turn:
        elapsed_ms = (time.time() - start_time) * 1000.0
        turn = Turn(
            session_id=session_id,
            tenant_id=tenant_id,
            turn_index=turn_index,
            user_input=user_input,
            model_output=state.final_model_output,
            tool_calls=state.executed_tool_calls,
            tool_results=state.executed_tool_results,
            tokens_in=state.accumulated_tokens_in,
            tokens_out=state.accumulated_tokens_out,
            cost_usd=ctx.cost_usd,
            latency_ms=elapsed_ms,
            created_at=datetime.now(timezone.utc),
        )
        with self.tracer.start_span("agent.persist_episode", parent_context=turn_span_context):
            self.episode_store.append_turn(turn)

        state.current_state = KernelState.COMPLETED
        self.metrics_collector.record_fsm_transition(
            KernelState.PERSIST_EPISODE.value, KernelState.COMPLETED.value
        )
        self.metrics_collector.record_span("agent.turn", "OK")
        
        ttft_ms = elapsed_ms * 0.4
        if state.first_token_time:
            ttft_ms = (state.first_token_time - start_time) * 1000.0

        self.metrics_collector.record_turn(
            model_id=active_config.version or "kernel-v1",
            latency_ms=elapsed_ms,
            ttft_ms=ttft_ms,
            tokens_in=state.accumulated_tokens_in,
            tokens_out=state.accumulated_tokens_out,
            cost_usd=ctx.cost_usd,
            status="success",
        )
        return turn


    def run_turn(
        self,
        user_input: str,
        session_id: str,
        active_config: ConfigPack,
        turn_index: int = 0,
        max_steps: int = 5,
        cost_budget_usd: float = 0.50,
        tenant_id: str = "default_tenant",
    ) -> Turn:
        """Execute a single multi-step interaction turn through the state machine."""
        start_time = time.time()
        ctx = ExecutionContext(
            session_id=session_id,
            step_index=0,
            max_steps=max_steps,
            cost_budget_usd=cost_budget_usd,
        )

        turn_span = self.tracer.start_span(
            "agent.turn",
            attributes={
                "session_id": session_id,
                "tenant_id": tenant_id,
                "turn_index": turn_index,
            },
        )

        state = TurnState()
        state.recent_turns = self.episode_store.get_recent_turns(session_id, limit=5, tenant_id=tenant_id)

        try:
            with turn_span:
                while state.current_state not in {KernelState.COMPLETED, KernelState.FAILED}:
                    self._check_budgets(ctx)

                    if state.current_state == KernelState.COMPOSE_CONTEXT:
                        self._handle_compose_context(state, user_input, active_config, session_id, tenant_id, turn_span.context)
                    elif state.current_state == KernelState.MODEL_CALL:
                        self._handle_model_call(state, ctx, turn_span.context)
                    elif state.current_state == KernelState.VALIDATE_TOOL_CALL:
                        self._handle_validate_tool_call(state)
                    elif state.current_state == KernelState.EXECUTE_TOOL:
                        self._handle_execute_tool(state, turn_span.context)
                    elif state.current_state == KernelState.VALIDATE_OUTPUT:
                        self._handle_validate_output(state, turn_span.context)
                    elif state.current_state == KernelState.PERSIST_EPISODE:
                        return self._handle_persist_episode(
                            state, ctx, start_time, user_input, session_id, active_config, turn_index, tenant_id, turn_span.context
                        )

            raise RuntimeError(f"State machine terminated in unexpected state '{state.current_state}'.")
        except Exception as exc:
            elapsed_ms = (time.time() - start_time) * 1000.0
            self.metrics_collector.record_span("agent.turn", "ERROR")
            self.metrics_collector.record_turn(
                model_id=active_config.version or "kernel-v1",
                latency_ms=elapsed_ms,
                ttft_ms=0.0,
                tokens_in=state.accumulated_tokens_in,
                tokens_out=state.accumulated_tokens_out,
                cost_usd=ctx.cost_usd,
                status="error",
                error_class=type(exc).__name__,
            )
            raise


    async def _handle_model_call_stream(
        self, state: TurnState, ctx: ExecutionContext, session_id: str, trace_id: str, span_id: str, turn_span_context: Any
    ) -> AsyncIterator[StreamEvent]:
        ctx.step_index += 1
        yield StreamEvent(
            event=StreamEventType.FSM_STATE,
            session_id=session_id,
            payload={"state": KernelState.MODEL_CALL.value, "step_index": ctx.step_index},
            trace_id=trace_id,
            span_id=span_id,
        )
        tool_schemas = self.tool_registry.get_schemas()

        streamed_content = ""
        raw_tool_calls: list[dict[str, Any]] = []

        with self.tracer.start_span(
            "agent.llm_call",
            parent_context=turn_span_context,
            attributes={"step_index": ctx.step_index},
        ) as call_span:
            async for chunk in self.llm.stream_async(
                messages=state.messages,
                tools=tool_schemas if tool_schemas else None,
            ):
                delta = chunk.get("delta", "")
                if delta:
                    if state.first_token_time is None:
                        state.first_token_time = time.time()
                    streamed_content += delta
                    yield StreamEvent(
                        event=StreamEventType.TOKEN,
                        session_id=session_id,
                        payload={"delta": delta},
                        trace_id=trace_id,
                        span_id=span_id,
                    )

                tcs = chunk.get("tool_calls", [])
                if tcs:
                    raw_tool_calls.extend(tcs)

                state.accumulated_tokens_in += chunk.get("tokens_in", 0)
                state.accumulated_tokens_out += chunk.get("tokens_out", 0)
                chunk_cost = chunk.get("cost_usd", 0.0)
                ctx.cost_usd += chunk_cost
                ctx.accumulated_tokens += (chunk.get("tokens_in", 0) + chunk.get("tokens_out", 0))

            call_span.set_attribute("tokens_in", state.accumulated_tokens_in)
            call_span.set_attribute("tokens_out", state.accumulated_tokens_out)
            call_span.set_attribute("cost_usd", ctx.cost_usd)

        if raw_tool_calls:
            state.raw_tool_calls = raw_tool_calls
            state.pending_calls = self._parse_tool_calls(raw_tool_calls, ctx)
            try:
                self._check_loop_detection(state.pending_calls, ctx)
            except LoopDetectedError as err:
                yield StreamEvent(
                    event=StreamEventType.ERROR,
                    session_id=session_id,
                    payload={"error": str(err), "type": "LoopDetectedError"},
                    trace_id=trace_id,
                    span_id=span_id,
                )
                raise err
            self.metrics_collector.record_fsm_transition(
                KernelState.MODEL_CALL.value, KernelState.VALIDATE_TOOL_CALL.value
            )
            state.current_state = KernelState.VALIDATE_TOOL_CALL
        else:
            state.final_model_output = streamed_content
            self.metrics_collector.record_fsm_transition(
                KernelState.MODEL_CALL.value, KernelState.VALIDATE_OUTPUT.value
            )
            state.current_state = KernelState.VALIDATE_OUTPUT


    async def _handle_execute_tool_stream(
        self, state: TurnState, session_id: str, trace_id: str, span_id: str, turn_span_context: Any
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(
            event=StreamEventType.FSM_STATE,
            session_id=session_id,
            payload={"state": KernelState.EXECUTE_TOOL.value},
            trace_id=trace_id,
            span_id=span_id,
        )
        for tc in state.pending_calls:
            state.executed_tool_calls.append(tc)
            yield StreamEvent(
                event=StreamEventType.TOOL_CALL,
                session_id=session_id,
                payload={"tool_id": tc.tool_id, "tool_name": tc.tool_name, "arguments": tc.arguments},
                trace_id=trace_id,
                span_id=span_id,
            )

        tasks = [self.tool_registry.execute_async(tc.tool_name, tc.arguments) for tc in state.pending_calls]
        results = await asyncio.gather(*tasks)

        for tc, res in zip(state.pending_calls, results):
            res.tool_id = tc.tool_id
            state.executed_tool_results.append(res)
            status_str = "error" if res.is_error else "success"
            self.metrics_collector.record_tool_execution(tc.tool_name, status_str)

            with self.tracer.start_span(
                "agent.execute_tool",
                parent_context=turn_span_context,
                attributes={"tool_name": tc.tool_name, "tool_id": tc.tool_id},
            ) as ts:
                ts.set_attribute("is_error", res.is_error)
                if res.is_error:
                    ts.set_status("ERROR", res.error_message)
                else:
                    ts.set_status("OK")

            yield StreamEvent(
                event=StreamEventType.TOOL_RESULT,
                session_id=session_id,
                payload={
                    "tool_id": res.tool_id,
                    "tool_name": res.tool_name,
                    "output": res.output,
                    "is_error": res.is_error,
                    "error_message": res.error_message,
                },
                trace_id=trace_id,
                span_id=span_id,
            )

        # Group all tool calls from this turn into a single assistant message
        state.messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": state.raw_tool_calls if state.raw_tool_calls else [
                {
                    "id": tc.tool_id,
                    "type": "function",
                    "function": {
                        "name": tc.tool_name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in state.pending_calls
            ],
        })

        for tc, res in zip(state.pending_calls, results):
            state.messages.append({
                "role": "tool",
                "tool_call_id": tc.tool_id,
                "content": json.dumps(res.output) if not res.is_error else f"Error: {res.error_message}",
            })

        self.metrics_collector.record_fsm_transition(
            KernelState.EXECUTE_TOOL.value, KernelState.MODEL_CALL.value
        )
        state.current_state = KernelState.MODEL_CALL


    async def run_turn_stream(
        self,
        user_input: str,
        session_id: str,
        active_config: ConfigPack,
        turn_index: int = 0,
        max_steps: int = 5,
        cost_budget_usd: float = 0.50,
        tenant_id: str = "default_tenant",
    ) -> AsyncIterator[StreamEvent]:
        """Execute a single multi-step interaction turn asynchronously, streaming state transitions, tokens, and tool events."""
        start_time = time.time()
        ctx = ExecutionContext(
            session_id=session_id,
            step_index=0,
            max_steps=max_steps,
            cost_budget_usd=cost_budget_usd,
        )

        turn_span = self.tracer.start_span(
            "agent.turn",
            attributes={
                "session_id": session_id,
                "tenant_id": tenant_id,
                "turn_index": turn_index,
            },
        )
        trace_id = turn_span.context.trace_id
        span_id = turn_span.context.span_id

        state = TurnState()
        state.recent_turns = self.episode_store.get_recent_turns(session_id, limit=5, tenant_id=tenant_id)

        try:
            with turn_span:
                while state.current_state not in {KernelState.COMPLETED, KernelState.FAILED}:
                    try:
                        self._check_budgets(ctx)
                    except BudgetExceededError as err:
                        yield StreamEvent(
                            event=StreamEventType.ERROR,
                            session_id=session_id,
                            payload={"error": str(err), "type": "BudgetExceededError"},
                            trace_id=trace_id,
                            span_id=span_id,
                        )
                        raise err

                    if state.current_state == KernelState.COMPOSE_CONTEXT:
                        yield StreamEvent(
                            event=StreamEventType.FSM_STATE,
                            session_id=session_id,
                            payload={"state": KernelState.COMPOSE_CONTEXT.value},
                            trace_id=trace_id,
                            span_id=span_id,
                        )
                        self._handle_compose_context(state, user_input, active_config, session_id, tenant_id, turn_span.context)

                    elif state.current_state == KernelState.MODEL_CALL:
                        async for event in self._handle_model_call_stream(state, ctx, session_id, trace_id, span_id, turn_span.context):
                            yield event

                    elif state.current_state == KernelState.VALIDATE_TOOL_CALL:
                        yield StreamEvent(
                            event=StreamEventType.FSM_STATE,
                            session_id=session_id,
                            payload={"state": KernelState.VALIDATE_TOOL_CALL.value},
                            trace_id=trace_id,
                            span_id=span_id,
                        )
                        self._handle_validate_tool_call(state)

                    elif state.current_state == KernelState.EXECUTE_TOOL:
                        async for event in self._handle_execute_tool_stream(state, session_id, trace_id, span_id, turn_span.context):
                            yield event

                    elif state.current_state == KernelState.VALIDATE_OUTPUT:
                        yield StreamEvent(
                            event=StreamEventType.FSM_STATE,
                            session_id=session_id,
                            payload={"state": KernelState.VALIDATE_OUTPUT.value},
                            trace_id=trace_id,
                            span_id=span_id,
                        )
                        self._handle_validate_output(state, turn_span.context)

                    elif state.current_state == KernelState.PERSIST_EPISODE:
                        yield StreamEvent(
                            event=StreamEventType.FSM_STATE,
                            session_id=session_id,
                            payload={"state": KernelState.PERSIST_EPISODE.value},
                            trace_id=trace_id,
                            span_id=span_id,
                        )
                        turn = self._handle_persist_episode(
                            state, ctx, start_time, user_input, session_id, active_config, turn_index, tenant_id, turn_span.context
                        )
                        
                        elapsed_ms = (time.time() - start_time) * 1000.0
                        yield StreamEvent(
                            event=StreamEventType.TURN_COMPLETED,
                            session_id=session_id,
                            payload={"turn": to_dict(turn), "total_latency_ms": elapsed_ms},
                            trace_id=trace_id,
                            span_id=span_id,
                        )
                        return

            raise RuntimeError(f"State machine terminated in unexpected state '{state.current_state}'.")
        except Exception as exc:
            elapsed_ms = (time.time() - start_time) * 1000.0
            self.metrics_collector.record_span("agent.turn", "ERROR")
            self.metrics_collector.record_turn(
                model_id=active_config.version or "kernel-v1",
                latency_ms=elapsed_ms,
                ttft_ms=0.0,
                tokens_in=state.accumulated_tokens_in,
                tokens_out=state.accumulated_tokens_out,
                cost_usd=ctx.cost_usd,
                status="error",
                error_class=type(exc).__name__,
            )
            raise

    async def run_turn_async(
        self,
        user_input: str,
        session_id: str,
        active_config: ConfigPack,
        turn_index: int = 0,
        max_steps: int = 5,
        cost_budget_usd: float = 0.50,
        tenant_id: str = "default_tenant",
    ) -> Turn:
        """Run turn asynchronously and return the completed Turn entity."""
        last_turn: Turn | None = None
        async for event in self.run_turn_stream(
            user_input=user_input,
            session_id=session_id,
            active_config=active_config,
            turn_index=turn_index,
            max_steps=max_steps,
            cost_budget_usd=cost_budget_usd,
            tenant_id=tenant_id,
        ):
            if event.event == StreamEventType.TURN_COMPLETED:
                last_turn = Turn(**event.payload["turn"])

        if last_turn is None:
            raise RuntimeError("Async turn execution did not produce a TurnCompleted event.")
        return last_turn


class DummyGuardrails:
    def validate_text_output(self, text: str) -> str:
        return text
