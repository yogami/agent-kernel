"""Deterministic Finite State Machine Execution Engine."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import time
from typing import Any

from core.context_ram import ContextRAM
from core.guardrails import OutputGuardrails
from domain.models import (
    ConfigPack,
    ExecutionContext,
    ToolCall,
    ToolResult,
    Turn,
)
from domain.ports import EpisodeStorePort, LLMProviderPort
from domain.state_machine import (
    BudgetExceededError,
    KernelState,
    LoopDetectedError,
    can_transition,
)
from tools.registry import ToolRegistry


class ExecutionEngine:
    """Step-budgeted Finite State Machine executing agent interactions with hard limits."""

    def __init__(
        self,
        llm: LLMProviderPort,
        episode_store: EpisodeStorePort,
        tool_registry: ToolRegistry,
        context_ram: ContextRAM,
        guardrails: OutputGuardrails | None = None,
    ) -> None:
        self.llm = llm
        self.episode_store = episode_store
        self.tool_registry = tool_registry
        self.context_ram = context_ram
        self.guardrails = guardrails or OutputGuardrails()

    def run_turn(
        self,
        user_input: str,
        session_id: str,
        active_config: ConfigPack,
        turn_index: int = 0,
        max_steps: int = 5,
        cost_budget_usd: float = 0.50,
    ) -> Turn:
        """Execute a single multi-step interaction turn through the state machine."""
        start_time = time.time()
        ctx = ExecutionContext(
            session_id=session_id,
            step_index=0,
            max_steps=max_steps,
            cost_budget_usd=cost_budget_usd,
        )

        current_state = KernelState.COMPOSE_CONTEXT
        executed_tool_calls: list[ToolCall] = []
        executed_tool_results: list[ToolResult] = []
        accumulated_tokens_in = 0
        accumulated_tokens_out = 0
        final_model_output = ""

        # Fetch recent turns for episodic context
        recent_turns = self.episode_store.get_recent_turns(session_id, limit=5)

        while current_state not in {KernelState.COMPLETED, KernelState.FAILED}:
            # Check step budget
            if ctx.step_index >= ctx.max_steps:
                raise BudgetExceededError(
                    f"Execution exceeded hard step budget of {ctx.max_steps} steps.",
                    {"step_index": ctx.step_index, "max_steps": ctx.max_steps},
                )

            # Check dollar cost budget
            if ctx.cost_usd >= ctx.cost_budget_usd:
                raise BudgetExceededError(
                    f"Execution exceeded dollar budget of ${ctx.cost_budget_usd:.4f}.",
                    {"cost_usd": ctx.cost_usd, "budget": ctx.cost_budget_usd},
                )

            if current_state == KernelState.COMPOSE_CONTEXT:
                messages = self.context_ram.compose_context(
                    user_input=user_input,
                    recent_turns=recent_turns,
                    active_config=active_config,
                    session_id=session_id,
                )
                current_state = KernelState.MODEL_CALL

            elif current_state == KernelState.MODEL_CALL:
                ctx.step_index += 1
                tool_schemas = self.tool_registry.get_schemas()
                llm_resp = self.llm.generate(
                    messages=messages,
                    tools=tool_schemas if tool_schemas else None,
                )

                tokens_in = llm_resp.get("tokens_in", 0)
                tokens_out = llm_resp.get("tokens_out", 0)
                step_cost = llm_resp.get("cost_usd", 0.0)

                accumulated_tokens_in += tokens_in
                accumulated_tokens_out += tokens_out
                ctx.cost_usd += step_cost
                ctx.accumulated_tokens += (tokens_in + tokens_out)

                raw_tool_calls = llm_resp.get("tool_calls", [])
                content = llm_resp.get("content", "")

                if raw_tool_calls:
                    # Parse tool calls
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

                    # Loop detection check
                    for tc in pending_calls:
                        sig = f"{tc.tool_name}:{json.dumps(tc.arguments, sort_keys=True)}"
                        if ctx.executed_tool_signatures.count(sig) >= 2:
                            raise LoopDetectedError(
                                f"Repeated execution loop detected on tool signature '{tc.tool_name}'.",
                                {"signature": sig, "repeat_count": ctx.executed_tool_signatures.count(sig)},
                            )
                        ctx.executed_tool_signatures.append(sig)

                    # Transition to tool execution
                    current_state = KernelState.VALIDATE_TOOL_CALL

                else:
                    # Model produced direct text output
                    final_model_output = content
                    current_state = KernelState.VALIDATE_OUTPUT

            elif current_state == KernelState.VALIDATE_TOOL_CALL:
                # Validation passes, transition to execute
                current_state = KernelState.EXECUTE_TOOL

            elif current_state == KernelState.EXECUTE_TOOL:
                for tc in pending_calls:
                    executed_tool_calls.append(tc)
                    res = self.tool_registry.execute(tc.tool_name, tc.arguments)
                    res.tool_id = tc.tool_id
                    executed_tool_results.append(res)

                    # Append tool result to working context
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [{
                            "id": tc.tool_id,
                            "type": "function",
                            "function": {
                                "name": tc.tool_name,
                                "arguments": json.dumps(tc.arguments),
                            },
                        }],
                    })
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.tool_id,
                        "content": json.dumps(res.output) if not res.is_error else f"Error: {res.error_message}",
                    })

                # Loop back to model call with tool outputs
                current_state = KernelState.MODEL_CALL

            elif current_state == KernelState.VALIDATE_OUTPUT:
                # Run through guardrail firewall
                sanitized_output = self.guardrails.validate_text_output(final_model_output)
                final_model_output = sanitized_output
                current_state = KernelState.PERSIST_EPISODE

            elif current_state == KernelState.PERSIST_EPISODE:
                elapsed_ms = (time.time() - start_time) * 1000.0
                turn = Turn(
                    session_id=session_id,
                    turn_index=turn_index,
                    user_input=user_input,
                    model_output=final_model_output,
                    tool_calls=executed_tool_calls,
                    tool_results=executed_tool_results,
                    tokens_in=accumulated_tokens_in,
                    tokens_out=accumulated_tokens_out,
                    cost_usd=ctx.cost_usd,
                    latency_ms=elapsed_ms,
                    created_at=datetime.now(timezone.utc),
                )
                self.episode_store.append_turn(turn)
                current_state = KernelState.COMPLETED
                return turn

        raise RuntimeError(f"State machine terminated in unexpected state '{current_state}'.")
