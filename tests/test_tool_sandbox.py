"""Unit and integration tests for tool sandboxing, hard timeouts, and egress firewalling."""

from __future__ import annotations

import asyncio
import os
import socket
import time
from typing import Any
import pytest

from core.sandbox.network_guard import NetworkSecurityViolationError, network_egress_firewall
from core.sandbox.runner import InProcessSandboxRunner, SubprocessSandboxRunner
from domain.models import ToolCapability, ToolResult
from domain.ports import ToolPort
from tools.registry import ToolRegistry
from tools.system_tools import CalculatorTool


class HangingTool(ToolPort):
    name = "hanging_tool"
    description = "Simulates an unresponsive or infinite looping tool."
    schema = {"type": "object", "properties": {}}
    capabilities = ToolCapability(timeout_seconds=0.2)

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        time.sleep(2.0)
        return ToolResult(tool_id="hang", tool_name=self.name, output="done")

    async def execute_async(self, arguments: dict[str, Any]) -> ToolResult:
        await asyncio.sleep(2.0)
        return ToolResult(tool_id="hang_async", tool_name=self.name, output="done")


class RogueNetworkTool(ToolPort):
    name = "rogue_network_tool"
    description = "Attempts unauthorized socket connection."
    schema = {"type": "object", "properties": {}}
    capabilities = ToolCapability(requires_network=False, timeout_seconds=2.0)

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect(("1.1.1.1", 80))
        finally:
            s.close()
        return ToolResult(tool_id="net", tool_name=self.name, output="connected")


class LegitimateNetworkTool(ToolPort):
    name = "legitimate_network_tool"
    description = "Tool with explicit network permissions."
    schema = {"type": "object", "properties": {}}
    capabilities = ToolCapability(requires_network=True, timeout_seconds=2.0)

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        # Should not raise NetworkSecurityViolationError
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.05)
            s.connect(("127.0.0.1", 65530))
            s.close()
        except NetworkSecurityViolationError:
            raise
        except Exception:
            # Normal socket failure (connection refused/timeout) is acceptable; security block is what we test against
            pass
        return ToolResult(tool_id="legit_net", tool_name=self.name, output={"authorized": True})


class EnvInspectorTool(ToolPort):
    name = "env_inspector_tool"
    description = "Inspects environment variables to verify credential stripping."
    schema = {"type": "object", "properties": {}}
    capabilities = ToolCapability(timeout_seconds=3.0)

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        leaked = [k for k in os.environ if any(bad in k.upper() for bad in ["SECRET", "KEY", "TOKEN", "PASSWORD"])]
        return ToolResult(
            tool_id="env",
            tool_name=self.name,
            output={"leaked_credentials": leaked, "in_sandbox": os.environ.get("AGENT_KERNEL_SANDBOX") == "1"},
        )


# ---------------------------------------------------------------------------
# 1. In-Process Sandbox Tests
# ---------------------------------------------------------------------------

def test_in_process_sandbox_timeout_sync():
    """Hanging tool must be aborted when deadline is reached."""
    runner = InProcessSandboxRunner()
    tool = HangingTool()

    start = time.perf_counter()
    res = runner.run(tool, {}, timeout_seconds=0.25)
    elapsed = time.perf_counter() - start

    assert res.is_error is True
    assert "Execution timeout" in res.error_message
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_in_process_sandbox_timeout_async():
    """Async execution must abort without hanging the event loop."""
    runner = InProcessSandboxRunner()
    tool = HangingTool()

    start = time.perf_counter()
    res = await runner.run_async(tool, {}, timeout_seconds=0.25)
    elapsed = time.perf_counter() - start

    assert res.is_error is True
    assert "Execution timeout" in res.error_message
    assert elapsed < 1.0


def test_in_process_sandbox_blocks_unauthorized_network():
    """Tool lacking network permission must be blocked from opening sockets."""
    runner = InProcessSandboxRunner()
    tool = RogueNetworkTool()

    res = runner.run(tool, {})
    assert res.is_error is True
    assert "Security violation" in res.error_message


def test_in_process_sandbox_allows_authorized_network():
    """Tool declaring network capability must pass without security violation."""
    runner = InProcessSandboxRunner()
    tool = LegitimateNetworkTool()

    res = runner.run(tool, {})
    assert res.is_error is False
    assert res.output == {"authorized": True}


# ---------------------------------------------------------------------------
# 2. Subprocess Sandbox Tests
# ---------------------------------------------------------------------------

def test_subprocess_sandbox_strips_environment_secrets(monkeypatch):
    """Subprocess runner must strip sensitive tokens and set AGENT_KERNEL_SANDBOX=1."""
    monkeypatch.setenv("MOCK_OPENAI_API_KEY", "sk-live-secret-key-12345")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "super-secret-aws-key")
    monkeypatch.setenv("NORMAL_CONFIG_VAR", "harmless_setting")

    runner = SubprocessSandboxRunner()
    tool = EnvInspectorTool()

    res = runner.run(tool, {})
    assert res.is_error is False
    assert res.output["in_sandbox"] is True
    assert "MOCK_OPENAI_API_KEY" not in res.output["leaked_credentials"]
    assert "AWS_SECRET_ACCESS_KEY" not in res.output["leaked_credentials"]


def test_subprocess_sandbox_timeout():
    """Subprocess runner must terminate workers that exceed hard deadline."""
    runner = SubprocessSandboxRunner()
    tool = HangingTool()

    start = time.perf_counter()
    res = runner.run(tool, {}, timeout_seconds=0.5)
    elapsed = time.perf_counter() - start

    assert res.is_error is True
    assert "Execution timeout" in res.error_message
    assert elapsed < 2.0


# ---------------------------------------------------------------------------
# 3. ToolRegistry Sandbox Integration
# ---------------------------------------------------------------------------

def test_tool_registry_with_sandbox_runner():
    """ToolRegistry executing tools through sandbox runner enforces guardrails seamlessly."""
    runner = InProcessSandboxRunner()
    registry = ToolRegistry(sandbox_runner=runner)

    # Register standard tool and rogue tool
    calc_tool = CalculatorTool()
    rogue_tool = RogueNetworkTool()
    hang_tool = HangingTool()

    registry.register(calc_tool)
    registry.register(rogue_tool)
    registry.register(hang_tool)

    # 1. Normal tool executes cleanly
    calc_res = registry.execute("calculate", {"expression": "40 + 2"})
    assert calc_res.is_error is False
    assert calc_res.output["result"] == 42

    # 2. Rogue tool is blocked by network firewall
    net_res = registry.execute("rogue_network_tool", {})
    assert net_res.is_error is True
    assert "Security violation" in net_res.error_message

    # 3. Hanging tool is terminated by timeout
    hang_res = registry.execute("hanging_tool", {})
    assert hang_res.is_error is True
    assert "Execution timeout" in hang_res.error_message


@pytest.mark.asyncio
async def test_tool_registry_async_sandbox_execution():
    """ToolRegistry execute_async handles sandboxed tools properly."""
    runner = InProcessSandboxRunner()
    registry = ToolRegistry(sandbox_runner=runner)

    calc_tool = CalculatorTool()
    hang_tool = HangingTool()
    registry.register(calc_tool)
    registry.register(hang_tool)

    # Normal async execution
    calc_res = await registry.execute_async("calculate", {"expression": "15 * 3"})
    assert calc_res.is_error is False
    assert calc_res.output["result"] == 45

    # Async timeout
    hang_res = await registry.execute_async("hanging_tool", {})
    assert hang_res.is_error is True
    assert "Execution timeout" in hang_res.error_message
