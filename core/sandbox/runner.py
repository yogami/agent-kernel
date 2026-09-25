"""Sandbox execution runners enforcing timeouts, process isolation, and egress security."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import json
import os
import subprocess
import sys
from typing import Any

from core.sandbox.command_guard import (
    CommandSecurityViolationError,
    inspect_arguments_for_command_injection,
)
from core.sandbox.network_guard import (
    NetworkSecurityViolationError,
    network_egress_firewall,
)
from core.sandbox.path_guard import (
    PathSecurityViolationError,
    filesystem_path_confinement,
    inspect_arguments_for_path_violations,
)
from domain.models import ToolCapability, ToolResult
from domain.ports import ToolPort, ToolSandboxPort


def get_tool_capabilities(tool: ToolPort) -> ToolCapability:
    """Extract declared capabilities or fallback to safe restrictive defaults."""
    if hasattr(tool, "capabilities") and isinstance(tool.capabilities, ToolCapability):
        return tool.capabilities
    return ToolCapability()


class InProcessSandboxRunner(ToolSandboxPort):
    """In-process sandbox enforcing execution deadlines and network egress firewalling."""

    def __init__(self, default_timeout_s: float = 5.0) -> None:
        self.default_timeout_s = default_timeout_s
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="sandbox-worker")

    def run(
        self,
        tool: ToolPort,
        arguments: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        """Execute a tool with network firewalling and a strict thread deadline."""
        caps = get_tool_capabilities(tool)
        deadline = timeout_seconds if timeout_seconds is not None else caps.timeout_seconds

        # Pre-flight argument checks for path traversal and command injection
        try:
            inspect_arguments_for_path_violations(arguments, allowed_roots=caps.allowed_paths)
            inspect_arguments_for_command_injection(arguments, allow_shell=caps.allow_shell)
        except (PathSecurityViolationError, CommandSecurityViolationError, PermissionError) as sec_err:
            return ToolResult(
                tool_id="security_violation",
                tool_name=tool.name,
                output=None,
                is_error=True,
                error_message=f"Security violation: {str(sec_err)}",
            )

        def _guarded_call() -> ToolResult:
            with network_egress_firewall(allow_network=caps.requires_network), \
                 filesystem_path_confinement(allowed_roots=caps.allowed_paths):
                return tool.execute(arguments)

        future = self._pool.submit(_guarded_call)
        try:
            return future.result(timeout=deadline)
        except FutureTimeoutError:
            return ToolResult(
                tool_id="timeout",
                tool_name=tool.name,
                output=None,
                is_error=True,
                error_message=f"Execution timeout: Tool '{tool.name}' exceeded hard deadline of {deadline:.2f}s.",
            )
        except (NetworkSecurityViolationError, PathSecurityViolationError, CommandSecurityViolationError, PermissionError) as sec_err:
            return ToolResult(
                tool_id="security_violation",
                tool_name=tool.name,
                output=None,
                is_error=True,
                error_message=f"Security violation: {str(sec_err)}",
            )
        except Exception as exc:
            return ToolResult(
                tool_id="err",
                tool_name=tool.name,
                output=None,
                is_error=True,
                error_message=f"Execution failure in '{tool.name}': {str(exc)}",
            )

    async def run_async(
        self,
        tool: ToolPort,
        arguments: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        """Execute a tool asynchronously under network firewall and asyncio deadline."""
        caps = get_tool_capabilities(tool)
        deadline = timeout_seconds if timeout_seconds is not None else caps.timeout_seconds

        # Pre-flight argument checks
        try:
            inspect_arguments_for_path_violations(arguments, allowed_roots=caps.allowed_paths)
            inspect_arguments_for_command_injection(arguments, allow_shell=caps.allow_shell)
        except (PathSecurityViolationError, CommandSecurityViolationError, PermissionError) as sec_err:
            return ToolResult(
                tool_id="security_violation",
                tool_name=tool.name,
                output=None,
                is_error=True,
                error_message=f"Security violation: {str(sec_err)}",
            )

        async def _guarded_async_call() -> ToolResult:
            with network_egress_firewall(allow_network=caps.requires_network), \
                 filesystem_path_confinement(allowed_roots=caps.allowed_paths):
                if hasattr(tool, "execute_async") and callable(getattr(tool, "execute_async")):
                    return await tool.execute_async(arguments)
                return await asyncio.to_thread(tool.execute, arguments)

        try:
            return await asyncio.wait_for(_guarded_async_call(), timeout=deadline)
        except asyncio.TimeoutError:
            return ToolResult(
                tool_id="timeout",
                tool_name=tool.name,
                output=None,
                is_error=True,
                error_message=f"Execution timeout: Tool '{tool.name}' exceeded hard deadline of {deadline:.2f}s.",
            )
        except (NetworkSecurityViolationError, PathSecurityViolationError, CommandSecurityViolationError, PermissionError) as sec_err:
            return ToolResult(
                tool_id="security_violation",
                tool_name=tool.name,
                output=None,
                is_error=True,
                error_message=f"Security violation: {str(sec_err)}",
            )
        except Exception as exc:
            return ToolResult(
                tool_id="err",
                tool_name=tool.name,
                output=None,
                is_error=True,
                error_message=f"Execution failure in '{tool.name}': {str(exc)}",
            )


class SubprocessSandboxRunner(ToolSandboxPort):
    """Subprocess sandbox isolating memory, stripping parent secrets, and enforcing timeouts."""

    def __init__(self, python_executable: str | None = None) -> None:
        self.python_bin = python_executable or sys.executable

    def _sanitize_env(self) -> dict[str, str]:
        """Strip sensitive credentials and tokens from child process environment."""
        forbidden_substrings = ["KEY", "SECRET", "PASSWORD", "TOKEN", "CREDENTIAL", "AUTH"]
        sanitized: dict[str, str] = {}
        for k, v in os.environ.items():
            if not any(bad in k.upper() for bad in forbidden_substrings):
                sanitized[k] = v
        sanitized["AGENT_KERNEL_SANDBOX"] = "1"
        return sanitized

    def run(
        self,
        tool: ToolPort,
        arguments: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        """Spawn worker subprocess to execute tool payload with stdin pipe."""
        caps = get_tool_capabilities(tool)
        deadline = timeout_seconds if timeout_seconds is not None else caps.timeout_seconds

        # Pre-flight argument checks
        try:
            inspect_arguments_for_path_violations(arguments, allowed_roots=caps.allowed_paths)
            inspect_arguments_for_command_injection(arguments, allow_shell=caps.allow_shell)
        except (PathSecurityViolationError, CommandSecurityViolationError, PermissionError) as sec_err:
            return ToolResult(
                tool_id="security_violation",
                tool_name=tool.name,
                output=None,
                is_error=True,
                error_message=f"Security violation: {str(sec_err)}",
            )

        payload = {
            "tool_module": tool.__class__.__module__,
            "tool_class": tool.__class__.__name__,
            "arguments": arguments,
            "allow_network": caps.requires_network,
            "max_memory_mb": caps.max_memory_mb,
            "allowed_paths": [str(p) for p in caps.allowed_paths],
            "allow_shell": caps.allow_shell,
        }

        cmd = [self.python_bin, "-m", "core.sandbox.worker"]
        env = self._sanitize_env()

        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
            )
            stdout, stderr = proc.communicate(input=json.dumps(payload), timeout=deadline)

            if proc.returncode != 0:
                return ToolResult(
                    tool_id="err",
                    tool_name=tool.name,
                    output=None,
                    is_error=True,
                    error_message=f"Subprocess terminated with code {proc.returncode}: {stderr.strip()}",
                )

            data = json.loads(stdout)
            return ToolResult(**data)

        except subprocess.TimeoutExpired:
            proc.kill()
            return ToolResult(
                tool_id="timeout",
                tool_name=tool.name,
                output=None,
                is_error=True,
                error_message=f"Execution timeout: Subprocess exceeded hard deadline of {deadline:.2f}s.",
            )
        except Exception as exc:
            return ToolResult(
                tool_id="err",
                tool_name=tool.name,
                output=None,
                is_error=True,
                error_message=f"Sandbox runner error: {str(exc)}",
            )

    async def run_async(
        self,
        tool: ToolPort,
        arguments: dict[str, Any],
        timeout_seconds: float | None = None,
    ) -> ToolResult:
        """Asynchronously delegate to subprocess worker."""
        return await asyncio.to_thread(self.run, tool, arguments, timeout_seconds)
