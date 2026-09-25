"""Standalone subprocess worker executing tool payloads with resource limits and firewalling."""

from __future__ import annotations

import importlib
import json
import os
import sys
from typing import Any

from core.sandbox.command_guard import inspect_arguments_for_command_injection
from core.sandbox.network_guard import network_egress_firewall
from core.sandbox.path_guard import filesystem_path_confinement, inspect_arguments_for_path_violations


def apply_resource_limits(max_memory_mb: int = 256) -> None:
    """Apply POSIX memory limits if supported by the operating system."""
    try:
        import resource
        limit_bytes = max_memory_mb * 1024 * 1024
        if hasattr(resource, "RLIMIT_AS"):
            resource.setrlimit(resource.RLIMIT_AS, (limit_bytes, limit_bytes))
    except Exception:
        # Ignore unsupported rlimit configurations on constrained test runtimes
        pass


def run_worker() -> None:
    """Read tool execution specification from stdin and output result JSON."""
    raw_input = sys.stdin.read()
    if not raw_input.strip():
        sys.stderr.write("Worker received empty input payload.\n")
        sys.exit(1)

    try:
        payload = json.loads(raw_input)
        tool_module = payload["tool_module"]
        tool_class_name = payload["tool_class"]
        arguments = payload.get("arguments", {})
        allow_network = payload.get("allow_network", False)
        max_memory_mb = payload.get("max_memory_mb", 256)
        allowed_paths = payload.get("allowed_paths", [])
        allow_shell = payload.get("allow_shell", False)

        apply_resource_limits(max_memory_mb)

        # Inspect arguments inside child worker
        inspect_arguments_for_path_violations(arguments, allowed_roots=allowed_paths)
        inspect_arguments_for_command_injection(arguments, allow_shell=allow_shell)

        # Import tool module dynamically
        mod = importlib.import_module(tool_module)
        tool_cls = getattr(mod, tool_class_name)
        tool_instance = tool_cls()

        # Run within network egress firewall and path confinement
        with network_egress_firewall(allow_network=allow_network), \
             filesystem_path_confinement(allowed_roots=allowed_paths):
            result = tool_instance.execute(arguments)

        output_data = {
            "tool_id": result.tool_id,
            "tool_name": result.tool_name,
            "output": result.output,
            "is_error": result.is_error,
            "error_message": result.error_message,
        }
        sys.stdout.write(json.dumps(output_data))
        sys.stdout.flush()
        sys.exit(0)

    except Exception as exc:
        err_data = {
            "tool_id": "err",
            "tool_name": payload.get("tool_class", "unknown") if "payload" in locals() else "unknown",
            "output": None,
            "is_error": True,
            "error_message": f"Sandbox worker failure: {str(exc)}",
        }
        sys.stdout.write(json.dumps(err_data))
        sys.stdout.flush()
        sys.exit(0)


if __name__ == "__main__":
    run_worker()
