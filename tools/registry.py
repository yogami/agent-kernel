"""Typed tool registry managing schemas and execution handlers."""

from __future__ import annotations

from typing import Any
from domain.models import ToolResult
from domain.ports import ToolPort


class ToolRegistry:
    """Registry mapping tool names to typed tool handlers and schemas."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolPort] = {}

    def register(self, tool: ToolPort) -> None:
        """Register a new tool instance."""
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolPort | None:
        """Fetch tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolPort]:
        """List all registered tools."""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Get OpenAI-compatible function calling schemas for all registered tools."""
        schemas: list[dict[str, Any]] = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                },
            })
        return schemas

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolResult:
        """Execute a tool by name with arguments and catch any unhandled exceptions."""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                tool_id="err",
                tool_name=tool_name,
                output=None,
                is_error=True,
                error_message=f"Tool '{tool_name}' not found in registry.",
            )
        try:
            return tool.execute(arguments)
        except Exception as exc:
            return ToolResult(
                tool_id="err",
                tool_name=tool_name,
                output=None,
                is_error=True,
                error_message=f"Execution error in '{tool_name}': {str(exc)}",
            )
