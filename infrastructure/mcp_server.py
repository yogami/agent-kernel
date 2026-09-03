"""Anthropic Model Context Protocol (MCP) Server implementing JSON-RPC 2.0 tool transport."""

from __future__ import annotations

import json
import sys
from typing import Any
from pydantic import BaseModel, Field

from domain.ports import ToolPort
from tools.registry import ToolRegistry


class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | str | None = None
    result: Any | None = None
    error: dict[str, Any] | None = None


class MCPServer:
    """Model Context Protocol (MCP) Server wrapping the kernel tool registry."""

    def __init__(self, tool_registry: ToolRegistry, server_name: str = "agent-kernel-mcp", server_version: str = "1.0.0") -> None:
        self.tool_registry = tool_registry
        self.server_name = server_name
        self.server_version = server_version

    def handle_request(self, request_dict: dict[str, Any]) -> dict[str, Any]:
        """Dispatch JSON-RPC 2.0 method calls for MCP protocol."""
        req_id = request_dict.get("id")
        method = request_dict.get("method")
        params = request_dict.get("params", {})

        if not method:
            return JSONRPCResponse(id=req_id, error={"code": -32600, "message": "Invalid Request: method is required"}).model_dump(exclude_none=True)

        # 1. initialize
        if method == "initialize":
            return JSONRPCResponse(
                id=req_id,
                result={
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": self.server_name, "version": self.server_version},
                    "capabilities": {"tools": {}},
                },
            ).model_dump(exclude_none=True)

        # 2. tools/list
        if method == "tools/list":
            tools = self.tool_registry.list_tools()
            tool_definitions = []
            for t in tools:
                tool_definitions.append({
                    "name": t.name,
                    "description": t.description,
                    "inputSchema": t.schema,
                })
            return JSONRPCResponse(id=req_id, result={"tools": tool_definitions}).model_dump(exclude_none=True)

        # 3. tools/call
        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments", {})

            if not tool_name:
                return JSONRPCResponse(id=req_id, error={"code": -32602, "message": "Invalid params: 'name' is required"}).model_dump(exclude_none=True)

            try:
                result = self.tool_registry.execute(tool_name, tool_args)
                content_text = (
                    f"Execution Error: {result.error_message}"
                    if result.is_error and result.error_message
                    else json.dumps(result.output, indent=2)
                )
                return JSONRPCResponse(
                    id=req_id,
                    result={
                        "content": [{
                            "type": "text",
                            "text": content_text,
                        }],
                        "isError": result.is_error,
                    },
                ).model_dump(exclude_none=True)
            except Exception as exc:
                return JSONRPCResponse(
                    id=req_id,
                    result={
                        "content": [{"type": "text", "text": f"Execution Error: {str(exc)}"}],
                        "isError": True,
                    },
                ).model_dump(exclude_none=True)

        # 4. ping
        if method == "ping":
            return JSONRPCResponse(id=req_id, result={"status": "pong"}).model_dump(exclude_none=True)

        return JSONRPCResponse(id=req_id, error={"code": -32601, "message": f"Method '{method}' not found"}).model_dump(exclude_none=True)

    def run_stdio(self) -> None:
        """Run MCP JSON-RPC server reading from standard input and writing to standard output."""
        for line in sys.stdin:
            line_str = line.strip()
            if not line_str:
                continue
            try:
                req_data = json.loads(line_str)
                resp = self.handle_request(req_data)
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
            except Exception as exc:
                err_resp = {"jsonrpc": "2.0", "error": {"code": -32700, "message": f"Parse error: {str(exc)}"}}
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    from tools.clinical_tools import (
        ClinicalAssertionCheckerTool,
        DeidentifyTextTool,
        FHIRValidatorTool,
        MedicalOntologyMapperTool,
    )
    from tools.system_tools import CalculatorTool, DateValidatorTool

    registry = ToolRegistry()
    registry.register(DeidentifyTextTool())
    registry.register(MedicalOntologyMapperTool())
    registry.register(FHIRValidatorTool())
    registry.register(ClinicalAssertionCheckerTool())
    registry.register(CalculatorTool())
    registry.register(DateValidatorTool())

    server = MCPServer(registry)
    server.run_stdio()
