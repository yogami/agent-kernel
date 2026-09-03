"""Tests for Model Context Protocol (MCP) Server."""

import json
from infrastructure.mcp_server import MCPServer
from tools.clinical_tools import DeidentifyTextTool
from tools.registry import ToolRegistry
from tools.system_tools import CalculatorTool


def test_mcp_initialize_and_ping():
    """Verify standard MCP initialize and ping handshakes."""
    registry = ToolRegistry()
    server = MCPServer(registry)

    # Initialize
    init_req = {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    init_resp = server.handle_request(init_req)

    assert init_resp["id"] == 1
    assert init_resp["result"]["protocolVersion"] == "2024-11-05"
    assert "tools" in init_resp["result"]["capabilities"]

    # Ping
    ping_req = {"jsonrpc": "2.0", "id": 2, "method": "ping", "params": {}}
    ping_resp = server.handle_request(ping_req)
    assert ping_resp["result"]["status"] == "pong"


def test_mcp_tools_list():
    """Verify MCP tools/list returns registered tool schemas in valid format."""
    registry = ToolRegistry()
    registry.register(DeidentifyTextTool())
    registry.register(CalculatorTool())
    server = MCPServer(registry)

    req = {"jsonrpc": "2.0", "id": 10, "method": "tools/list", "params": {}}
    resp = server.handle_request(req)

    tools = resp["result"]["tools"]
    assert len(tools) == 2
    tool_names = [t["name"] for t in tools]
    assert "deidentify_clinical_text" in tool_names
    assert "calculate" in tool_names
    assert "inputSchema" in tools[0]


def test_mcp_tool_call_execution():
    """Verify MCP tools/call executes tool and returns structured text content."""
    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(DeidentifyTextTool())
    server = MCPServer(registry)

    # Call calculator
    calc_req = {
        "jsonrpc": "2.0",
        "id": 20,
        "method": "tools/call",
        "params": {
            "name": "calculate",
            "arguments": {"expression": "25 * 4 + 10"},
        },
    }
    calc_resp = server.handle_request(calc_req)

    assert calc_resp["id"] == 20
    content = calc_resp["result"]["content"][0]["text"]
    parsed_out = json.loads(content)
    assert parsed_out["result"] == 110.0
    assert calc_resp["result"]["isError"] is False


def test_mcp_unknown_tool_error():
    """Verify MCP tools/call handles non-existent tool gracefully."""
    registry = ToolRegistry()
    server = MCPServer(registry)

    req = {
        "jsonrpc": "2.0",
        "id": 30,
        "method": "tools/call",
        "params": {"name": "non_existent_tool", "arguments": {}},
    }
    resp = server.handle_request(req)
    assert resp["result"]["isError"] is True
    assert "Execution Error" in resp["result"]["content"][0]["text"]
