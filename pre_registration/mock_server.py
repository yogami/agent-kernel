import json
import logging
from typing import Dict, Any, Optional
try:
    from pre_registration.policy_catalog import evaluate_policy_catalog, PolicyViolation
    from pre_registration.constants import ToolStatus, ObservationSource
except ImportError:
    from policy_catalog import evaluate_policy_catalog, PolicyViolation
    from constants import ToolStatus, ObservationSource


class DeterministicMockServer:
    """
    Mock Environment for the Pre-registration Ablation Study.
    Enforces the frozen Policy Catalog (P) before executing write tools.
    All functions adhere to cyclomatic complexity <= 3.
    """

    def __init__(self, schemas_path: str):
        with open(schemas_path, 'r') as f:
            self.schemas = json.load(f)

        self.read_tools = {t["name"]: t for t in self.schemas.get("read_tools", [])}
        self.write_tools = {t["name"]: t for t in self.schemas.get("write_tools", [])}
        self.fixtures: Dict[str, Dict[str, Any]] = {}

    def load_fixtures(self, fixtures_dict: Dict[str, Dict[str, Any]]):
        self.fixtures = fixtures_dict

    def _get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        return self.read_tools.get(tool_name) or self.write_tools.get(tool_name)

    def _dispatch_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        if tool_name in self.read_tools:
            return self._handle_read_tool(tool_name, args)
        return self._handle_write_tool(tool_name, args)

    def _execute_with_policy(self, tool_name: str, args: Dict[str, Any], current_state: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        try:
            evaluate_policy_catalog(tool_name, args, current_state, schema)
            return self._dispatch_tool(tool_name, args)
        except PolicyViolation as e:
            logging.warning(f"BLOCKED BY POLICY: {str(e)}")
            return {"status": ToolStatus.REJECTED.value, "reason": str(e)}

    def execute_tool(self, tool_name: str, args: Dict[str, Any], current_state: Dict[str, Any], skip_policy: bool = False) -> Dict[str, Any]:
        schema = self._get_tool_schema(tool_name)
        if not schema:
            return {"status": ToolStatus.ERROR.value, "message": f"Tool '{tool_name}' not registered."}
        if skip_policy:
            return self._dispatch_tool(tool_name, args)
        return self._execute_with_policy(tool_name, args, current_state, schema)


    def _extract_page_content(self, content: Any, args: Dict[str, Any]) -> Any:
        if isinstance(content, dict) and "page" in args:
            return content.get(str(args["page"]), "No data available.")
        return content

    def _handle_read_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        patient_id = args.get("patient_id")
        if not patient_id:
            return {"status": ToolStatus.ERROR.value, "message": f"Read tool '{tool_name}' requires 'patient_id'."}

        patient_data = self.fixtures.get(patient_id, {})
        raw_content = patient_data.get("read_fixtures", {}).get(tool_name, "No data available.")
        content = self._extract_page_content(raw_content, args)

        return {
            "source": ObservationSource.TOOL_OBSERVATION.value,
            "untrusted": True,
            "content": content
        }

    def _handle_write_tool(self, tool_name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "status": ToolStatus.AUTHORIZED.value,
            "tool": tool_name,
            "recorded_args": args
        }
