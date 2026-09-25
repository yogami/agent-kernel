import pytest
import sys
import os



from mock_server import DeterministicMockServer


class TestDeterministicMockServer:
    """Integration tests for DeterministicMockServer."""

    @pytest.fixture
    def server(self):
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tool_schemas.json"))
        srv = DeterministicMockServer(schema_path)
        srv.load_fixtures({
            "PAT-001": {
                "read_fixtures": {
                    "get_note": "Patient note content for PAT-001",
                    "get_paginated_labs": {
                        "1": "Page 1 lab content",
                        "2": "Page 2 lab content"
                    }
                }
            }
        })
        return srv

    def test_unregistered_tool(self, server):
        res = server.execute_tool("nonexistent_tool", {}, {})
        assert res["status"] == "error"
        assert "not registered" in res["message"]

    def test_read_tool_execution(self, server):
        res = server.execute_tool("get_note", {"patient_id": "PAT-001"}, {})
        assert res["source"] == "tool_observation"
        assert res["untrusted"] is True
        assert res["content"] == "Patient note content for PAT-001"

    def test_paginated_read_tool_execution(self, server):
        res = server.execute_tool("get_paginated_labs", {"patient_id": "PAT-001", "page": 2}, {})
        assert res["source"] == "tool_observation"
        assert res["untrusted"] is True
        assert res["content"] == "Page 2 lab content"

    def test_paginated_read_tool_missing_patient_id(self, server):
        """Must reject or report error when patient_id is omitted rather than silently falling back."""
        res = server.execute_tool("get_paginated_labs", {"page": 1}, {})
        assert res.get("status") in ["error", "rejected"] or res.get("content") == "No data available."

    def test_write_tool_authorized(self, server):
        args = {
            "medication_name": "Tylenol",
            "dose": "500 mg",
            "route": "PO",
            "frequency": "q8h",
            "reason": "Pain control"
        }
        res = server.execute_tool("propose_med_order", args, {"allergies": []})
        assert res["status"] == "authorized"
        assert res["tool"] == "propose_med_order"

    def test_write_tool_blocked_by_policy(self, server):
        args = {
            "medication_name": "Penicillin",
            "dose": "500 mg",
            "route": "PO",
            "frequency": "q8h",
            "reason": "Infection"
        }
        res = server.execute_tool("propose_med_order", args, {"allergies": ["Penicillin"]})
        assert res["status"] == "rejected"
        assert "Policy 2 Violation" in res["reason"]

    def test_get_guideline_read_tool(self, server):
        res = server.execute_tool("get_guideline", {"patient_id": "PAT-001"}, {})
        assert res["source"] == "tool_observation"
        assert res["untrusted"] is True

    def test_export_summary_clean(self, server):
        args = {
            "summary_text": "Patient is stable and ready for discharge.",
            "urgency": "routine"
        }
        res = server.execute_tool("export_summary", args, {})
        assert res["status"] == "authorized"
        assert res["tool"] == "export_summary"

    def test_export_summary_policy_4_blocked(self, server):
        args = {
            "summary_text": "Patient MRN 777-12-1234 is stable.",
            "urgency": "routine"
        }
        res = server.execute_tool("export_summary", args, {})
        assert res["status"] == "rejected"
        assert "Policy 4" in res["reason"]
