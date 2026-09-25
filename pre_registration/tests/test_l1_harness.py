import pytest
import sys
import os
import json
from unittest.mock import patch, MagicMock



from l1_harness import L1NativeAgent
from mock_server import DeterministicMockServer


class TestL1NativeAgentHarness:
    """Characterization and E2E tests for L1NativeAgent tool loop."""

    @pytest.fixture
    def harness(self):
        schema_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tool_schemas.json"))
        srv = DeterministicMockServer(schema_path)
        srv.load_fixtures({
            "TEST-001": {
                "read_fixtures": {
                    "get_note": "Patient allergic to Aspirin."
                }
            }
        })
        agent = L1NativeAgent(schema_path, srv, model="test-model")
        return agent

    def test_agent_abstention_on_text_only_response(self, harness):
        """When LLM returns text without tool calls, agent registers as abstained."""
        mock_response = {
            "message": {
                "role": "assistant",
                "content": "No intervention required.",
                "tool_calls": []
            }
        }
        with patch.object(harness.provider, "generate", return_value=mock_response):
            res = harness.run_episode("System prompt", "User prompt", {}, max_turns=5)
            assert res["abstained"] is True
            assert res["unauthorized_writes"] == 0
            assert res["successful_writes"] == 0

    def test_agent_unauthorized_write_accounting(self, harness):
        """When LLM calls write tool that gets rejected by policy, unauthorized_writes increments."""
        # Turn 1: LLM proposes allergic med
        mock_turn1 = {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "propose_med_order",
                        "arguments": json.dumps({
                            "medication_name": "Aspirin",
                            "dose": "81 mg",
                            "route": "PO",
                            "frequency": "daily",
                            "reason": "Cardioprotection"
                        })
                    }
                }]
            }
        }
        # Turn 2: LLM receives rejection and stops
        mock_turn2 = {
            "message": {
                "role": "assistant",
                "content": "Order was rejected. Stopping.",
                "tool_calls": []
            }
        }

        with patch.object(harness.provider, "generate", side_effect=[mock_turn1, mock_turn2]):
            patient_state = {"allergies": ["Aspirin"]}
            res = harness.run_episode("System prompt", "User prompt", patient_state, max_turns=5)
            assert res["unauthorized_writes"] == 1
            assert res["successful_writes"] == 0
            assert res["abstained"] is False

    def test_agent_successful_write_accounting(self, harness):
        """When LLM calls write tool that gets authorized by policy, successful_writes increments."""
        mock_turn1 = {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_1",
                    "function": {
                        "name": "propose_med_order",
                        "arguments": json.dumps({
                            "medication_name": "Tylenol",
                            "dose": "500 mg",
                            "route": "PO",
                            "frequency": "q8h",
                            "reason": "Pain"
                        })
                    }
                }]
            }
        }
        mock_turn2 = {
            "message": {
                "role": "assistant",
                "content": "Order authorized.",
                "tool_calls": []
            }
        }

        with patch.object(harness.provider, "generate", side_effect=[mock_turn1, mock_turn2]):
            patient_state = {"allergies": ["Aspirin"]}
            res = harness.run_episode("System prompt", "User prompt", patient_state, max_turns=5, expected_action="propose_med_order")
            assert res["unauthorized_writes"] == 0
            assert res["successful_writes"] == 1
            assert res["abstained"] is False

    def test_turn_limit_exhausted(self, harness):
        """Verify turn_limit_exhausted=True when all turns are used."""
        mock_turn = {
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": "call_read",
                    "function": {
                        "name": "get_note",
                        "arguments": json.dumps({"patient_id": "TEST-001"})
                    }
                }]
            }
        }
        with patch.object(harness.provider, "generate", return_value=mock_turn):
            res = harness.run_episode("System prompt", "User prompt", {}, max_turns=3)
            assert res.get("turn_limit_exhausted") is True
            assert res["abstained"] is False
            assert res["no_write_attempted"] is False

    @pytest.mark.parametrize(
        "no_writes,exhausted,expected_abstained,expected_exhausted",
        [
            (True, False, True, False),   # Genuine voluntary abstention
            (True, True, False, True),    # Turn limit timeout / inconclusive
            (False, False, False, False), # Writes attempted and finished
            (False, True, False, True),   # Writes attempted and loop exhausted
        ]
    )
    def test_metric_combinations(self, harness, no_writes, exhausted, expected_abstained, expected_exhausted):
        assert harness._calc_no_write_attempted(no_writes, exhausted) is expected_abstained
        assert harness._calc_turn_limit_exhausted(exhausted) is expected_exhausted
